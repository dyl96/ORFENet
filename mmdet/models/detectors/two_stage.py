# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector

import torch.nn as nn

from torch.nn import functional as F

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution + batch norm + relu"""
    return torch.nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU()
    )

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class BasicIRNet(nn.Module):
    """
    Implementation based on methods from the AIM 2022 Challenge on
    Efficient and Accurate Quantized Image Super-Resolution on Mobile NPUs
    https://arxiv.org/pdf/2211.05910.pdf
    """

    def __init__(self,
                 in_plane,
                 upscale) -> None:
        super(BasicIRNet, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(in_plane, in_plane, 3, padding=1)
        )

        self.body = nn.ModuleList()
        self.num_upsample = 2 if upscale is 4 else 3
        for i in range(self.num_upsample):
            self.body.append(nn.Sequential(
                conv3x3(int(in_plane/2**i), int(in_plane / 2**(i+1))),
                conv3x3(int(in_plane / 2**(i+1)), int(in_plane / 2**(i+1))),
                nn.BatchNorm2d(int(in_plane / 2**(i+1))),
                nn.ReLU()))
                

        self.end = nn.Sequential(
            nn.Conv2d(int(in_plane / 2**(self.num_upsample)), int(in_plane / 2**(self.num_upsample)), kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.Conv2d(int(in_plane / 2**(self.num_upsample)), 2, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False))

    def forward(self, x):

        x = self.head(x)
        for i in range(self.num_upsample):
            x = resize(self.body[i](x), scale_factor=(2, 2), mode='bilinear')
        out = self.end(x)
        return out

@DETECTORS.register_module()
class TwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 add_or=False,
                 add_hrfe=False,
                 weight_or=1):
        super(TwoStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # add by ldy
        if add_or:
            self.add_or = True
            # self.branch_ir = self.build_ir(num_channels=256, num_feats=48, upscale=bbox_head['strides'][0], num_blocks=1)
            self.branch_ir = BasicIRNet(in_plane=neck['out_channels'], upscale=rpn_head['anchor_generator']['strides'][0])
            self.loss_or = nn.CrossEntropyLoss()
            self.weight_or = weight_or

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        # TODO：adopte R Module and add a image reconstruction loss
        if hasattr(self, 'add_or'):
            object_maps = self.build_target_obj(gt_bboxes, img_metas)
            loss_or = self.loss_reconstruction(x[0], object_maps)
            losses.update(loss_or)

        return losses

    # 构建目标mask
    def build_target_obj(self, gt_bboxes, img_metas):
        # build object map
        list_object_maps = []
        for i, gt_bbox in enumerate(gt_bboxes):
            object_map = torch.zeros(img_metas[0]["pad_shape"][0:2], device=gt_bboxes[0].device)
            for index in range(gt_bbox.shape[0]):
                gt = gt_bbox[index]
                # 宽和高都小于64为条件
                if (int(gt[2]) - int(gt[0])) <= 64 and (int(gt[3]) - int(gt[1])) <= 64:
                    object_map[int(gt[1]):(int(gt[3]) + 1), int(gt[0]):(int(gt[2]) + 1)] = 1

            list_object_maps.append(object_map[None])

        object_maps = torch.cat(list_object_maps, dim=0)
        return object_maps.long()

    # TODO：define a construction image function by ldy
    def loss_reconstruction(self, x, img):
        """
        Args:
            x (Tensor): the frature map used for reconstruction img
            img (Tensor): Input images of shape (N, C, H, W).
        Returns:
            dict[str, Tensor]: A dictionary of reconstruction loss.
        """
        loss = dict()
        x = self.branch_ir(x)
        loss_rec = self.weight_or * self.loss_or(x, img)
        loss['loss_or'] = loss_rec
        return loss

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def onnx_export(self, img, img_metas):

        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        x = self.extract_feat(img)
        proposals = self.rpn_head.onnx_export(x, img_metas)
        if hasattr(self.roi_head, 'onnx_export'):
            return self.roi_head.onnx_export(x, proposals, img_metas)
        else:
            raise NotImplementedError(
                f'{self.__class__.__name__} can not '
                f'be exported to ONNX. Please refer to the '
                f'list of supported models,'
                f'https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html#list-of-supported-models-exportable-to-onnx'  # noqa E501
            )
