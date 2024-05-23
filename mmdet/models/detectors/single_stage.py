# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector

from torch.nn import functional as F


class SRNet(nn.Module):
    """
    Implementation based on methods from the AIM 2022 Challenge on
    Efficient and Accurate Quantized Image Super-Resolution on Mobile NPUs
    https://arxiv.org/pdf/2211.05910.pdf
    """

    def __init__(self,
                 num_channels,
                 num_feats,
                 num_blocks,
                 upscale) -> None:
        super(SRNet, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(num_channels, num_feats, 3, padding=1)
        )

        body = []
        for i in range(num_blocks):
            body.append(nn.Conv2d(num_feats, num_feats, 3, padding=1))
            body.append(nn.ReLU(True))

        self.body = nn.Sequential(*body)

        self.upsample = nn.Sequential(
            nn.Conv2d(num_feats, 3 * (upscale ** 2), 3, padding=1),
            nn.PixelShuffle(upscale)
        )

    def forward(self, x):
        res = self.head(x)
        out = self.body(res)
        out = self.upsample(res + out)
        return out


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
            self.body.append(conv3x3(int(in_plane/2**i), int(in_plane / 2**(i+1))))

        self.end = nn.Conv2d(int(in_plane / 2**(self.num_upsample)), 2, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)

    def forward(self, x):

        x = self.head(x)
        for i in range(self.num_upsample):
            x = resize(self.body[i](x), scale_factor=(2, 2), mode='bilinear')
        out = self.end(x)
        return out


class HRFE(nn.Module):
    """
    A high resolution feature enhancement module for tiny object detection
    """

    def __init__(self,
                 in_channels,
                 num_blocks) -> None:
        """
        Args:
            in_channels: the channel of input feature map
            num_blocks: the nums of hrfe module
        """
        super(HRFE, self).__init__()

        body = []
        for i in range(num_blocks):
            body.append(nn.Conv2d(in_channels, in_channels, 3, padding=1))
            body.append(nn.BatchNorm2d(in_channels))
            body.append(nn.ReLU(True))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        out = self.body(x)
        return out

class MRFAFE(nn.Module):
    def __init__(self, in_channels, group, kernel_sizes=(3, 7, 21)):
        super(MRFAFE, self).__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_sizes[0],
                                               padding=kernel_sizes[0]//2),
                                     nn.BatchNorm2d(in_channels),
                                     nn.ReLU())
        self.branch2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_sizes[1],
                                               padding=kernel_sizes[1]//2, groups=in_channels),
                                     nn.BatchNorm2d(in_channels),
                                     nn.Conv2d(in_channels, in_channels, kernel_size=kernel_sizes[1],
                                               padding=kernel_sizes[1]//2, groups=in_channels),
                                     nn.BatchNorm2d(in_channels),
                                     nn.ReLU())
        self.branch3 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_sizes[2],
                                               padding=kernel_sizes[2] // 2, groups=in_channels),
                                     nn.BatchNorm2d(in_channels),
                                     nn.Conv2d(in_channels, in_channels, kernel_size=kernel_sizes[2],
                                               padding=kernel_sizes[2] // 2, groups=in_channels),
                                     nn.BatchNorm2d(in_channels),
                                     nn.ReLU())

        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)

        self.weight_module = Weight(3 * in_channels, group)
        self.in_channels = in_channels

    def forward(self, x):
        res = x
        x = torch.cat((self.branch1(x), self.branch2(x), self.branch3(x)), dim=1)
        weight = self.weight_module(x)
        x = x * weight
        x = x[:, 0:self.in_channels] + x[:, self.in_channels:2*self.in_channels] + \
            x[:, 2*self.in_channels:3*self.in_channels]
        x = self.conv1(x)
        return x + res


class Weight(nn.Module):

    def __init__(self, in_channels, group):
        super(Weight, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=group, kernel_size=(3, 3),
                                            padding=1, groups=group))
        self.softmax = nn.Softmax(dim=1)
        self.group = group
        self.repeat = int(in_channels/group)

    def forward(self, x):
        x = self.conv(x)
        x = torch.sum(x, (2, 3), keepdim=True)
        weight = self.softmax(x)
        weight = weight.repeat(1, self.repeat, 1, 1)
        weight = torch.cat(tuple((weight[:, i::self.group, :, :] for i in range(self.group))), dim=1)
        return weight


@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 add_or=False,
                 add_hrfe=False,
                 weight_or=10):
        super(SingleStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # add by ldy
        if add_or:
            self.add_or = True
            # self.branch_ir = self.build_ir(num_channels=256, num_feats=48, upscale=bbox_head['strides'][0], num_blocks=1)

            if bbox_head['type'] == 'RepPointsHead':
                self.branch_ir = BasicIRNet(in_plane=neck['out_channels'], upscale=bbox_head['point_strides'][0])
            elif bbox_head['type'] == 'FCOSHead':
                self.branch_ir = BasicIRNet(in_plane=neck['out_channels'], upscale=bbox_head['strides'][0])
            elif bbox_head['type'] == 'ATSSHead':
                self.branch_ir = BasicIRNet(in_plane=neck['out_channels'], upscale=bbox_head['anchor_generator']['strides'][0])
            else:
                self.branch_ir = BasicIRNet(in_plane=neck['out_channels'], upscale=bbox_head['strides'][0])

            self.loss_or = nn.CrossEntropyLoss()
            self.weight_or = weight_or

        if add_hrfe:
            self.add_hrfe = True
            self.branch_hrfe = MRFAFE(256, 3)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)

        if hasattr(self, 'add_hrfe'):
            x0 = x[0]
            x_ = x[1:]
            x0 = self.branch_hrfe(x0)
            x = (x0,) + x_
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)

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
            object_map = torch.zeros(img_metas[0]["batch_input_shape"], device=gt_bboxes[0].device)
            for index in range(gt_bbox.shape[0]):
                gt = gt_bbox[index]
                # 宽和高都小于64为条件
                if (int(gt[2])-int(gt[0])) <= 64 and (int(gt[3]) - int(gt[1])) <= 64:
                    object_map[int(gt[1]):(int(gt[3])+1), int(gt[0]):(int(gt[2])+1)] = 1

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

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas, with_nms=True):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape

        if len(outs) == 2:
            # add dummy score_factor
            outs = (*outs, None)
        # TODO Can we change to `get_bboxes` when `onnx_export` fail
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            *outs, img_metas, with_nms=with_nms)

        return det_bboxes, det_labels
