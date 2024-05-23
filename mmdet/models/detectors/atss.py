# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class ATSS(SingleStageDetector):
    """Implementation of `ATSS <https://arxiv.org/abs/1912.02424>`_."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 add_or=False,
                 add_hrfe=False,
                 weight_or=10):
        super(ATSS, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg, add_or=add_or, add_hrfe=add_hrfe, weight_or=weight_or)
