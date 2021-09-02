# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
import numpy as np
import skimage
import skimage.filters
import torch.nn.functional as F
import scipy.ndimage as nd
import cv2
from ..DRD import DenseRelationDistill
class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        
        
        self.dense_relation = cfg.MODEL.DENSE_RELATION
        self.dense_sum = cfg.MODEL.DENSE_SUM
        
        self.layer3 = cfg.MODEL.LAYER3
        self.layer1 = cfg.MODEL.LAYER1
        self.layer0 = cfg.MODEL.LAYER0
        self.maskpool = cfg.MODEL.MASKPOOL
        self.num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        if self.dense_relation:
            self.drd_opt = DenseRelationDistill(256,32,128,self.dense_sum)
        
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16384,1024)
        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print('froze backbone parameters')
        
        if cfg.MODEL.RPN.FREEZE:
            for p in self.rpn.parameters():
                p.requires_grad = False
            print('froze rpn parameters')

        if cfg.MODEL.ROI_BOX_HEAD.FREEZE_FEAT:
            for p in self.roi_heads.box.feature_extractor.parameters():
                p.requires_grad = False
            print('froze roi_box_head parameters')
    
    def meta_extractor(self,meta_data,dr=False):

        if self.layer3:
            base_feat = self.backbone((meta_data,1))[3]
        elif self.layer1:
            base_feat = self.backbone((meta_data,1))[1]
        elif self.layer0:
            base_feat = self.backbone((meta_data,1))[0]
        elif self.maskpool:
            base_feat = self.backbone((meta_data[:,:-1,:,:],0))[2]
            h, w = base_feat.shape[2:]
            mask = meta_data[:,-1,:,:].unsqueeze(1)
            mask = F.interpolate(mask,size=(h,w),mode='bilinear',align_corners=True)
            base_feat = base_feat*mask
        else:
            base_feat = self.backbone((meta_data,1))[2]
        if dr:
            return base_feat
     
        feature=self.fc1(self.maxpool(base_feat).view(base_feat.shape[0],-1))
        feature = self.sigmoid(feature) 
     
        return feature

    def forward(self, images, targets=None, meta_input=None, meta_attentions=None, average_shot=False):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        torch.cuda.empty_cache()
        if average_shot:
            attentions = self.meta_extractor(meta_input,dr=self.dense_relation)
            return attentions
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training and meta_input is not None:
            attentions = self.meta_extractor(meta_input,dr=self.dense_relation)
        if meta_attentions is not None:
            attentions = meta_attentions

        images = to_image_list(images)
        features = self.backbone((images.tensors,0))
        
        if self.dense_relation:
            features = self.drd_opt(features, attentions)

        proposals, proposal_losses = self.rpn(images, features, targets)
       
        if self.roi_heads:    
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}
        torch.cuda.empty_cache()
        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses
        
        return result
