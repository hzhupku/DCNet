# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.structures.image_list import to_image_list
import torch


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0,meta=False):
        self.size_divisible = size_divisible
        self.meta = meta
        
    def __call__(self, batch):
        transposed_batch = list(zip(*batch))    
        if len(transposed_batch) == 3:
            images = to_image_list(transposed_batch[0], self.size_divisible)
            targets = transposed_batch[1]
            img_ids = transposed_batch[2]
            return images, targets, img_ids
        if len(transposed_batch) == 4:
            images = to_image_list(transposed_batch[0], self.size_divisible)
            targets = transposed_batch[1]
            img_ids = transposed_batch[2]
            for idx, level in enumerate(transposed_batch[3]):
                if level is None:
                    continue
                if idx==0:
                    origin = level.unsqueeze(0)
                else:
                    origin = torch.cat((origin,level.unsqueeze(0)),dim=1)

            return images, targets, img_ids, origin
        if len(transposed_batch) == 15 or len(transposed_batch) == 60:
            images = [torch.stack(image_per_level) for image_per_level in list(zip(*transposed_batch))]
            return images
        if len(transposed_batch) == 20 or len(transposed_batch) == 80:
            images = [torch.stack(image_per_level) for image_per_level in list(zip(*transposed_batch))]
            return images
        else:
            images = [torch.stack(image_per_level) for image_per_level in list(zip(*transposed_batch[0]))]
            targets = torch.cat(transposed_batch[1])
            return images, targets


class BBoxAugCollator(object):
    """
    From a list of samples from the dataset,
    returns the images and targets.
    Images should be converted to batched images in `im_detect_bbox_aug`
    """

    def __call__(self, batch):
        return list(zip(*batch))

