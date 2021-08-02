# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os, sys
import cv2

import torch
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug

def draw(pic_id, boxes, labels):
    pic = cv2.imread('datasets/voc/VOC2007/JPEGImages/' + pic_id, cv2.IMREAD_COLOR)
    for i in range(len(labels)):
        box = boxes[i]
        box = [int(b.item())for b in box]
        cv2.rectangle(pic, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
        cv2.putText(pic, labels[i], (box[0], box[1] - 3 * i), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 0, 0))
    cv2.imwrite('demopics/%s'%pic_id, pic)

def compute_on_dataset(model, data_loader, device, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    if not os.path.exists('demopics'):
        os.mkdir('demopics')
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        #if _ <= 0:
        #    continue
        with torch.no_grad():
            if timer:
                timer.tic()
            if cfg.TEST.BBOX_AUG.ENABLED:
                output = im_detect_bbox_aug(model, images, device)
            else:
                output = model(images.to(device), [target.to(device) for target in targets])
            if timer:
                if not cfg.MODEL.DEVICE == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        for i in range(len(image_ids)):
            pic_id = '%s.jpg'%data_loader.dataset.ids[image_ids[i]]
            image_info = data_loader.dataset.get_img_info(image_ids[i])
            pred = output[i].resize((image_info['width'], image_info['height']))
            boxes = pred.bbox
            labels = pred.extra_fields
            novel = [3, 6, 10, 14, 18]
            hasnovel = False
            for c in novel:
                if (labels['labels'] == c).sum() > 0:
                    hasnovel = True
            if not hasnovel:
                continue
            labels = ['%s: %.3f'%(data_loader.dataset.CLASSES[labels['labels'][s].item()], labels['scores'][s].item()) for s in range(len(labels['scores']))]
            draw(pic_id, boxes, labels)
        #if _ >= 100:
        #    sys.exit(0)
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
