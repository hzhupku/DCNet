# Dense Relation Distillation with Context-aware Aggregation for Few-Shot Object Detection, CVPR 2021

Our code is based on  [https://github.com/facebookresearch/maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and developed with Python 3.6.5 & PyTorch 1.1.0.

## Abstract
Conventional deep learning based methods for object detection require a large amount of bounding box annotations
for training, which is expensive to obtain such high quality annotated data. Few-shot object detection, which learns
to adapt to novel classes with only a few annotated examples, is very challenging since the fine-grained feature of
novel object can be easily overlooked with only a few data
available. In this work, aiming to fully exploit features of
annotated novel object and capture fine-grained features of
query object, we propose Dense Relation Distillation with
Context-aware Aggregation (DCNet) to tackle the few-shot
detection problem. Built on the meta-learning based framework, Dense Relation Distillation module targets at fully exploiting support features, where support features and query
feature are densely matched, covering all spatial locations
in a feed-forward fashion. The abundant usage of the guidance information endows model the capability to handle
common challenges such as appearance changes and occlusions. Moreover, to better capture scale-aware features,
Context-aware Aggregation module adaptively harnesses
features from different scales for a more comprehensive feature representation. Extensive experiments illustrate that
our proposed approach achieves state-of-the-art results on
PASCAL VOC and MS COCO datasets. For more details, please refer to our CVPR paper ([arxiv](https://arxiv.org/pdf/2103.17115.pdf)). 


<div align=center>
<img src="https://github.com/hzhupku/DCNet/blob/master/tools/fewshot_exp/arch.PNG" width="600">
</div>

## Installation
Check INSTALL.md for installation instructions. Since maskrcnn-benchmark has been deprecated, please follow these instructions carefully (e.g. version of Python packages).

## Prepare datasets

### Prepare original Pascal VOC & MS COCO datasets
First, you need to download the VOC & COCO datasets.
We recommend to symlink the path of the datasets to `datasets/` as follows

We use `minival` and `valminusminival` sets from [Detectron](https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/data/README.md#coco-minival-annotations) ([filelink](https://dl.fbaipublicfiles.com/detectron/coco/coco_annotations_minival.tgz)).

```bash
mkdir -p datasets/coco
ln -s /path_to_coco_dataset/annotations datasets/coco/annotations
ln -s /path_to_coco_dataset/train2014 datasets/coco/train2014
ln -s /path_to_coco_dataset/test2014 datasets/coco/test2014
ln -s /path_to_coco_dataset/val2014 datasets/coco/val2014

ln -s /path_to_VOCdevkit_dir datasets/voc
```

### Prepare base and few-shot datasets
For multiple runs, you need to specify the seed in the script.
```bash
bash tools/fewshot_exp/datasets/init_fs_dataset_standard.sh
```
This will also generate the datasets on base classes for base training.

## Training and Evaluation
Scripts for training and evaluation on PASCAL VOC dataset.
```bash
experiments/DRD/
├── prepare_dataset.sh
├── base_train.sh
├── fine_tune.sh
└── get_result.sh
```

Configurations of base & few-shot experiments are:
```base
experiments/DRD/configs/
├── base
│   └── e2e_voc_split*_base.yaml
└── standard
    └── e2e_voc_split*_*shot_finetune.yaml
```
Modify them if needed. If you have any question about these parameters (e.g. batchsize), please refer to [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) for quick solutions.

### Perform few-shot training on VOC dataset
1. Run the following for base training on 3 VOC splits
```bash
cd experiments/DRD
bash base_train.sh
```
This will generate base models (e.g. `model_voc_split1_base.pth`) and corresponding pre-trained models (e.g. `voc0712_split1base_pretrained.pth`).

2. Run the following for few-shot fine-tuning
```bash
bash fine_tune.sh
```
This will perform evaluation on 1/2/3/5/10 shot of 3 splits. 
Result folder is `fs_exp/voc_standard_results` by default, and you can get a quick summary by:
```bash
bash get_result.sh
```

## Citation
```
@inproceedings{hu2021dense,
  title={Dense Relation Distillation with Context-aware Aggregation for Few-Shot Object Detection},
  author={Hu, Hanzhe and Bai, Shuai and Li, Aoxue and Cui, Jinshi and Wang, Liwei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10185--10194},
  year={2021}
}
```

#### TODO
- [ ] Context-aware Aggregation
- [ ] Training scripts on COCO dataset
