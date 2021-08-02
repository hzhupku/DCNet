from maskrcnn_benchmark.data.datasets.voc import PascalVOCDataset
from collections import OrderedDict
import sys
#split = int(sys.argv[1])
for split in range(1, 4):
    for year in ["07", "12"]:
        dataset = PascalVOCDataset('datasets/voc/VOC20%s'%year, 'trainval')
        keeps = []
        all_cls = PascalVOCDataset.CLASSES
        novel_cls = [PascalVOCDataset.CLASSES_SPLIT1_NOVEL,
                     PascalVOCDataset.CLASSES_SPLIT2_NOVEL,
                     PascalVOCDataset.CLASSES_SPLIT3_NOVEL,][split - 1]
        novel_index = [all_cls.index(c) for c in novel_cls]
        print(novel_index)
        for i in range(len(dataset.ids)):
            anno = dataset.get_groundtruth(i)
            label = anno.get_field('labels')
            count = [(label == j).sum().item() for j in novel_index]
            if sum(count) == 0:
                keeps.append(i)
        
        box_count = [0] * 21
        for i in keeps:
            anno = dataset.get_groundtruth(i)
            label = anno.get_field('labels')
            for j in label:
                box_count[j] += 1
        print("trainval%s:%d"%(year, len(keeps)))
        print(dict(zip(all_cls, box_count)))
        with open('datasets/voc/VOC20%s/ImageSets/Main/trainval_split%d_base.txt'%(year, split), 'w+') as f:
            for i in keeps:
                f.write(dataset.ids[i] + '\n')

        if year == '12':
            continue
        dataset = PascalVOCDataset('datasets/voc/VOC20%s'%year, 'test', use_difficult=True)
        keeps = []
        all_cls = PascalVOCDataset.CLASSES
        novel_cls = [PascalVOCDataset.CLASSES_SPLIT1_NOVEL,
                     PascalVOCDataset.CLASSES_SPLIT2_NOVEL,
                     PascalVOCDataset.CLASSES_SPLIT3_NOVEL,][split - 1]
        novel_index = [all_cls.index(c) for c in novel_cls]
        print(novel_index)
        for i in range(len(dataset.ids)):
            anno = dataset.get_groundtruth(i)
            label = anno.get_field('labels')
            count = [(label == j).sum().item() for j in novel_index]
            if sum(count) == 0:
                keeps.append(i)
        
        box_count = [0] * 21
        for i in keeps:
            anno = dataset.get_groundtruth(i)
            label = anno.get_field('labels')
            for j in label:
                box_count[j] += 1
        print("test%s:%d"%(year, len(keeps)))
        print(dict(zip(all_cls, box_count)))
        with open('datasets/voc/VOC20%s/ImageSets/Main/test_split%d_base.txt'%(year, split), 'w+') as f:
            for i in keeps:
                f.write(dataset.ids[i] + '\n')

