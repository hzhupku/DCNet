from maskrcnn_benchmark.data.datasets.voc import PascalVOCDataset
import sys
# arg 1: pathdir;
for split in range(1, 4):
    for shot in [1, 2, 3, 5, 10]:
        all_cls = PascalVOCDataset.CLASSES
        novel_cls = [PascalVOCDataset.CLASSES_SPLIT1_NOVEL, 
                     PascalVOCDataset.CLASSES_SPLIT2_NOVEL,
                     PascalVOCDataset.CLASSES_SPLIT3_NOVEL,][split - 1]
        novel_index = [all_cls.index(c) for c in novel_cls]
        AP = [0] * 5
        try:
            with open(sys.argv[1] + '/result_split%d_%dshot.txt'%(split, shot), 'r') as f:
                content = f.readlines()
            for k, j in enumerate(novel_index):
                AP[k] += float(content[j][18 : 24])
            print("VOC split%d  %2dshot:novel map:%.4f"%(split, shot, sum(AP) / 5))
        except Exception as e:
            continue
