import torch
import torch.utils.data
from PIL import Image
import sys
import cv2
import os
import os.path
import random
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

import collections
from maskrcnn_benchmark.structures.bounding_box import BoxList


class PascalVOCDataset_Meta(torch.utils.data.Dataset):

    CLASSES = (
        "__background__ ",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )

    CLASSES_SPLIT1_BASE = (
        "__background__ ",
        "aeroplane",
        "bicycle",
        "boat",
        "bottle",
        "car",
        "cat",
        "chair",
        "diningtable",
        "dog",
        "horse",
        "person",
        "pottedplant",
        "sheep",
        "train",
        "tvmonitor",
    )

    CLASSES_SPLIT2_BASE = (
        "__background__ ",
        "bicycle",
        "bird",
        "boat",
        "bus",
        "car",
        "cat",
        "chair",
        "diningtable",
        "dog",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "train",
        "tvmonitor",
    )

    CLASSES_SPLIT3_BASE = (
        "__background__ ",
        "aeroplane",
        "bicycle",
        "bird",
        "bottle",
        "bus",
        "car",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "person",
        "pottedplant",
        "train",
        "tvmonitor",
    )

    CLASSES_SPLIT1_NOVEL = (
        "bird",
        "bus",
        "cow",
        "motorbike",
        "sofa",
    )
    CLASSES_SPLIT2_NOVEL = (
        "aeroplane",
        "bottle",
        "cow",
        "horse",
        "sofa"
    )
    CLASSES_SPLIT3_NOVEL = (
        "boat",
        "cat",
        "motorbike",
        "sheep",
        "sofa",
    )

    def __init__(self, data_dir, split, use_difficult=False, transforms=None, toofew=True, shots=200, size=224, seed=0):

        # data_dir: "voc/VOC2007"  ,split: "trainval_split1_base"
        self.root = data_dir
        self.image_set = split
        self.keep_difficult = use_difficult
        self.transforms = transforms

        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")

        phase = 1
        if 'split1_base' in split:
            cls = PascalVOCDataset_Meta.CLASSES_SPLIT1_BASE
        elif 'split2_base' in split:
            cls = PascalVOCDataset_Meta.CLASSES_SPLIT2_BASE
        elif 'split3_base' in split:
            cls = PascalVOCDataset_Meta.CLASSES_SPLIT3_BASE
        else:
            phase = 2
            cls = PascalVOCDataset_Meta.CLASSES
        self.cls = cls[1:]
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.categories = dict(zip(range(len(cls)), cls))
        self.list_root = os.path.join('/workspace/code/Meta-FSOD','fs_list')
        if 'base' in split:
            fname = 'voc_traindict_full.txt'
            metafile = os.path.join(self.list_root,fname)
        else:
            fname = 'voc_traindict_bbox_'+str(shots)+'shot.txt'
            metafile = os.path.join(self.list_root,fname)
        if 'standard' in split and seed > 0:
            fname =  'voc_traindict_bbox_'+str(shots)+'shot_seed'+str(seed)+'.txt'
            metafile = os.path.join(self.list_root,fname)
        metainds = [[]] * len(self.cls)
        with open(metafile, 'r') as f:
            metafiles = []
            for line in f.readlines():
                pair = line.rstrip().split()
                if len(pair) == 2:
                    pass
                elif len(pair) == 4:
                    pair = [pair[0]+' '+pair[1], pair[2]+' '+pair[3]]
                else:
                    raise NotImplementedError('{} not recognized'.format(pair))
                metafiles.append(pair)
            metafiles = {k: v for k, v in metafiles}
            self.metalines = [[]] * len(self.cls)
            for i, clsname in enumerate(self.cls):
                with open(metafiles[clsname], 'r') as imgf:
                    lines = [l for l in imgf.readlines()]
                    self.metalines[i] = lines
                    if(shots>100):
                        self.metalines[i] = random.sample(self.metalines[i],shots)
        
        self.ids=[]
        self.img_size = size
        if (phase == 2):
            for j in range(len(self.cls)):
                self.metalines[j]=np.random.choice(self.metalines[j],shots*64).tolist()
            for i in range(shots*64):
                metaid=[]
                for j in range(len(self.cls)):
                    metaid.append([j,self.metalines[j][i].rstrip()])
                self.ids.append(metaid)
        else:
            for i in range(shots):
                metaid=[]
                for j in range(len(self.cls)):
                    metaid.append([j,self.metalines[j][i].rstrip()])
                self.ids.append(metaid)
            


    def __getitem__(self, index):
        img_ids = self.ids[index]
        data = []
        for cls_id, img_id in img_ids:
            
            img = cv2.imread(img_id, cv2.IMREAD_COLOR)
            img = img.astype(np.float32, copy=False)
            
            img -= np.array([[[102.9801, 115.9465, 122.7717]]])
            height, width, _ = img.shape
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            #img = Image.open(img_id).convert("RGB")
            mask = self.get_mask(img_id, cls_id, height, width)
            img = torch.from_numpy(img).unsqueeze(0)
            mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(3)
            imgmask = torch.cat([img,mask],dim=3)
            imgmask = imgmask.permute(0, 3, 1, 2).contiguous()
            data.append(imgmask)
        res = torch.cat(data,dim=0)

        
        return res

    def get_img_info(self, index):
        cls_id, img_id = self.ids[index][0]
        path = img_id.split('JPEG')[0]+'Annotations/'+img_id.split('/')[-1].split('.jpg')[0]+'.xml'
        anno = ET.parse(path).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def __len__(self):
        return len(self.ids)
    
    def get_mask(self, img_id, cls_id, height, width):
        mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        y_ration = float(height) / self.img_size
        x_ration = float(width) / self.img_size

        path = img_id.split('JPEG')[0]+'Annotations/'+img_id.split('/')[-1].split('.jpg')[0]+'.xml'
        target = ET.parse(path).getroot()
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if difficult:
                continue
            name = obj.find('name').text.strip()
            if (name != self.cls[cls_id]):
                continue
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(float(bbox.find(pt).text)) - 1
                if i % 2 == 0:
                    cur_pt = int(cur_pt / x_ration)
                    bndbox.append(cur_pt)
                elif i % 2 == 1:
                    cur_pt = int(cur_pt / y_ration)
                    bndbox.append(cur_pt)
            mask[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]] = 1
            break
        return mask

    def get_groundtruth(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno)

        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("difficult", anno["difficult"])
        return target

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1

        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.lower().strip()
            bb = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [
                bb.find("xmin").text,
                bb.find("ymin").text,
                bb.find("xmax").text,
                bb.find("ymax").text,
            ]
            bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, box)))
            )

            boxes.append(bndbox)
            gt_classes.append(self.class_to_ind[name])
            difficult_boxes.append(difficult)

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),
            "im_info": im_info,
        }
        return res

    def map_class_id_to_class_name(self, class_id):
        #return PascalVOCDataset.CLASSES[class_id]
        return self.cls[class_id]
