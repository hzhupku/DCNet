import os

import torch
import torch.utils.data
from PIL import Image
import sys
import numpy as np
import torchvision.transforms.functional as F
from torchvision.transforms import RandomCrop,ToTensor, Compose
from maskrcnn_benchmark.data.transforms.transforms import Normalize
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


from maskrcnn_benchmark.structures.bounding_box import BoxList

normalize_opt = Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.], to_bgr255=True)
totensor_opt = ToTensor()
patch_opt = Compose([totensor_opt,normalize_opt])

class PascalVOCDataset(torch.utils.data.Dataset):

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

    def __init__(self, data_dir, split, use_difficult=False, transforms=None, toofew=True, seed=0):

        # data_dir: "voc/VOC2007"  ,split: "trainval_split1_base"
        self.root = data_dir
        self.image_set = split
        self.keep_difficult = use_difficult
        self.transforms = transforms

        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")
        if seed > 0 and "standard" in split:
            self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s_seed"+str(seed)+".txt")
        

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]

        #too few ids lead to an unfixed bug in dataloader
        if len(self.ids) < 50 and toofew:
            self.ids = self.ids * (int(100 / len(self.ids)) + 1)
        
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        if 'split1_base' in split:
            cls = PascalVOCDataset.CLASSES_SPLIT1_BASE
        elif 'split2_base' in split:
            cls = PascalVOCDataset.CLASSES_SPLIT2_BASE
        elif 'split3_base' in split:
            cls = PascalVOCDataset.CLASSES_SPLIT3_BASE
        else:
            cls = PascalVOCDataset.CLASSES
        self.cls = cls
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.categories = dict(zip(range(len(cls)), cls))



    def __getitem__(self, index):
        img_id = self.ids[index]
        img = Image.open(self._imgpath % img_id).convert("RGB")
      
        height, width = np.array(img).shape[:2]
        target = self.get_groundtruth(index)
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)     
        return img, target, index


    def __len__(self):
        return len(self.ids)

    def get_groundtruth(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno)

        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("difficult", anno["difficult"])
        return target

    def get_mask(self,index,origin_h,origin_w,h,w):
        mask = torch.zeros(h,w).float()
        img_id = self.ids[index]
        y_ration = float(origin_h)/h
        x_ration = float(origin_w)/w
        target = ET.parse(self._annopath % img_id).getroot()
        for obj in target.iter('object'):
            difficult = int(obj.find("difficult").text) == 1
            if difficult:
                continue
            name = obj.find('name').text.lower().strip()
            if name not in self.cls:
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
            mask[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]] = self.class_to_ind[name]
            
        return mask          



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

    def get_img_info(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        #return PascalVOCDataset.CLASSES[class_id]
        return self.cls[class_id]



def image_to_patches(img):

    img = F.resize(img,size=292)
    img = RandomCrop((255,255))(img)
    split_per_side = 3  # split of patches per image side
    patch_jitter = 21  # jitter of each patch from each grid
    h, w = img.size
    h_grid = h // split_per_side
    w_grid = w // split_per_side
    h_patch = h_grid - patch_jitter
    w_patch = w_grid - patch_jitter
    assert h_patch > 0 and w_patch > 0
    patches = []
    for i in range(split_per_side):
        for j in range(split_per_side):
            p = F.crop(img, i * h_grid, j * w_grid, h_grid, w_grid)
            p = RandomCrop((h_patch, w_patch))(p)
            patches.append(p)
    patches = [patch_opt(p) for p in patches]
    perms = []
    [perms.append(torch.cat((patches[i], patches[4]), dim=0)) for i in range(9) if i != 4]
    #patch_labels = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7])

    patches = torch.stack(perms)

    return patches