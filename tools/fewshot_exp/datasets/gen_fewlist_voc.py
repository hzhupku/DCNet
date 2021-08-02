import argparse
import random
import os
import numpy as np
from os import path
import sys

seed = sys.argv[1]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# few_nums = [1, 10]
few_nums = [1, 2, 3, 5, 10]
# few_nums = [20]
DROOT = '/workspace/data/pascal_voc'
root =  DROOT + '/voclist' + str(seed) + '/'
rootfile =  DROOT + '/voc_train.txt'


def get_bbox_fewlist(rootfile, shot):
    with open(rootfile) as f:
        names = f.readlines()
    random.seed(seed)
    cls_lists = [[] for _ in range(len(classes))]
    cls_counts = [0] * len(classes)
    while min(cls_counts) < shot:
        imgpath = random.sample(names, 1)[0]
        labpath = imgpath.strip().replace('images', 'labels') \
                                 .replace('JPEGImages', 'labels') \
                                 .replace('.jpg', '.txt').replace('.png','.txt')
        # To avoid duplication
        names.remove(imgpath)

        if not os.path.getsize(labpath):
            continue
        # Load converted annotations
        bs = np.loadtxt(labpath)
        bs = np.reshape(bs, (-1, 5))
        if bs.shape[0] > 3:
            continue

        # Check total number of bbox per class so far
        overflow = False
        bcls = bs[:,0].astype(np.int).tolist()
        for ci in set(bcls):
            if cls_counts[ci] + bcls.count(ci) > shot:
                overflow = True
                break
        if overflow:
            continue

        # Add current imagepath to the file lists 
        for ci in set(bcls):
            cls_counts[ci] += bcls.count(ci)
            cls_lists[ci].append(imgpath)

    return cls_lists


def gen_bbox_fewlist():
    print('-----------------------------------------------------------')
    print('----------- Generating fewlist  (bboxes) ------------------')
    for n in few_nums:
        print('===> On {} shot ...'.format(n))
        filelists = get_bbox_fewlist(rootfile, n)
        for i, clsname in enumerate(classes):
            print('   | Processing class: {}'.format(clsname))
            with open(path.join(root, 'box_{}shot_{}_train.txt'.format(n, clsname)), 'w') as f:
                for name in filelists[i]:
                    f.write(name)


def main():
    gen_bbox_fewlist()



if __name__ == '__main__':
    main()
