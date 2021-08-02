from maskrcnn_benchmark.data.datasets.coco import COCODataset
import json
yolodir = '../Fewshot_Detection'
train = json.load(open('datasets/coco/annotations/instances_train2014.json'))
val = json.load(open('datasets/coco/annotations/instances_val2014.json'))
images = {}
annotations = {}

for data in [train, val]:
    for im in data['images']:
        images[im['id']] = im
        annotations[im['id']] = []
for data in [train, val]:
    for anno in data['annotations']:
        annotations[anno['image_id']].append(anno)

with open(yolodir + '/data/coco.names') as f:
    cls = f.readlines()
    cls = [i.strip() for i in cls]

for shot in [10, 30]:
    for split in ['train', 'val']:
        n_json = {'info': train['info'],
                  'images': [],
                  'licenses': train['licenses'],
                  'annotations': [],
                  'categories': train['categories']}
        ids = []
        for c in cls:
            with open(yolodir + '/data/cocosplit/full_box_%dshot_%s_trainval.txt'%(shot, c)) as f:
                content = f.readlines()
                ids += [int(i.strip()[-16: -4]) for i in content if split in i]
        ids = list(set(ids))
        for i in ids:
            n_json['images'].append(images[i])
            n_json['annotations'] += annotations[i]
        json.dump(n_json, open('./datasets/coco/annotations/instances_%s2014_%dshot_novel_standard.json'%(split, shot), 'w+'))
