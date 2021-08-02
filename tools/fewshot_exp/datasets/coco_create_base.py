from maskrcnn_benchmark.data.datasets.coco import COCODataset
from collections import OrderedDict
import json

def dele(split):
    with open('./datasets/coco/annotations/instances_' + split + '2014.json', 'r') as f:
        o_annos = json.load(f)
    sets = [i['id'] for i in o_annos['images']]
    dic = {i['id']: {'image': i, 'aim': False, 'over': False, 'annos': []} for i in o_annos['images']}
    n_annos = {}
    n_annos['info'] = o_annos['info']
    n_annos['licenses'] = o_annos['licenses']
    n_annos['categories'] = o_annos['categories']
    print('ori img:%d'%len(o_annos['images']))
    n_annos['images'] = []
    n_annos['annotations'] = []
    catesdict = {anno['name']: anno['id'] for anno in o_annos['categories']}
    allcate = [anno['name'] for anno in o_annos['categories']]
    print(allcate)
    aimcate = ['airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorcycle', 'person', 'potted plant', 'sheep', 'couch', 'train', 'tv']
    aimcate = [a for a in allcate if a not in aimcate]
    print(aimcate)
    n_annos['categories'] = [c for c in o_annos['categories'] if c['name'] in aimcate]
    aimcateid = [catesdict[cate] for cate in aimcate]
    ##for im in o_annos['images']:
    ##    if im['id'] in sets:
    ##        n_annos['images'].append(im)
    for an in o_annos['annotations']:
        if an['category_id'] in aimcateid:
            dic[an['image_id']]['aim'] = True
            dic[an['image_id']]['annos'].append(an)
        else:
            dic[an['image_id']]['over'] = True
    for key in dic:
        if dic[key]['aim'] and not dic[key]['over']:
            n_annos['images'].append(dic[key]['image'])
            n_annos['annotations'] += dic[key]['annos']
    print('im:%d'%len(n_annos['images']))
    print('ins:%d'%len(n_annos['annotations']))
    json.dump(n_annos, open('./datasets/coco/annotations/instances_' + split + '2014_base.json', 'w+'))

if __name__ == '__main__':
    dele('valminusminival')
    dele('train')
    dele('minival')
