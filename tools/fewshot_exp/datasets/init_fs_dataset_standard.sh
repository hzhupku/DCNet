#!/bin/bash
cd /workspace/data/pascal_voc
cp voc_per_class.py /workspace/data/pascal_voc
python voc_per_class.py
wget http://pjreddie.com/media/files/voc_label.py
python voc_label.py
cat 2007_train.txt 2007_val.txt 2012_*.txt > voc_train.txt

mkdir voclist1 # for random seed 1

cd /workspace/code/Meta-FSOD

python tools/fewshot_exp/datasets/gen_fewlist_voc.py 1 # for random seed 1

cd fs_list

python random_dict.py

cd ..

#init base/novel sets for fewshot exps
python tools/fewshot_exp/datasets/voc_create_base.py
python tools/fewshot_exp/datasets/voc_create_standard.py 1 # for random seed 1
mkdir fs_exp
