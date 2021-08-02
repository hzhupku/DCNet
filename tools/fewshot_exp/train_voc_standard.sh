#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export NGPUS=2
SPLIT=(1 2 3)
SHOT=(10 5 3 2 1)
mkdir fs_exp/voc_standard_results
for shot in ${SHOT[*]} 
do
  for split in ${SPLIT[*]} 
  do
    configfile=configs/fewshot/standard/e2e_voc_split${split}_${shot}shot_finetune.yaml
    python tools/fewshot_exp/crops/create_crops_voc_standard.py ${shot}
    python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/train_net.py --config-file ${configfile}
    rm model_final.pth
    rm last_checkpoint
    mv inference/voc_2007_test/result.txt fs_exp/voc_standard_results/result_split${split}_${shot}shot.txt
  done
done
python tools/fewshot_exp/cal_novel_voc.py
