#!/bin/bash

gpu=0
exp=expts/2/idea3/img2pu_idea3_sofa_cdemd \
eval_set=valid
data_dir_imgs=data/shapenet/ShapeNetRendering
data_dir_pcl=data/shapenet/ShapeNet_pointclouds
declare -a categs=("sofa")

for cat in "${categs[@]}"; do
	echo python idea3_metrics_img2pu.py --gpu $gpu --data_dir_imgs ${data_dir_imgs} --data_dir_pcl ${data_dir_pcl} --exp $exp  --category $cat --load_best --bottleneck 512 --bn_decoder --eval_set ${eval_set} --batch_size 128
	python idea3_metrics_img2pu.py --gpu $gpu --data_dir_imgs ${data_dir_imgs} --data_dir_pcl ${data_dir_pcl} --exp $exp  --category $cat --load_best --bottleneck 512 --bn_decoder --eval_set ${eval_set} --batch_size 128
done

declare -a categs=("sofa")
for cat in "${categs[@]}"; do
	echo ${cat}
	cat ${exp}/metrics/${eval_set}/${cat}.csv
	echo
done
