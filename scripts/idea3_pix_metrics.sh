#!/bin/bash

gpu=0
exp=expts/2/idea3/img2pu_idea3_table_cdemd \
eval_set=test
dataset=pix3d
data_dir_imgs=/home/ubuntu/3Dreconstruction/Dataset_pix/pix3d
data_dir_pcl=/home/ubuntu/3Dreconstruction/Dataset_pix/pix3d/pointclouds
declare -a categs=("table")

for cat in "${categs[@]}"; do
	python idea3_pix_metrics.py \
		--gpu $gpu \
		--dataset $dataset \
		--data_dir_imgs ${data_dir_imgs} \
		--data_dir_pcl ${data_dir_pcl} \
		--exp $exp \
		--category $cat \
		--load_best \
		--bottleneck 512 \
		--bn_decoder \
		--eval_set ${eval_set} \
		--batch_size 1 \

done

clear
declare -a categs=("table")
for cat in "${categs[@]}"; do
	echo ${cat}
	cat ${exp}/metrics_$dataset/${eval_set}/${cat}.csv
	echo
done