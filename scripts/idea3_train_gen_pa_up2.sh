python 2_train_gen_pa_up2_idea3.py \
	--data_dir_pcl data/shapenet/ShapeNet_pointclouds \
	--exp expts/2/idea3/pa_up2_idea3_2048_table \
	--gpu 0 \
	--category table \
	--up_ratio 2 \
	--batch_size 64 \
	--lr 5e-4 \
    --max_epoch 50 \
	--bn_decoder \
	--print_n 20 \