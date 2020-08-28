python 2_train_img2pu_new_idea3.py \
	--data_dir_imgs data/shapenet/ShapeNetRendering \
	--data_dir_pcl data/shapenet/ShapeNet_pointclouds \
	--exp expts/2/idea3/img2pu_idea3_sofa_cdemd \
	--gpu 0 \
	--ae_logs expts/2/idea3/pa_up2_idea3_2048_sofa \
	--category sofa \
	--bottleneck 512 \
	--up_ratio 2 \
	--loss cd_emd \
	--batch_size 64 \
	--lr 5e-5 \
	--bn_decoder \
	--load_best_ae \
	--max_epoch 20 \
	--print_n 100
	# --sanity_check
