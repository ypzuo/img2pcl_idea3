python 2_train_img2pu_new_idea4.py \
	--data_dir_imgs data/shapenet/ShapeNetRendering \
	--data_dir_pcl /home/ubuntu/3Dreconstruction/AttentionDPCR/data/4096 \
	--exp expts/2/idea4/img2pu_idea4_telephone_cdemd \
	--gpu 0 \
	--ae_logs expts/2/idea4/pa_up2_idea3_2048_telephone \
	--category telephone \
	--bottleneck 512 \
	--up_ratio 4 \
	--loss cd_emd \
	--batch_size 32 \
	--lr 5e-5 \
	--bn_decoder \
	--load_best_ae \
	--max_epoch 10 \
	--print_n 100
	# --sanity_check
