python ./train_fed_sam_savenpy.py --unseen_site 8 --gpu 0 --batch_size 4 --exp FeTS1e-4 --base_lr 1e-4 --sam_ckpt "/mnt/diskB/lyx/FedSAM/FedSAM/SAM/sam_vit_b_01ec64.pth" # 2>&1 | tee log.log