python ./train_fed_sam.py --data Prostate --unseen_site 5 --gpu 3 --batch_size 5 --sam_ckpt "/mnt/diskB/lyx/FedSAM/FedSAM/SAM/sam_vit_b_01ec64.pth" # 2>&1 | tee log.log