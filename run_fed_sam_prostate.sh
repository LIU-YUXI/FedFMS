python ./train_fed_sam.py --data Prostate --unseen_site 0 --gpu 3 --batch_size 4 --exp prostate1e-4-0 --base_lr 1e-4 --sam_ckpt "/mnt/diskB/lyx/FedSAM/FedSAM/SAM/sam_vit_b_01ec64.pth" --max_epoch 60 # 2>&1 | tee log.log
python ./train_fed_sam.py --data Prostate --unseen_site 1 --gpu 3 --batch_size 4 --exp prostate1e-4-1 --base_lr 1e-4 --sam_ckpt "/mnt/diskB/lyx/FedSAM/FedSAM/SAM/sam_vit_b_01ec64.pth" --max_epoch 60 # 2>&1 | tee log.log
python ./train_fed_sam.py --data Prostate --unseen_site 2 --gpu 3 --batch_size 4 --exp prostate1e-4-2 --base_lr 1e-4 --sam_ckpt "/mnt/diskB/lyx/FedSAM/FedSAM/SAM/sam_vit_b_01ec64.pth" --max_epoch 60 # 2>&1 | tee log.log
python ./train_fed_sam.py --data Prostate --unseen_site 3 --gpu 3 --batch_size 4 --exp prostate1e-4-3 --base_lr 1e-4 --sam_ckpt "/mnt/diskB/lyx/FedSAM/FedSAM/SAM/sam_vit_b_01ec64.pth" --max_epoch 60 # 2>&1 | tee log.log
python ./train_fed_sam.py --data Prostate --unseen_site 4 --gpu 3 --batch_size 4 --exp prostate1e-4-4 --base_lr 1e-4 --sam_ckpt "/mnt/diskB/lyx/FedSAM/FedSAM/SAM/sam_vit_b_01ec64.pth" --max_epoch 60 # 2>&1 | tee log.log
python ./train_fed_sam.py --data Prostate --unseen_site 5 --gpu 3 --batch_size 4 --exp prostate1e-4 --base_lr 1e-4 --sam_ckpt "/mnt/diskB/lyx/FedSAM/FedSAM/SAM/sam_vit_b_01ec64.pth" # 2>&1 | tee log.log