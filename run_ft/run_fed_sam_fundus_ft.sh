python ./train_fed_sam_ft.py --data Fundus --num_classes 2 --unseen_site 0 --gpu 2 --batch_size 6  --exp fundus-ft-0 --base_lr 1e-4 --sam_ckpt "/mnt/diskB/xxx/FedSAM/FedSAM/SAM/sam_vit_b_01ec64.pth" 
python ./train_fed_sam_ft.py --data Fundus --num_classes 2 --unseen_site 1 --gpu 2 --batch_size 6  --exp fundus-ft-1 --base_lr 1e-4 --sam_ckpt "/mnt/diskB/xxx/FedSAM/FedSAM/SAM/sam_vit_b_01ec64.pth" 
python ./train_fed_sam_ft.py --data Fundus --num_classes 2 --unseen_site 2 --gpu 2 --batch_size 6  --exp fundus-ft-2 --base_lr 1e-4 --sam_ckpt "/mnt/diskB/xxx/FedSAM/FedSAM/SAM/sam_vit_b_01ec64.pth" 
python ./train_fed_sam_ft.py --data Fundus --num_classes 2 --unseen_site 3 --gpu 2 --batch_size 6  --exp fundus-ft-3 --base_lr 1e-4 --sam_ckpt "/mnt/diskB/xxx/FedSAM/FedSAM/SAM/sam_vit_b_01ec64.pth" 