export OMP_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 \
--node_rank=0 --master_addr="127.0.0.1" --master_port=1234 --use_env \
./train_ELCFS_sam_parallel.py --gpu 0 --batch_size 1 --sam_ckpt "/mnt/diskB/lyx/FedSAM-main/FedSAM/SAM/sam_vit_b_01ec64.pth" 
ps -ef | grep "train_ELCFS_sam_parallel" | grep -v grep | awk '{print "kill -9 "$2}' | sh