# Dynamic-Asymmetric

This is the Pytorch code for our paper "Dynamic and Asymmetric Enhancement for Remote Sensing Image Change Captioning" (Submitted).


## LEVIR-CC Dataset 
**Download [Link](https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset)**


### Train
CUDA_VISIBLE_DEVICES='0,1' python -m torch.distributed.launch --master_port=3150 --nproc_per_node=2 train.py --split TEST --batch_size 20 --max_len 52 --n_enc 3 --n_dec 1 --model danet --exp_dir exps 

### Eval
CUDA_VISIBLE_DEVICES=0 python eval.py --batch_size 100 --beam_size 3 --n_enc 3 --n_dec 1 --max_len 52 --exp_dir exps --exp_name 'e3d1_len52'

### pretrained Weight
Download [Link](https://pan.baidu.com/s/1EL0TI1yROffpcmttyqalmA?pwd=eses) code: eses
