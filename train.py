import random
import time
import argparse
import os
import sys
import json
import numpy as np

import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from utils import *
from eval import evaluate_transformer
from datasets import *

from models.danet import Changer as DANet 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def init_process(local_rank, world_size):
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=local_rank)


def main(args):
    print(f"Local rank: {args.local_rank}, Time: {time.strftime('%m-%d  %H : %M : %S', time.localtime(time.time()))}")

    with open(os.path.join(args.data_folder, 'WORDMAP_' + args.data_name + '.json'), 'r') as j:
        vocab = json.load(j)
    args.vocab = vocab
        
    if args.model == 'danet':
        model = DANet(args)         
    else:
        raise ValueError('model not supported')

    model.set_finetune(finetune=False)
    # model.set_finetune2(finetune=False) # DFS-finetune

    checkpoint_path = 'exps/e3d1_len52/BEST_checkpoint.pth.tar'
    ckpt = torch.load(checkpoint_path, map_location=str(device))['model']
    msg = model.load_state_dict(ckpt, strict=True)
    print(msg)

    distributed = torch.cuda.device_count() > 1
    if distributed:
        torch.cuda.set_device(args.local_rank)
        init_process(args.local_rank, torch.cuda.device_count())
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) 
        model = DDP(
            model.to(device),
            device_ids = [args.local_rank],
            output_device = args.local_rank,
            broadcast_buffers = True,
            find_unused_parameters = False,
        )
    else:
        model = model.to(device)


    if not distributed or (distributed and dist.get_rank() == 0):
        names = ['backbone', 'encoder', 'decoder']
        if not distributed:
            print_component_params(model, names)
        else:
            print_component_params(model.module, names)

        if args.exp_name != '':
            name = f'e{args.n_enc}d{args.n_dec}_len{args.max_len}_{args.exp_name}'
        else:
            name = f'e{args.n_enc}d{args.n_dec}_len{args.max_len}'

        args.exp_dir = os.path.join(args.exp_dir, name)

        os.makedirs(args.exp_dir, exist_ok=True)   

        with open(os.path.join(args.exp_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f)

        with open(os.path.join(args.exp_dir, 'log.txt'), 'w') as f:
            pass

        with open(os.path.join(args.exp_dir, 'model_print.txt'), 'w') as f:
            print(model, file=f)


    train_dataset = CaptionDataset(args, 'TRAIN')
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,        
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_dataset = CaptionDataset(args, args.split)
    test_loader = DataLoader(
        test_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )


    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    total_iter_per_batch = len(train_loader)
    print('Number of training samples: ', total_iter_per_batch) 

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=total_iter_per_batch*args.epoch, 
        eta_min=1e-6)
    
    # DFS-finetune
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, 
    #     step_size=total_iter_per_batch, 
    #     gamma=0.75)

    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)

    best_bleu4 = 0. 
    epochs_since_improvement = 0  
    for epoch in range(1, args.epoch+1):
        if distributed:
            train_loader.sampler.set_epoch(epoch)
            print(f"Rank {dist.get_rank()}, LR: {lr_scheduler.get_last_lr()}")
        else:
            print(f"LR: {lr_scheduler.get_last_lr()}")

        print(f"Time: {time.strftime('%m-%d  %H : %M : %S', time.localtime(time.time()))}")

        model.train()
        train(args, train_loader, criterion, model, optimizer, lr_scheduler, epoch, distributed)

        if distributed:
            dist.barrier()

        model.eval()
        if not distributed or (distributed and dist.get_rank() == 0):
            metrics = evaluate_transformer(args, device, model if not distributed else model.module, vocab, test_loader)
            
            with open(os.path.join(args.exp_dir, 'log.txt'), 'a+') as f:
                f.write(str(metrics) + '\n\n')

            recent_bleu4 = metrics["Bleu_4"]
            is_best = recent_bleu4 > best_bleu4
            best_bleu4 = max(recent_bleu4, best_bleu4)

            if not is_best:
                epochs_since_improvement += 1
                print(f"\nEpochs since last improvement: {epochs_since_improvement}\n")
            else:
                epochs_since_improvement = 0

            save_checkpoint(args.exp_dir, '', model.state_dict() if not distributed else model.module.state_dict(), None, is_best, epoch)

        if distributed:
            dist.barrier()
            epochs_since_improvement_tensor = torch.tensor([epochs_since_improvement], dtype=torch.int, device=device)
            dist.broadcast(epochs_since_improvement_tensor, src=0)
            epochs_since_improvement = epochs_since_improvement_tensor.item()

        if epochs_since_improvement >= args.stop_criteria:
            print(f"The model has not improved in the last {args.stop_criteria} epochs. Stopping training.")
            sys.exit()  


def train(args, train_loader, criterion, model, optimizer, lr_scheduler, epoch, distributed):
    loss_meter = AverageMeter()
    start = time.time()
    
    for i, (image_pairs, captions) in enumerate(train_loader):

        # if i > 10:
        #     break

        image_pairs = image_pairs.cuda(non_blocking=True)
        captions = captions.cuda(non_blocking=True)

        images_a = image_pairs[:, 0, :, :, :]
        images_b = image_pairs[:, 1, :, :, :]

        loss = model(images_a, images_b, captions)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        lr_scheduler.step()

        if distributed:
            dist.barrier()  

        if not distributed or (distributed and dist.get_rank() == 0):
            loss_meter.update(loss.item())
            if i > 0 and i % args.print_freq == 0:
                log = f"Epoch: {epoch}/{args.epoch}  step: {i}/{len(train_loader)}  loss: {loss_meter.avg}  lr: {lr_scheduler.get_last_lr()}"
                with open(os.path.join(args.exp_dir, 'log.txt'), 'a+') as f:
                    f.write(log + '\n')
                print(log)

    end = time.time()
    duration = end - start
    print(f"Time: {duration:.2f} s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image_Change_Captioning')

    parser.add_argument('--data_folder', type=str, default="./data/levir-cc/")
    parser.add_argument('--data_name', type=str, default="LEVIR_CC_5_cap_per_img_5_min_word_freq")
    parser.add_argument('--exp_dir', type=str, default='exps')
    parser.add_argument('--exp_name', type=str, default='')

    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--n_enc', type=int, default=3)
    parser.add_argument('--n_dec', type=int, default=1)
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--max_len', type=int, default=41, help='levir-cc: 41 | dubai-cc: 27')
    parser.add_argument('--enc_dropout', type=float, default=0.1)
    parser.add_argument('--dec_dropout', type=float, default=0.1)

    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--stop_criteria', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--split', default="VAL")
    parser.add_argument('--beam_size', type=int, default=3)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-rank", type=int, default=0)

    args = parser.parse_args()


    SEED = args.seed
    set_seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    main(args)
