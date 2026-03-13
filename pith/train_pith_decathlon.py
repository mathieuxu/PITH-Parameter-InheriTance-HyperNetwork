# Copyright (c) 2023. Samsung Electronics Co., Ltd. All Rights Reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.




import sys
from pathlib import Path
# sys.path.append('./ppuda-main')
sys.path.append(str(Path(__file__).resolve().parent.parent))
import argparse
import torch.distributed as dist
from functools import partial
from config import init_config
import ema as ema
from ppuda.vision.loader import image_loader
from ghn3_mtl.vit1m import DeepNets1MDDP
from ghn3_mtl.nn_pith import GHN3
from ghn3_mtl.utils import log
from ghn3_mtl.trainer_pith import Trainer
from ghn3_mtl.ddp_utils import setup_ddp, clean_ddp
import time
import torchvision.models as models
from torchvision.models import ViT_L_16_Weights
from torchvision import datasets, transforms
from multi_tasks import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import imdbfolder_coco as imdbfolder
from sample import  TaskSampler

log = partial(log, flush=True)


def main():
    parser = argparse.ArgumentParser(description='GHN-3 training')
    parser.add_argument('--heads', type=int, default=8, help='number of self-attention heads in GHN-3')
    parser.add_argument('--compile', type=str, default=None, help='use pytorch2.0 compilation for potential speedup')
    parser.add_argument('--ghn2', action='store_true', help='train GHN-2, also can use code from'
                                                            ' https://github.com/facebookresearch/ppuda to train GHN-2')
    parser.add_argument('--interm_epoch', type=int, default=25, help='intermediate epochs to keep checkpoints for')
    parser.add_argument('--use_teacher', action='store_true', help='use teacher model (ViT-B/16) for knowledge distillation')
    ghn2 = parser.parse_known_args()[0].ghn2

    ddp = setup_ddp()
    args = init_config(mode='train_ghn', parser=parser, verbose=ddp.rank == 0,
                       debug=0,   # to avoid extra sanity checks and make training faster
                       layers=3,  # default number of layers in GHN-3
                       shape_multiplier=2 if ghn2 else 1)  # max_shape default setting (can be overriden by --max_shape)

    if hasattr(args, 'multigpu') and args.multigpu:
        raise NotImplementedError(
            'the `multigpu` argument was meant to use nn.DataParallel in the GHN-2 code. '
            'nn.DataParallel is likely to be deprecated in PyTorch in favor of nn.DistributedDataParallel '
            '(https://github.com/pytorch/pytorch/issues/659360).'
            'Therefore, this repo is not supporting DataParallel anymore as it complicates some steps. '
            'nn.DistributedDataParallel is used if this script is called with torchrun (see examples on top).')
    is_imagenet = args.dataset.startswith('imagenet')

    log('loading the %s dataset...' % args.dataset.upper())
    # Define the 9 tasks.
    dataset_names = ['aircraft', 'cifar100', 'daimlerpedcls', 'dtd',
                    'gtsrb', 'omniglot', 'svhn', 'ucf101', 'vgg-flowers']

    # Load data for all tasks.
    train_loaders, _, num_classes = imdbfolder.prepare_data_loaders(
        dataset_names, 
        batch_size=args.batch_size
    )


    # Load task embeddings
    task_embeddings = torch.load('./task_embeddings.pt')
    task_embedding_dim = len(task_embeddings[0])
    
    # Compute dataset sizes.
    dataset_sizes = [len(loader.dataset) for loader in train_loaders]

    # Use TaskSampler for multi-task sampling.
    temperature = 2
    task_sampler = TaskSampler(train_loaders, dataset_sizes, temperature, 
                            batch_size=args.batch_size)
        
        
    hid = args.hid
    s = 16 if is_imagenet else 11
    dmax_shape = 2048
    #default_max_shape = (hid * 2, hid * 2, s, s) if ghn2 else (hid, hid, s, s)
    default_max_shape = (dmax_shape, dmax_shape, s, s) if ghn2 else (dmax_shape, dmax_shape, s, s)
    log('current max_shape: {} {} default max_shape: {}'.format(args.max_shape,
                                                                '=' if args.max_shape == default_max_shape else '!=',
                                                                default_max_shape))

    config = {'max_shape': args.max_shape, 'num_classes': 1000, 'hypernet': args.hypernet,
              'lora': args.lora, 'lora_r': args.lora_r, 'max_ck_lora': args.max_ck_lora, 'use_1d_decoder': args.use_1d_decoder,
              'decoder': args.decoder, 'weight_norm': args.weight_norm, 've': args.virtual_edges > 1,
              'layernorm': args.ln, 'hid': hid, 'layers': args.layers, 'heads': args.heads, 'is_ghn2': ghn2, 'task_embedding_dim': task_embedding_dim}

    ghn = GHN3(**config, debug_level=args.debug).to(args.device)
    
    
    # Load checkpoint file
    checkpoint = torch.load('./checkpoints/pith/checkpoint.pt', weights_only=False)
    
    
    # Assume the model weights are saved under the 'state_dict' key in the checkpoint
    pretrained_state_dict = checkpoint['state_dict']

    # Get the state dict of the new model
    new_model_state_dict = ghn.state_dict()

    # Create lists to store parameter keys by category
    loaded_parameters = []
    unloaded_parameters = []
    frozen_parameters = []
    trainable_parameters = []

    # Update the new model's state dict with matching parameters from the checkpoint
    for k, v in new_model_state_dict.items():
        if k in pretrained_state_dict and pretrained_state_dict[k].size() == v.size():
            new_model_state_dict[k] = pretrained_state_dict[k]
            loaded_parameters.append(k)
        else:
            unloaded_parameters.append(k)

    # Load the updated weights into the new model
    ghn.load_state_dict(new_model_state_dict)
    # Set all parameters as trainable.
    for name, param in ghn.named_parameters():
        param.requires_grad = True
        trainable_parameters.append(name)

    # Print loaded and unloaded parameters
    loaded_params_count = sum([new_model_state_dict[k].numel() for k in loaded_parameters])
    unloaded_params_count = sum([new_model_state_dict[k].numel() for k in unloaded_parameters])

    print("Parameters successfully loaded from the checkpoint:")
    for key in loaded_parameters:
        print(key)
    print(f"\nTotal number of parameters successfully loaded: {loaded_params_count}\n")

    print("Parameters in the new model that were not loaded from the checkpoint:")
    for key in unloaded_parameters:
        print(key)
    print(f"\nTotal number of parameters not loaded: {unloaded_params_count}\n")

    # Print frozen and trainable parameters
    frozen_params_count = sum([new_model_state_dict[k].numel() for k in frozen_parameters])
    trainable_params_count = sum([new_model_state_dict[k].numel() for k in trainable_parameters])

    print("Frozen parameters (not trainable):")
    for key in frozen_parameters:
        print(key)
    print(f"\nTotal number of frozen parameters: {frozen_params_count}\n")

    print("Trainable parameters:")
    for key in trainable_parameters:
        print(key)
    print(f"\nTotal number of trainable parameters: {trainable_params_count}\n")


    
    # Load teacher model.
    teacher_model = None
    if hasattr(args, 'use_teacher') and args.use_teacher:
        log('Loading teacher model (ViT-L/16)...')
        teacher_model = models.vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)
        teacher_model = teacher_model.to(args.device)
        teacher_model.eval()
        log('Teacher model loaded successfully!')
    
    ema_helper = None
    ### Apply EMA ###
    if args.ema:
        ema_helper = ema.EMAHelper(mu = args.ema_rate)
        ema_helper.register(ghn)
    ### Apply EMA ###

    graphs_queue, sampler = DeepNets1MDDP.loader(args.meta_batch_size // (ddp.world_size if ddp.ddp else 1),
                                                 dense=ghn.is_dense(),
                                                 wider_nets=is_imagenet,
                                                 split=args.split,
                                                 nets_dir=args.data_dir,
                                                 virtual_edges=args.virtual_edges,
                                                 num_nets=args.num_nets,
                                                 large_images=is_imagenet,
                                                 verbose=ddp.rank == 0,
                                                 debug=args.debug > 0)

    trainer = Trainer(ghn,
                      opt=args.opt,
                      opt_args={'lr': args.lr, 'weight_decay': args.wd, 'momentum': args.momentum},
                      scheduler='mstep' if args.scheduler is None else args.scheduler,
                      scheduler_args={'milestones': args.lr_steps, 'gamma': args.gamma},
                      # n_batches=len(train_queue),
                      n_batches=task_sampler.steps_per_epoch,
                      grad_clip=args.grad_clip,
                      device=args.device,
                      log_interval=args.log_interval,
                      amp=args.amp,
                      amp_min_scale=1024,       # this helped stabilize AMP training
                      amp_growth_interval=100,  # this helped stabilize AMP training
                      predparam_wd=0 if ghn2 else 3e-5,
                      label_smoothing=0.1 if is_imagenet else 0.0,
                      save_dir=args.save,
                      ckpt=args.ckpt,
                      epochs=args.epochs,
                      verbose=ddp.rank == 0,
                      compile_mode=args.compile,
                      ema=args.ema,
                      ema_helper=ema_helper,
                      teacher_model=teacher_model,
                      )

    log('\nStarting training GHN with {} parameters!'.format(sum([p.numel() for p in ghn.parameters()])))
    if ddp.ddp:
        # make sure sample order is different for each seed
        sampler.sampler.seed = args.seed
        log(f'shuffle DeepNets1MDDP train loader: set seed to {args.seed}')
        # for each DeepNets1MDDP epoch, the graph loader will be shuffled inside the ghn3/deepnets1m.py

    graphs_queue = iter(graphs_queue)

    start_time = time.time()
    for epoch in range(trainer.start_epoch, args.epochs):
  
        log('\nepoch={:03d}/{:03d}, lr={:e}'.format(epoch + 1, args.epochs, trainer.get_lr()))

        trainer.reset_metrics(epoch)
        
        
        
        
        for step in range(task_sampler.steps_per_epoch):
            # Sample a task from TaskSampler.
            images, targets, task_id = task_sampler.sample_task()

            # Retrieve task-related information.
            task_embedding = task_embeddings[task_id]
            task_class_num = num_classes[task_id]
            dataset_name = dataset_names[task_id]

            trainer.update(images, targets, task_class_num, task_embedding, 
                        dataset_name, graphs=next(graphs_queue), is_imagenet=True)
            trainer.log(step)
            
            if args.save:
                trainer.save(epoch, step, {'args': args, 'config': config}, 
                            interm_epoch=args.interm_epoch)
        

        trainer.scheduler_step()  # lr scheduler step

    log('done at {}!'.format(time.strftime('%Y%m%d-%H%M%S')))
    if ddp.ddp:
        clean_ddp()

    end_time = time.time()
    log('Training took {:.2f} mins'.format((end_time - start_time) / 60))

if __name__ == '__main__':
    main()
