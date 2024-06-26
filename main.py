# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import yaml
import copy
import numpy as np
import torch
import torch.nn.functional as F
import src.deit as deit
from src.utils import (get_rank, get_world_size, init_distributed_mode, trunc_normal_, WarmupCosineSchedule, CosineWDSchedule, CSVLogger, grad_logger, AllReduce, AverageMeter, is_main_process)
from src.data_manager import (init_data, make_transforms)
from src.mask_generators import MaskCollator
from torch.nn.parallel import DistributedDataParallel

# --
log_timings = True
log_freq = 100
checkpoint_freq = 25
checkpoint_freq_itr = 2500
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()

parser.add_argument(
    '--fname', type=str,
    help='yaml file containing config file names to launch',
    default='configs.yaml')
# distributed training parameters
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--dist_on_itp', action='store_true')
parser.add_argument('--gpu', default=0)
parser.add_argument('--dist_url', default='env://',
                    help='url used to set up distributed training')
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser.add_argument("--distributed", action="store_true")



def load_checkpoint(
        r_path,
        encoder,
        predictor,
        target_encoder,
        opt,
        scaler,
):
    checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
    epoch = checkpoint['epoch']

    # -- loading encoder
    pretrained_dict = checkpoint['encoder']
    msg = encoder.load_state_dict(pretrained_dict)
    print(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

    # -- loading predictor
    pretrained_dict = checkpoint['predictor']
    msg = predictor.load_state_dict(pretrained_dict)
    print(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

    # -- loading target_encoder
    if target_encoder is not None:
        print(list(checkpoint.keys()))
        pretrained_dict = checkpoint['target_encoder']
        msg = target_encoder.load_state_dict(pretrained_dict)
        print(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

    # -- loading optimizer
    opt.load_state_dict(checkpoint['opt'])
    if scaler is not None:
        scaler.load_state_dict(checkpoint['scaler'])
    print(f'loaded optimizers from epoch {epoch}')
    print(f'read-path: {r_path}')
    del checkpoint
    return encoder, predictor, target_encoder, opt, scaler, epoch


def init_model(device,
        patch_size=16,
        use_projector=False,
        model_name='deit_base',
        crop_size=224,
        pred_depth=6,
        emb_dim=384,
        include_mask_token=True,
        learned_pos_emb=False,
        apply_stop=False,
        noise_var=0.25
):
    encoder = deit.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size,
        use_projector=use_projector).to(device)
    
    predictor = deit.__dict__['deit_predictor'](
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads,
        include_mask_token=include_mask_token,
        learned_pos_emb=learned_pos_emb,
        apply_stop=apply_stop,
        noise_var=noise_var).to(device)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)

    for m in predictor.modules():
        init_weights(m)

    print(encoder)
    return encoder, predictor

def init_opt(
        encoder,
        predictor,
        iterations_per_epoch,
        start_lr,
        ref_lr,
        warmup,
        num_epochs,
        wd=1e-6,
        final_wd=1e-6,
        final_lr=0.0,
        use_float16=False,
        ipe_scale=1.25,
        fix_lr_thres=-1,
        fix_wd_thres=-1,
        fix_lr_strategy='const',
        fix_wd_strategy='const'

):
    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]

    print('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        fix_lr_thres=fix_lr_thres * iterations_per_epoch,
        fix_strategy=fix_lr_strategy)
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        fix_wd_thres=fix_wd_thres * iterations_per_epoch,
        fix_strategy=fix_wd_strategy
        )
    scaler = torch.cuda.amp.GradScaler() if use_float16 else None
    return optimizer, scaler, scheduler, wd_scheduler

def main(config, args):



    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    target_blocks = config['meta']['target_blocks']
    use_float16 = config['meta']['use_float16']
    model_name = config['meta']['model_name']
    load_model = config['meta']['load_checkpoint']
    r_file = config['meta']['read_checkpoint']
    copy_data = config['meta']['copy_data']
    pred_depth = config['meta']['pred_depth']
    emb_dim = config['meta']['emb_dim']

    if use_float16:
        bfloat16_supported = torch.cuda.is_bf16_supported()
        if bfloat16_supported:
            dtype = torch.bfloat16
            print(f'Using bfloat16')
        else:
            dtype = torch.float16
            print(f'No bfloat16 support, using float16')
    else:
        dtype = None

    # -- DATA
    homo = config['data']['homo'] if 'homo' in config['data'] else None
    batch_size = config['data']['batch_size']
    pin_mem = config['data']['pin_mem'] if 'pin_mem' in config['data'] else False
    num_workers = config['data']['num_workers'] if 'num_workers' in config['data'] else 1
    color_jitter = config['data']['color_jitter_strength']
    use_gaussian_blur = config['data']['use_gaussian_blur']
    use_horizontal_flip = config['data']['use_horizontal_flip']
    use_color_distortion = config['data']['use_color_distortion']

    root_path = config['data']['root_path']
    image_folder = config['data']['image_folder']
    crop_size = config['data']['crop_size']
    crop_scale = config['data']['crop_scale']
    # --

    # -- MASK
    patch_size = config['mask']['patch_size']
    num_enc_masks = config['mask']['num_enc_masks']
    enc_mask_scale = config['mask']['enc_mask_scale']
    pred_mask_scale = config['mask']['pred_mask_scale']
    aspect_ratio = config['mask']['aspect_ratio']
    num_pred_masks = config['mask']['num_pred_masks']
    min_keep = config['mask']['min_keep']

    # -- OPTIMIZATION
    ipe_scale = config['optimization']['ipe_scale']
    clip_grad = config['optimization']['clip_grad']
    wd = float(config['optimization']['weight_decay'])
    final_wd = float(config['optimization']['final_weight_decay'])
    num_epochs = config['optimization']['epochs']
    warmup = config['optimization']['warmup']
    start_lr = config['optimization']['start_lr']
    lr = config['optimization']['lr']
    final_lr = config['optimization']['final_lr']
    all_gather = config['optimization'].get('all_gather', False)
    fix_lr_thres = config['optimization'].get('fix_lr_thres', -1)
    fix_wd_thres = config['optimization'].get('fix_wd_thres', -1)
    fix_lr_strategy = config['optimization'].get('fix_lr_strategy', 'const')
    fix_wd_strategy = config['optimization'].get('fix_wd_strategy', 'const')
    
    # -- LOGGING
    folder = config['logging']['folder']
    tag = config['logging']['write_tag']

    # DIFFUSION
    apply_stop = config['stop_params']['apply_stop']
    noise_var = config['stop_params']['var']


    # ----------------------------------------------------------------------- #

    # -- init torch distributed backend
    rank = get_rank()
    world_size = get_world_size()
    print(f'Initialized (rank/world-size) {rank}/{world_size}')

    init_distributed_mode(args)
    device = torch.device(f'cuda:%s'%args.gpu)

    if is_main_process():
        os.makedirs(folder, exist_ok=True)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = None
    if load_model:
        if r_file is not None:
            load_path = os.path.join(folder, r_file)
        elif os.path.exists(latest_path):
            load_path = latest_path
        else:
            load_model = False


    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'mask-A'),
                           ('%.5f', 'mask-B'),
                        #    ('%d', 'time (ms)')
                           )

    # -- init model
    encoder, predictor = init_model(
        device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        emb_dim=emb_dim,
        model_name=model_name,
        apply_stop=apply_stop,
        noise_var=noise_var
    )
    target_encoder = copy.deepcopy(encoder)

    # -- make data transforms
    mask_collator = MaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        min_keep=min_keep)

    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter)

    # -- init data-loaders/samplers
    (unsupervised_loader,
     unsupervised_sampler) = init_data(
         collator=mask_collator,
         homo=homo,
         transform=transform,
         batch_size=batch_size,
         pin_mem=pin_mem,
         num_workers=num_workers,
         world_size=world_size,
         rank=rank,
         root_path=root_path,
         image_folder=image_folder,
         training=True,
         copy_data=copy_data)
    ipe = len(unsupervised_loader)
    print(f'iterations per epoch: {ipe}')
  
    
    
    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_float16=use_float16,
        fix_lr_thres=fix_lr_thres,
        fix_wd_thres=fix_wd_thres,
        fix_lr_strategy=fix_lr_strategy,
        fix_wd_strategy=fix_wd_strategy
)
    if args.distributed:
        encoder = DistributedDataParallel(encoder, static_graph=True)
        predictor = DistributedDataParallel(predictor, static_graph=True)
        target_encoder = DistributedDataParallel(target_encoder)

    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule
    _start_m, _final_m = 0.996, 1.0
    _increment = (_final_m - _start_m) / (ipe * num_epochs * ipe_scale)
    momentum_scheduler = (_start_m + (_increment*i) for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()

    def save_checkpoint(epoch):

        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0 \
                    or (epoch + 1) % 10 == 0 and epoch < checkpoint_freq:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):

        print('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        if unsupervised_sampler is not None:
            unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):

            def load_imgs():
                # -- unsupervised imgs
                imgs = udata[0].to(device, non_blocking=True)
                labels = udata[1].to(device, non_blocking=True)
                masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                return (imgs, masks_1, masks_2, labels)

            imgs, masks_enc, masks_pred, _ = load_imgs()
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            def train_step():
                scheduler.step()
                wd_scheduler.step()

                def get_targets(h):
                    _, _, D = h.shape
                    all_h = []
                    for m in masks_pred:
                        mask_keep = m.unsqueeze(-1).repeat(1, 1, D)
                        all_h += [torch.gather(h, dim=1, index=mask_keep)]
                    h = torch.cat(all_h, dim=0)
                    return h.repeat_interleave(len(masks_enc), dim=0)

                def forward_target():
                    with torch.no_grad():
                        orig_h = target_encoder(imgs, K=target_blocks)
                        h = F.layer_norm(orig_h, (orig_h.size(-1),))  # normalize over feature-dim
                        return get_targets(h)

                def forward_anchor():
                    z = encoder(imgs, masks_enc)
                    z = predictor(z, masks_enc, masks_pred)
                    return z

                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=dtype, enabled=use_float16):
                    h = forward_target()
                    z = forward_anchor()

                    pred_loss = F.smooth_l1_loss(z, h, reduction='none').mean(dim=(1,2))
                    loss = pred_loss.mean()
                    
                    if all_gather:
                        loss = AllReduce.apply(loss)

                #  Step 2. Backward & step
                if use_float16:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
                    optimizer.step()
                optimizer.zero_grad()

                # Step 3. momentum update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return float(loss)
            loss = train_step()
            loss_meter.update(loss)
            # time_meter.update(etime)

            # -- Save Checkpoint
            if itr % checkpoint_freq_itr == 0:
                save_checkpoint(epoch)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val)#, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    print('[%d, %5d] loss: %.3f '
                                'masks: %.1f %.1f '
                                '[mem: %.2e] '
                                # '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   maskA_meter.avg,
                                   maskB_meter.avg,
                                   torch.cuda.max_memory_allocated() / 1024.**2,
                                   ))

            if is_main_process():
                log_stats()

            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint after every epoch
        print('avg. loss %.3f' % loss_meter.avg)
        save_checkpoint(epoch+1)
        
if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
    main(params, args)
