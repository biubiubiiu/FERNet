
import argparse
import logging
import os
from collections import defaultdict
from pprint import pformat

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

import losses
import utils
from dataset import L3FDataset, repeater
from net import FERNet


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to configuration')
    parser.add_argument('--resume_from', type=str, help='Resume training from existing checkpoint')
    parser.add_argument('--save_images', action='store_true', help='Dump predicted images')
    parser.add_argument('--no_save_images', dest='save_images', action='store_false')
    parser.set_defaults(save_images=True)
    parser.add_argument('--cpu', action='store_true')
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = utils.parse_config(args.config)
    env = utils.init_env(args, config)
    logging.info(f'using config file:\n{pformat(config)}')
    logging.info(f'using device {env.device}')

    model = FERNet(resolution=config.model.resolution).to(env.device)
    optimizer = Adam(model.parameters(), lr=config.optim.base_lr)

    train_dataset = L3FDataset(config.data, mode='train', memorize=True)
    train_loader = DataLoader(train_dataset, batch_size=config.optim.batch_size,
                              shuffle=True, num_workers=config.env.num_workers)
    train_loader = repeater(train_loader)  # infinite sampling

    val_dataset = L3FDataset(config.data, mode='test')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    val_metrics = [utils.init_metrics(it) for it in config.metrics]
    primary_val_metric_idx = next((i for i, m in enumerate(config.metrics) if m.primary == True), 0)
    primary_metric_best = -999  # assume that the higher, the better

    writer = SummaryWriter(log_dir=os.path.join(env.save_dir, 'tensorboard'))

    start_iter = 1
    if args.resume_from:
        logging.info(f'resume training from {args.resume_from}')
        ckpt = torch.load(args.resume_from, map_location=env.device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_iter = int(ckpt['start_iter'])

    acc_loss_dict = defaultdict(lambda: 0)
    vgg_perceptual_loss = losses.VGGPerceptualLoss(feature_layers=[8, 16],
                                                   weights=[1, 1],
                                                   resize=False, normalized=False).to(env.device)
    for iteration in tqdm(range(start_iter, config.optim.num_iters+1), dynamic_ncols=True, desc='Training'):
        model.train()
        data = next(train_loader)
        lq, gt = data['lq'].to(env.device), data['gt'].to(env.device)
        out = model(lq)

        l1_loss = F.l1_loss(out, gt)
        freq_loss = 0.2 * F.l1_loss(torch.abs(torch.fft.rfft2(out.flatten(0, 2), norm='ortho')),
                                    torch.abs(torch.fft.rfft2(gt.flatten(0, 2), norm='ortho')))
        perceptual_loss = 0.2 * vgg_perceptual_loss(out.flatten(0, 2), gt.flatten(0, 2))
        loss = l1_loss + freq_loss + perceptual_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc_loss_dict['l1_loss'] += l1_loss.item()
        acc_loss_dict['freq_loss'] += freq_loss.item()
        acc_loss_dict['perceptual_loss'] += perceptual_loss.item()

        if iteration % len(train_dataset) == 0:
            logging.debug(f'[Iter {iteration}] ' +
                          ', '.join([f'{name}: {val:.3f}' for name, val in acc_loss_dict.items()]))
            for name, val in acc_loss_dict.items():
                writer.add_scalar(name, val, iteration)
            acc_loss_dict.clear()

        if iteration % config.optim.eval_step == 0:
            model.eval()
            with torch.inference_mode():
                for data in tqdm(val_loader, leave=False, dynamic_ncols=True, desc=f'Evaluating'):
                    lq, gt, stem = (
                        data['lq'].to(env.device),
                        data['gt'].to(env.device),
                        data['stem'][0]
                    )
                    out = model(lq)

                    # crop back to original shape
                    h, w = gt.shape[-2:]
                    out = out[..., :h, :w]

                    # reshape to [U*V, C, H, W]
                    out = out.flatten(0, 2)
                    gt = gt.flatten(0, 2)

                    quant_out = out.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8)
                    quant_gt = gt.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8)

                    for (_, metric) in val_metrics:
                        metric.update(quant_out.float(), quant_gt.float())

                    if args.save_images:
                        save_path = os.path.join(env.visual_dir(iter=iteration), f'{stem}.png')
                        save_image(out, save_path, nrow=config.model.resolution, padding=0, normalize=False)

                metric_vals = []
                for i, (name, metric) in enumerate(val_metrics):
                    metric_val = metric.compute()
                    metric_vals.append(metric_val)

                    writer.add_scalar(f'{name.lower()}', metric_val)

                    if i == primary_val_metric_idx and metric_val > primary_metric_best:
                        primary_metric_best = metric_val
                        ckpt_path = os.path.join(env.save_dir, 'best.pth')
                        utils.save_state_dict(model, optimizer, iteration, ckpt_path)
                        logging.debug(f'save checkpoint to {ckpt_path} with {name}={metric_val}')

                    # reset internal state such that metric ready for new data
                    metric.reset()

                summary = '; '.join([f'{name} {val:.3f}' for (name, _), val in zip(val_metrics, metric_vals)])
                logging.debug(f'iter {iteration} validation: {summary}')

        utils.save_state_dict(model, optimizer, iteration, os.path.join(env.save_dir, 'latest.pth'))


if __name__ == '__main__':
    main()
