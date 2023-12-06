import argparse
import logging
import os
from pprint import pformat

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

import utils
from dataset import L3FDataset
from net import FERNet


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to configuration')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--save_images', action='store_true', help='Dump predicted images')
    parser.add_argument('--no_save_images', dest='save_images', action='store_false')
    parser.set_defaults(save_images=True)
    parser.add_argument('--cpu', action='store_true')
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = utils.parse_config(args.config)
    env = utils.init_env(args, config, training=False)
    logging.info(f'using config file:\n{pformat(config)}')
    logging.info(f'using device {env.device}')

    model = FERNet(resolution=config.model.resolution).to(env.device)
    ckpt = torch.load(args.ckpt, map_location=env.device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    logging.info(f'loading model from {args.ckpt}')

    test_dataset = L3FDataset(config.data, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config.env.num_workers)
    test_metrics = [utils.init_metrics(it) for it in config.metrics]

    with torch.inference_mode():
        for data in tqdm(test_loader, leave=False, dynamic_ncols=True, desc='Testing'):
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

            for (_, metric) in test_metrics:
                metric.update(quant_out.float(), quant_gt.float())

            if args.save_images:
                save_path = os.path.join(env.visual_dir(iter='final'), f'{stem}.png')
                save_image(out, save_path, nrow=config.model.resolution, padding=0, normalize=False)

    summary = '; '.join([f'{name} {metric.compute():.4f}' for (name, metric) in test_metrics])
    logging.info(f'test result: {summary}')


if __name__ == '__main__':
    main()
