import logging
import os
import random
import sys
from functools import partial
from time import localtime, strftime

import numpy as np
import toml
import torch
from addict import Dict
from termcolor import colored
from torch.backends import cudnn
from torchmetrics import Metric
from torchmetrics.image import (LearnedPerceptualImagePatchSimilarity,
                                PeakSignalNoiseRatio,
                                StructuralSimilarityIndexMeasure)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def parse_config(path):
    return Dict(toml.load(path))


class Env(object):

    def __init__(self, args, config, training=True) -> None:
        super().__init__()

        self.device = self.setup_device(args, config)

        # experiment folder
        self._save_dir = os.path.join(config.env.exp_dir, config.env.exp_name)

        # folder to save predicted images
        self._visual_dir = os.path.join(self._save_dir, 'visual_results')

        # setup logger
        self.setup_logger(self.save_dir, save_local=training, console_high_pri_msg_only=True)

        # set seed
        if config.env.seed:
            logging.info(f'using random seed = {config.env.seed}')
            self.seed_everything(config.env.seed)

        # cudnn setup
        if config.env.deterministic:
            logging.info(f'set cudnn deterministic = True')
            cudnn.deterministic = True
            cudnn.benchmark = False

    @property
    def save_dir(self):
        return ensure_dir(self._save_dir)

    def visual_dir(self, iter):
        return ensure_dir(os.path.join(self._visual_dir, str(iter)))

    def setup_device(self, args, config):
        device = torch.device('cpu') if args.cpu or not torch.cuda.is_available() else torch.device('cuda')
        return device

    def setup_logger(self, exp_dir, save_local, console_high_pri_msg_only):
        fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
        color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
            colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'
        datefmt = '%Y-%m-%d %H:%M:%S'

        handlers = []
        if save_local:
            log_fpath = os.path.join(exp_dir, f'{strftime("%m%d_%H%M%S", localtime())}.log')
            file_handler = logging.FileHandler(log_fpath)
            file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
            handlers.append(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(fmt=color_fmt, datefmt=datefmt))
        if console_high_pri_msg_only:
            console_handler.setLevel(logging.INFO)
        handlers.append(console_handler)

        logging.basicConfig(handlers=handlers, level=logging.DEBUG, force=True)

    def seed_everything(self, seed):
        """ Set random seed """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)


def init_env(args, config, training=True):
    return Env(args, config, training)


def save_state_dict(model, optimizer, curr_iter, save_path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'start_iter': curr_iter + 1   # next iter
    }, save_path)


def init_metrics(cfg):
    METRICS = {
        # all metrics are computed in torch.uint8 data type
        'psnr': partial(PeakSignalNoiseRatio, data_range=255.0, dim=(1, 2, 3)),
        'ssim': partial(StructuralSimilarityIndexMeasure, data_range=255.0),
        'lpips': partial(MeanLearnedPerceptualImagePatchSimilarity, net_type='alex', data_range=255.0)
    }
    metric_cls = METRICS.get(cfg.name.lower(), None)
    if not metric_cls:
        raise NotImplementedError

    metric = metric_cls()
    return (cfg.name, metric)


class MeanLearnedPerceptualImagePatchSimilarity(Metric):
    def __init__(self, net_type, data_range):
        super().__init__()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type)
        self.data_range = data_range

    def update(self, preds, target):
        # scale to [0, 1]
        preds = preds / self.data_range
        target = target / self.data_range

        self.lpips.update(preds, target)

    def compute(self):
        return self.lpips.compute()

    def reset(self):
        self.lpips.reset()
