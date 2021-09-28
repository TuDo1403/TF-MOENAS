from yaml import safe_load

from easydict import EasyDict as edict

import os

import logging
from logging.handlers import RotatingFileHandler

from pprint import pformat

import datetime


def load_cfg(path, seed, console_log=True):
    cfg, _ = load_yaml_cfg(path)
    cfg.exp_name = getattr(
        cfg,
        'exp_name',
        'exp-{}'.format(
            datetime.datetime.now().strftime("%Y%m%d-%H%M")
        )
    )

    cfg.exp_name += '_S{:0>2d}'.format(seed)
    cfg.summary_dir = os.path.join('exps', cfg.exp_name, 'summaries/')
    cfg.checkpoint_dir = os.path.join('exps', cfg.exp_name, 'checkpoints/')
    cfg.out_dir = os.path.join('exps', cfg.exp_name, 'out/')
    cfg.log_dir = os.path.join('exps', cfg.exp_name, 'logs/')
    cfg.gif_dir = os.path.join('exps', cfg.exp_name, 'gifs/')

    makedirs(
        cfg.summary_dir,
        cfg.checkpoint_dir,
        cfg.out_dir,
        cfg.log_dir,
        cfg.gif_dir
    )

    setup_logging(cfg.log_dir, console_log)
    logging.getLogger().info(' {} '.format('*' * 40))
    logging.getLogger().info('Exp name: {}'.format(cfg.exp_name))
    logging.getLogger().info(' {} '.format('*' * 40))
    logging.getLogger().info('Configuration: \n')
    logging.getLogger().info(pformat(cfg))

    return cfg


def makedirs(*args):
    for arg in args:
        os.makedirs(arg, exist_ok=True)


def load_yaml_cfg(path):
    with open(path, 'r') as f:
        cfg_ori = safe_load(f)
        cfg = edict(cfg_ori)
    return cfg, cfg_ori


def setup_logging(log_dir: str, console_log=True) -> None:
    file_format = \
        '[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d'
    console_format = '[%(levelname)s]: %(message)s'

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    exp_file_handler = RotatingFileHandler(filename=os.path.join(log_dir, 'exp_debug.log'),
                                           maxBytes=1e6,
                                           backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(logging.Formatter(file_format))

    exp_errors_file_handler = RotatingFileHandler(filename=os.path.join(log_dir, 'exp_error.log'),
                                                  maxBytes=1e6,
                                                  backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(logging.Formatter(file_format))

    if main_logger.hasHandlers():
        main_logger.handlers.clear()
    if console_log:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(console_format))

        main_logger.addHandler(console_handler)

    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)
