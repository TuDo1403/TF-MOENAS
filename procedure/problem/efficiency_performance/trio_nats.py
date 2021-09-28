from procedure.problem.base import nats as base

from util.net.linear_region import Linear_Region_Collector
from util.net.ntk import get_ntk_n
from util.net.init_weight import init_model

import loader

import random

from lib.model.bench201 import get_cell_based_tiny_net

from easydict import EasyDict as edict

import time

import torch

import numpy as np


class EfficiencyExpressivityTrainability201(base.NATS):
    def __init__(self, efficiency, lr_ntk_cfg, **kwargs):
        super().__init__(n_obj=3, **kwargs)
        self.msg += efficiency + '={:.3f}, ntk={:.3f}, lr={:.3f}'

        self.lr_ntk_cfg = lr_ntk_cfg

        self.lr_counter = Linear_Region_Collector(
            input_size=lr_ntk_cfg['lr']['input_size'],
            sample_batch=lr_ntk_cfg['lr']['sample_batch'],
            dataset=lr_ntk_cfg['dataset'],
            data_path=lr_ntk_cfg['root'],
            seed=lr_ntk_cfg['seed'],
            num_workers=lr_ntk_cfg['num_workers']
        )

        self.loader = getattr(loader, lr_ntk_cfg['dataset'].upper())(
            root=lr_ntk_cfg['root'],
            loader_kwargs={
                'num_workers': lr_ntk_cfg['num_workers'],
                'pin_memory': False,
                'drop_last': False,
                'batch_size': lr_ntk_cfg['ntk']['batch_size'],
                'worker_init_fn':random.seed(lr_ntk_cfg['seed'])
            }
        ).loaders['train']

        self.efficiency = efficiency

    def __calc_ntk(self, network):
        NTK = []
        for _ in range(self.lr_ntk_cfg['ntk']['n_repeats']):
            network = init_model(network, method='kaiming_norm_fanout')
            ntk = get_ntk_n(self.loader, [network], recalbn=0, train_mode=True, num_batch=1)
            NTK += ntk
        network.zero_grad()
        torch.cuda.empty_cache()

        return NTK

    def __calc_lr(self, network):
        LR = []
        network.train()
        with torch.no_grad():
            for _ in range(self.lr_ntk_cfg['lr']['n_repeats']):
                network = init_model(network, method='kaiming_norm_fanin')
                self.lrc_model.reinit([network])
                lr = self.lrc_model.forward_batch_sample()
                LR += lr
                self.lrc_model.clear()

        torch.cuda.empty_cache()
        return LR

    def _calc_F(self, genotype, **kwargs):
        idx = self.api.query_index_by_arch(genotype)
        cfg = self.api.get_net_config(idx, self.dataset)
        cfg.update({'use_stem': True})
        
        cfg_thin = cfg.copy()
        cfg_thin.update({
            'C': 1,
            'use_stem': False
        })

        network = get_cell_based_tiny_net(edict(cfg)).cuda()
        network_thin = get_cell_based_tiny_net(edict(cfg_thin)).cuda()

        efficiency = self.api.get_cost_info(idx, self.dataset, hp=self.hp)[self.efficiency]

        start = time.time()
        lrs = self.__calc_lr(network_thin)
        lr = min(lrs)

        ntks = self.__calc_ntk(network)
        ntk = np.log(max(ntks))
        end = time.time()

        runtime = end - start

        F = [efficiency, ntk, -lr]

        return F, runtime
