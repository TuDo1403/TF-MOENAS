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

from copy import deepcopy


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

    def __getstate__(self):
        state_dict = super().__getstate__()
        del state_dict['lr_counter']
        del state_dict['loader']
        return state_dict

    def __setstate__(self, state_dict):
        super().__setstate__(state_dict)
        self.lr_counter = Linear_Region_Collector(
            input_size=self.lr_ntk_cfg['lr']['input_size'],
            sample_batch=self.lr_ntk_cfg['lr']['sample_batch'],
            dataset=self.lr_ntk_cfg['dataset'],
            data_path=self.lr_ntk_cfg['root'],
            seed=self.lr_ntk_cfg['seed'],
            num_workers=self.lr_ntk_cfg['num_workers']
        )
        self.loader = getattr(loader, self.lr_ntk_cfg['dataset'].upper())(
            root=self.lr_ntk_cfg['root'],
            loader_kwargs={
                'num_workers': self.lr_ntk_cfg['num_workers'],
                'pin_memory': False,
                'drop_last': False,
                'batch_size': self.lr_ntk_cfg['ntk']['batch_size'],
                'worker_init_fn':random.seed(self.lr_ntk_cfg['seed'])
            }
        ).loaders['train']

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
                self.lr_counter.reinit([network])
                lr = self.lr_counter.forward_batch_sample()
                LR += lr
                self.lr_counter.clear()

        torch.cuda.empty_cache()
        return LR

    def _calc_F(self, genotype, **kwargs):
        idx = self.api.query_index_by_arch(genotype)
        cfg = self.api.get_net_config(idx, self.dataset)
        cfg.update({'use_stem': True})
        
        cfg_thin = deepcopy(cfg)
        cfg_thin.update({
            'C': 1,
            'use_stem': False
        })

        network = get_cell_based_tiny_net(edict(cfg)).cuda()
        network_thin = get_cell_based_tiny_net(edict(cfg_thin)).cuda()

        efficiency = self.api.get_cost_info(idx, self.dataset, hp=self.api.full_train_epochs)[self.efficiency]

        start = time.time()
        lrs = self.__calc_lr(network_thin)
        lr = min(lrs)

        ntks = self.__calc_ntk(network)
        ntk = np.log(max(ntks))
        end = time.time()

        runtime = end - start

        F = [efficiency, ntk, -lr]

        return F, runtime

    def _convert_to_pf_space(self, X):
        F = []
        dataset = self.pf_dict['dataset']
        for x in X:
            genotype = self._decode(x)
            idx = self.api.query_index_by_arch(genotype)
            efficiency = self.api.get_cost_info(
                idx, dataset, hp=self.api.full_train_epochs
            )[self.efficiency]
            acc = self.api.get_more_info(
                idx, dataset, hp=self.api.full_train_epochs, is_random=False
            )['test-accuracy']
            err = 100 - acc
            f = [efficiency, err]
            F += [np.column_stack(f)]
        F = np.row_stack(F)
        return F
