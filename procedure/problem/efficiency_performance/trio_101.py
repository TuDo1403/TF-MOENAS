from util.net.init_weight import init_model
from util.net.flops_benchmark import get_model_infos
from util.net.ntk import get_ntk_n
from util.net.linear_region import Linear_Region_Collector

from procedure.problem.base import bench101 as base

import loader

import random

import torch

from lib.api.bench101.api import ModelSpec
from lib.model.bench101.model import Network

import numpy as np

import time

from copy import deepcopy


class EfficiencyExpressivityTrainability101(base.Bench101):
    def __init__(self,
                 lr_ntk_cfg, 
                 pf_benchmark=None,
                 efficiency='flops',
                 **kwargs):
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

        self.pf_benchmark_path = pf_benchmark
        self.pf_benchmark = None
        if self.pf_benchmark_path:
            self.pf_benchmark = torch.load(self.pf_benchmark_path)

    def __getstate__(self):
        state_dict = super().__getstate__()
        del state_dict['lr_counter']
        del state_dict['loader']
        del state_dict['pf_benchmark']
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

        if self.pf_benchmark_path:
            self.pf_benchmark = torch.load(self.pf_benchmark_path)


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
        matrix, ops = genotype
        spec = ModelSpec(
            matrix=matrix,
            ops=ops
        )

        out_channels = max(1, matrix[1:].sum(axis=0)[-1])
        network_ori = Network(spec, **self.net_cfg).cuda()
        if self.efficiency == 'flops' or self.efficiency == 'n_params':
            flops, n_params = get_model_infos(network_ori, self.net_cfg.input_size)
            efficiency = eval(self.efficiency)
        else:
            efficiency = self.api.query(spec, epochs=self.epoch)[self.efficiency]

        net_config_for_ntk = deepcopy(self.net_cfg)
        net_config_for_ntk.stem_out_channels = 16
        
        network = Network(spec, **net_config_for_ntk).cuda()
        network_thin = Network(
            spec, 
            in_channels=1, 
            num_labels=self.net_cfg.num_labels,
            stem_out_channels=out_channels,
            num_stack=self.net_cfg.num_stack,
            num_modules_per_stack=self.net_cfg.num_modules_per_stack,
            use_stem=False
        ).cuda()

        start = time.time()
        lrs = self.__calc_lr(network_thin)
        ntks = self.__calc_ntk(network)
        end = time.time()

        runtime = end - start

        ntk = np.log(max(ntks))
        lr = min(lrs)

        F = [efficiency, ntk, -lr]

        return F, runtime

    def _convert_to_pf_space(self, X):
        F = []
        
        for x in X:
            matrix, ops = self._decode(x)
            spec = ModelSpec(matrix, ops)
            key = self.api._hash_spec(spec)
            efficiency = self.pf_benchmark[key][self.efficiency]
            accuracies =self.pf_benchmark[key]['computed_stat'][108]
            avg_acc = np.mean([accuracies[i]['final_test_accuracy'] for i in range(3)])
            err = (1 - avg_acc) * 100
            f = [efficiency, err]
            F += [np.column_stack(f)]

        F = np.row_stack(F)
        return F



