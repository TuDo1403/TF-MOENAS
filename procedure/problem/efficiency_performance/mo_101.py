import torch
from procedure.problem.base import bench101 as base

from lib.api.bench101.api import ModelSpec
from lib.model.bench101.model import Network

from util.net.flops_benchmark import get_model_infos

import numpy as np

class EfficiencyAccuracy101(base.Bench101):
    def __init__(self, 
                 efficiency, 
                 pf_benchmark=None, 
                 **kwargs):
        super().__init__(n_obj=2, **kwargs)

        self.efficiency = efficiency
        self.msg += efficiency + '={:.3f}, ' + 'validation_accuracy' + '={:.3f}'
        self.pf_benchmark_path = pf_benchmark
        self.pf_benchmark = None
        if self.pf_benchmark_path:
            self.pf_benchmark = torch.load(self.pf_benchmark_path)

    def __getstate__(self):
        state_dict = super().__getstate__()
        del state_dict['pf_benchmark']
        del state_dict['api']
        return state_dict

    def __setstate__(self, state_dict):
        super().__setstate__(state_dict)
        if self.pf_space_dict_path:
            self.pf_space_dict = torch.load(self.pf_space_dict_path)

    def _calc_F(self, genotype, **kwargs):
        matrix, ops = genotype
        spec = ModelSpec(matrix, ops)

        err = (1 - self.api.query(spec, epochs=self.epoch)['validation_accuracy']) * 100 
        network = Network(spec, self.net_cfg.num_labels)
        if self.efficiency == 'flops' or self.efficiency == 'n_params':
            flops, n_params = get_model_infos(network, self.net_cfg.input_size)
            efficiency = eval(self.efficiency)
        else:
            efficiency = self.api.query(spec, epochs=self.epoch)[self.efficiency]
        
        runtime = self.api.query(spec, epochs=self.epoch)['training_time']

        F = [efficiency, err]

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