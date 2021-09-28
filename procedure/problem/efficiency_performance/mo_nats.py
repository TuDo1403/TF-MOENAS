from procedure.problem.base import nats as base

import numpy as np

class EfficiencyAccuracyNATS(base.NATS):
    def __init__(self, efficiency, performance, **kwargs):
        super().__init__(n_obj=2, **kwargs)
        self.msg += efficiency + '={:.3f}, ' + performance + '={:.3f}'
        self.efficiency = efficiency
        self.performance = performance

    def _calc_F(self, genotype, **kwargs):
        idx = self.api.query_index_by_arch(genotype)
        efficiency = self.api.get_cost_info(idx, self.dataset, hp=self.hp)[self.efficiency]
        data = self.api.query_by_index(idx, self.dataset, hp=self.hp)
        trial = list(data.keys())[self.trial_idx]
        data_trial = data[trial]
        runtime = sum([data_trial.get_train(i)['cur_time'] for i in range(self.epoch+1)])

        accuracy = self.api.get_more_info(
            idx,
            self.dataset,
            self.epoch,
            self.hp
        )[self.performance]

        error = 100 - accuracy
        F = [efficiency, error]
        return F, runtime

    def _convert_to_pf_space(self, X):
        F = []
        dataset = self.pf_dict['dataset']
        hp = self.pf_dict['hp']
        
        for x in X:
            genotype = self._decode(x)
            idx = self.api.query_index_by_arch(genotype)
            efficiency = self.api.get_cost_info(idx, dataset, hp=hp)[self.efficiency]
            acc = \
                self.api.get_more_info(idx, dataset, hp=hp, is_random=False)['test-accuracy']
            err = 100 - acc
            f = [efficiency, err]
            F += [np.column_stack(f)]
        F = np.row_stack(F)
        return F
        
    
