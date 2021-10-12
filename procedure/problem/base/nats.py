from procedure.problem.base import base

from nats_bench import create

import lib.model.bench201.cell_operations as ops

import numpy as np


class NATS(base.NAS):
    SS = {
        'tss': {
            'max_nodes': 4, 
            'ops': np.array(ops.NAS_BENCH_201),
            'nodes': [0, 0, 1, 0, 1, 2]
        },
        'sss': {'channels': np.array([8, 16, 24, 32, 40, 48, 56, 64]), 'N': 5}
    }
    
    def __init__(self,
                 search_space,
                 dataset,
                 path=None,
                 epoch=12,
                 **kwargs):

        ss = self.SS[search_space]
        if search_space == 'tss':
            max_nodes, ops = ss['max_nodes'], ss['ops']
            n_var = max_nodes * (max_nodes - 1)//2
            xl, xu = np.zeros(n_var), np.ones(n_var) * max(range(len(ops)))
        else:
            n_var = ss['N']
            channels = ss['channels']
            xl, xu = np.zeros(n_var), np.ones(n_var) * max(range(len(channels)))

        super().__init__(
            filter_duplicate_by_key=False, 
            n_var=n_var,
            xl=xl,
            xu=xu,
            **kwargs
        )

        self.epoch = epoch
        self.dataset = dataset

        self.path = path
        self.search_space = search_space

        self.api = create(
            path,
            search_space,
            fast_mode=True,
            verbose=False
        )

    def __getstate__(self):
        state_dict = dict(self.__dict__)
        del state_dict['api']
        return state_dict

    def __setstate__(self, state_dict):
        self.__dict__ = state_dict
        self.api = create(
            self.path,
            self.search_space,
            fast_mode=True,
            verbose=False
        )

    def __decode_tss(self, x):
        nodes = self.SS['tss']['nodes']
        ops = self.SS['tss']['ops'][x]
        strings = ['|']

        for i, op in enumerate(ops):
            strings.append(op+'~{}|'.format(nodes[i]))
            if i < len(nodes) - 1 and nodes[i+1] == 0:
                strings.append('+|')
        return ''.join(strings)

    def __decode_sss(self, x):
        channels = self.SS['sss']['channels'][x]
        return ':'.join(str(channel) for channel in channels)

    def _decode(self, x, **kwargs):
        if self.search_space == 'sss':
            return self.__decode_sss(x)
        return self.__decode_tss(x)