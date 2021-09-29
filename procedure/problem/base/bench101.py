from os import path
from procedure.problem.base import base

import numpy as np

import torch

from lib.api.bench101.api import NASBench, ModelSpec

import os
from os.path import expanduser

class Bench101(base.NAS):
    INPUT = 'input'
    OUTPUT = 'output'
    CONV3X3 = 'conv3x3-bn-relu'
    CONV1X1 = 'conv1x1-bn-relu'
    MAXPOOL3X3 = 'maxpool3x3'
    NUM_VERTICES = 7
    ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
    EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) // 2   # Upper triangular matrix
    OP_SPOTS = NUM_VERTICES - 2   # Input/output vertices are fixed
    def __init__(self,
                 path,
                 net_cfg,
                 epoch=36,
                 **kwargs):
        edge_ub = np.ones(self.EDGE_SPOTS)
        edge_lwb = np.zeros(self.EDGE_SPOTS)
        op_ub = np.ones(self.OP_SPOTS) * max(range(len(self.ALLOWED_OPS)))
        op_lwb = np.zeros(self.OP_SPOTS)
        super().__init__(
            n_var=self.EDGE_SPOTS+self.OP_SPOTS, 
            xl=np.concatenate([edge_lwb, op_lwb]), 
            xu=np.concatenate([edge_ub, op_ub]), 
            **kwargs
        )

        self.net_cfg = net_cfg
        self.epoch = epoch

        self.path = path
        if '~' in path:
            self.path = os.path.join(expanduser('~'), path[2:])
        self.api = NASBench(self.path)


    def __getstate__(self):
        state_dict = dict(self.__dict__)
        del state_dict['api']
        return state_dict

    def __setstate__(self, state_dict):
        self.__dict__ = state_dict
        self.api = NASBench(self.path)

    def _decode(self, x):
        dag, ops = np.split(x, [self.EDGE_SPOTS])
        matrix = np.zeros((self.NUM_VERTICES, self.NUM_VERTICES))
        iu = np.triu_indices(self.NUM_VERTICES, 1)
        matrix[iu] = dag

        ops = np.array(self.ALLOWED_OPS)[ops.astype(np.int)].tolist()

        return matrix.astype(np.int), [self.INPUT] + ops + [self.OUTPUT]


        