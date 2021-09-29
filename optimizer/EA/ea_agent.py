import pymoo.factory as factory

from pymoo.core.repair import NoRepair
from pymoo.core.duplicate import NoDuplicateElimination, DefaultDuplicateElimination

# from pymoo.model.repair import NoRepair
# from pymoo.model.duplicate import NoDuplicateElimination, DefaultDuplicateElimination

from optimizer.EA.base import AgentBase

import torch

import copy

import procedure.problem as problem
import procedure.operator.duplicate as duplicate
import procedure.operator.repair as repair

import os

from util.prepare_seed import prepare_seed

class EvoAgent(AgentBase):
    def __init__(self, cfg, seed=0, **kwargs):
        super().__init__(cfg, **kwargs)
        self.seed = seed
        self.cfg = cfg

        op_kwargs = self.__build_model_operators(cfg.operators)
        self.model_ori = factory.get_algorithm(
            name=cfg.algorithm.name,
            **cfg.algorithm.kwargs,
            **op_kwargs
        )

        self.model = None

    def _initialize(self, **kwargs):
        prepare_seed(self.seed)
        self.model = copy.deepcopy(self.model_ori)

        try:
            problem = factory.get_problem(
                self.cfg.problem.name, 
                **self.cfg.problem.kwargs
            )
        except:
            problem = eval(self.cfg.problem.name)(
                **self.cfg.problem.kwargs
            )
        termination = factory.get_termination(
            self.cfg.termination.name, 
            **self.cfg.termination.kwargs
        )

        self.model.setup(
            problem,
            termination,
            seed=self.seed,
            save_history=False
        )

        if 'checkpoint' in self.config:
            self._load_checkpoint(f=self.config.checkpoint)

    def _load_checkpoint(self, **kwargs):
        try:
            ckp = super()._load_checkpoint(torch, cmd=None, **kwargs)
        except:
            self.logger.warn('Checkpoint not found, proceed algorithm from scratch!')
            return

        self.model = ckp['model']
        self.cfg = ckp['cfg']


    def __build_model_operators(self, cfg):
        op_dict = {
            'repair': NoRepair(),
            'eliminate_duplicates': NoDuplicateElimination() 
        }
        op2ctor = {
            'sampling': factory.get_sampling,
            'crossover': factory.get_crossover,
            'mutation': factory.get_mutation,
            'ref_dirs': factory.get_reference_directions
        }
        for key, val in cfg.items():
            try:
                op_dict[key] = op2ctor[key](val.name, **val.kwargs)
            except Exception as e:
                op_dict[key] = eval(val.name)(**val.kwargs)

        return op_dict

    def _finalize(self, **kwargs):
        result = self.model.result()
        torch.save(result, f=os.path.join(self.config.out_dir, 'result.pth.tar'))



