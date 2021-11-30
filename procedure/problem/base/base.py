from abc import abstractmethod

from typing import OrderedDict

from pymoo.core.problem import ElementwiseProblem

from util.MOEA.elitist_archive import ElitistArchive

import numpy as np

import logging


class NAS(ElementwiseProblem):
    def __init__(self, 
                 pf_dict=None,
                 pf_path=None, 
                 verbose=True, 
                 filter_duplicate_by_key=True, 
                 **kwargs):
        # super().__init__(elementwise_evaluation=True, **kwargs)
        super().__init__(**kwargs)
        self.verbose = verbose
        self.logger = logging.getLogger(self.__class__.__name__)
        self.history = OrderedDict({
            'eval': OrderedDict(), 
            'runtime': OrderedDict()
        })
        self.archive = {}
        self.elitist_archive = ElitistArchive(self.archive, verbose, filter_duplicate_by_key=filter_duplicate_by_key)
        self.msg = '[{:0>2d}/{:0>2d}]: time={:.3f}s, '
        self.counter = 0
        self.pf_path = pf_path
        self.pf_dict = pf_dict

    def _evaluate(self, x, out, algorithm, *args, **kwargs):
        self.counter += 1
        genotype = self._decode(x)
        key = tuple(x.tolist())

        if key in self.history['eval']:
            out['F'] = self.history['eval'][key]
            self.elitist_archive.insert(x, out['F'], key)
            self.logger.info('Re-evaluated arch: {}'.format(key))
            return

        F, runtime = self._calc_F(genotype)
        out['F'] = np.column_stack(F)

        if self.verbose:
            count = self.counter % algorithm.pop_size
            self.logger.info(self.msg.format(
                algorithm.pop_size if count == 0 else count,
                algorithm.pop_size,
                runtime,
                *F
            ))

        self.history['eval'][key] = out['F']

        n_gen = algorithm.n_gen
        if n_gen not in self.history['runtime']:
            self.history['runtime'][n_gen] = []
        self.history['runtime'][n_gen] += [runtime]

        self.elitist_archive.insert(x, out['F'], key)

    def _convert_to_pf_space(self, X, **kwargs):
        pass

    @abstractmethod
    def _decode(self, **kwargs):
        raise NotImplementedError
    
    def _calc_F(self, genotype, **kwargs):
        raise NotImplementedError

    def _calc_pareto_front(self, *args, **kwargs):
        pf = np.load(self.pf_path)
        return pf
    
    