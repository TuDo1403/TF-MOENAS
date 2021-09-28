
from pymoo.util.nds.non_dominated_sorting import find_non_dominated

import numpy as np

import logging

class ElitistArchive:
    def __init__(self, archive, verbose=True, filter_duplicate_by_key=True) -> None:
        self.archive = archive
        self.verbose = verbose
        self.logger = logging.getLogger(self.__class__.__name__)
        self.filter_duplicate_by_key = filter_duplicate_by_key

    def get(self, key):
        return self.archive[key]

    def __acceptance_test(self, f, key):
        if len(self.archive) == 0:
            return True 
        elif not self.__is_duplicate(f, key) and\
            len(find_non_dominated(f, self.archive['F'])) > 0:
            return True
        else:
            return False

    def __is_duplicate(self, f, key):
        if self.filter_duplicate_by_key:
            return key in self.archive['keys']
        else:
            return f.tolist() in self.archive['F'].tolist()

    def insert(self, x, f, key):
        if self.__acceptance_test(f, key):
            if len(self.archive) == 0:
                self.archive.update({
                    'X': x,
                    'F': f,
                    'keys': [key]
                })
            else:
                keys = np.row_stack([self.archive['keys'], key])
                X = np.row_stack([self.archive['X'], x])
                F = np.row_stack([self.archive['F'], f])
                I = find_non_dominated(F, F)

                self.archive.update({
                    'X': X[I],
                    'F': F[I],
                    'keys': keys[I].tolist()
                })
            if self.verbose:
                self.logger.info('Current archive size: {}'.format(len(self.archive['F'])))
            return True
        return False
