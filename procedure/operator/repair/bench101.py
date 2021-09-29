import random

import copy

from pymoo.core.repair import Repair
# from pymoo.model.repair import Repair

from lib.api.bench101.api import ModelSpec

import numpy as np

class Bench101Repair(Repair):
    def _do(self, problem, pop, **kwargs):
        Z = pop.get('X')
        for i in range(Z.shape[0]):
            x_edge, x_ops = np.split(Z[i], [problem.EDGE_SPOTS])

            matrix = np.zeros((problem.NUM_VERTICES, problem.NUM_VERTICES))
            iu = np.triu_indices(problem.NUM_VERTICES, 1)
            matrix[iu] = x_edge
            ops = np.array(problem.ALLOWED_OPS)[x_ops.astype(np.int)].tolist()
            ops = [problem.INPUT] + ops + [problem.OUTPUT]
            spec = ModelSpec(
                matrix=matrix,
                ops=ops
            )
            if not problem.api.is_valid(spec):
                matrix = self.mutate_spec(problem, spec)
                spec = ModelSpec(
                    matrix=matrix,
                    ops=ops
                )
            Z[i][:problem.EDGE_SPOTS] = matrix[iu]


        pop.set('X', Z)
        return pop

    @staticmethod
    def mutate_spec(problem, old_spec, mutation_rate=1.0):
        """Computes a valid mutated spec from the old_spec."""
        while True:
            new_matrix = copy.deepcopy(old_spec.original_matrix)
            new_ops = copy.deepcopy(old_spec.original_ops)

            # In expectation, V edges flipped (note that most end up being pruned).
            edge_mutation_prob = mutation_rate / problem.NUM_VERTICES
            for src in range(0, problem.NUM_VERTICES - 1):
                for dst in range(src + 1, problem.NUM_VERTICES):
                    if random.random() < edge_mutation_prob:
                        new_matrix[src, dst] = 1 - new_matrix[src, dst]
                
            # # In expectation, one op is resampled.
            # op_mutation_prob = mutation_rate / OP_SPOTS
            # for ind in range(1, NUM_VERTICES - 1):
            #     if random.random() < op_mutation_prob:
            #         available = [o for o in api.config['available_ops'] if o != new_ops[ind]]
            #         new_ops[ind] = random.choice(available)
                
            new_spec = ModelSpec(new_matrix, new_ops)
            if problem.api.is_valid(new_spec):
                return new_matrix