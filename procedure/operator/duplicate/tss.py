from pymoo.core.duplicate import ElementwiseDuplicateElimination
# from pymoo.model.duplicate import ElementwiseDuplicateElimination

import numpy as np

import logging

import lib.model.bench201.cell_operations as ops


class TSSDuplicateElimination(ElementwiseDuplicateElimination):
    MAX_NODES = 4
    def __init__(self, isomorphic=False, **kwargs) -> None:
        super().__init__(cmp_func=self.is_equal, **kwargs)
        self.predefined_ops = np.array(ops.NAS_BENCH_201)
        self.arch_dict = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.isomorphic = isomorphic
        self.nodes = [0, 0, 1, 0, 1, 2]

    def is_equal(self, a, b):
        x = self.__decode(a.get('X')); y = self.__decode(b.get('X'))
        same_representation = x == y
        if same_representation:
            self.logger.info(
                'Same repr: {} = {}'.format(
                    a.get('X').tolist(), 
                    b.get('X').tolist()
                )
            )
            return True
        if not self.isomorphic:
            return False
        key_a = tuple(a.get('X').tolist()); key_b = tuple(b.get('X').tolist())
        if key_a in self.arch_dict:
            unique_a = self.arch_dict[key_a]
        else:
            unique_a = self.topology_str2structure(x).to_unique_str(consider_zero=True)
            self.arch_dict[key_a] = unique_a
        
        if key_b in self.arch_dict:
            unique_b = self.arch_dict[key_b]
        else:
            unique_b = self.topology_str2structure(y).to_unique_str(consider_zero=True)
            self.arch_dict[key_b] = unique_b
        
        isomorphic = unique_a == unique_b
        if isomorphic:
            logging.info('isomorphic: - {} = {}'.format(key_a, key_b))
            return isomorphic

        return False


    def __decode(self, x):
        ops = self.predefined_ops[x]
        strings = ['|']

        for i, op in enumerate(ops):
            strings.append(op+'~{}|'.format(self.nodes[i]))
            if i < len(self.nodes) - 1 and self.nodes[i+1] == 0:
                strings.append('+|')
        return ''.join(strings)


#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.09 #
#####################################################
# Moved from https://github.com/D-X-Y/AutoDL-Projects/blob/main/lib/models/cell_searchs/genotypes.py
#####################################################


    @staticmethod
    def topology_str2structure(xstr):
        return TopologyStructure.str2structure(xstr)

from copy import deepcopy
class TopologyStructure:
    """A class to describe the topology, especially that used in NATS-Bench."""

    def __init__(self, genotype):
        assert isinstance(genotype, list) or isinstance(genotype, tuple), "invalid class of genotype : {:}".format(
            type(genotype)
        )
        self.node_num = len(genotype) + 1
        self.nodes = []
        self.node_N = []
        for idx, node_info in enumerate(genotype):
            assert isinstance(node_info, list) or isinstance(
                node_info, tuple
            ), "invalid class of node_info : {:}".format(type(node_info))
            assert len(node_info) >= 1, "invalid length : {:}".format(len(node_info))
            for node_in in node_info:
                assert isinstance(node_in, list) or isinstance(node_in, tuple), "invalid class of in-node : {:}".format(
                    type(node_in)
                )
                assert len(node_in) == 2 and node_in[1] <= idx, "invalid in-node : {:}".format(node_in)
            self.node_N.append(len(node_info))
            self.nodes.append(tuple(deepcopy(node_info)))

    def tolist(self, remove_str):
        # convert this class to the list, if remove_str is 'none', then remove the 'none' operation.
        # note that we re-order the input node in this function
        # return the-genotype-list and success [if unsuccess, it is not a connectivity]
        genotypes = []
        for node_info in self.nodes:
            node_info = list(node_info)
            node_info = sorted(node_info, key=lambda x: (x[1], x[0]))
            node_info = tuple(filter(lambda x: x[0] != remove_str, node_info))
            if len(node_info) == 0:
                return None, False
            genotypes.append(node_info)
        return genotypes, True

    def node(self, index):
        assert index > 0 and index <= len(self), "invalid index={:} < {:}".format(index, len(self))
        return self.nodes[index]

    def tostr(self):
        strings = []
        for node_info in self.nodes:
            string = "|".join([x[0] + "~{:}".format(x[1]) for x in node_info])
            string = "|{:}|".format(string)
            strings.append(string)
        return "+".join(strings)

    def check_valid(self):
        nodes = {0: True}
        for i, node_info in enumerate(self.nodes):
            sums = []
            for op, xin in node_info:
                if op == "none" or nodes[xin] is False:
                    x = False
                else:
                    x = True
                sums.append(x)
            nodes[i + 1] = sum(sums) > 0
        return nodes[len(self.nodes)]

    def to_unique_str(self, consider_zero=False):
        # this is used to identify the isomorphic cell, which rerquires the prior knowledge of operation
        # two operations are special, i.e., none and skip_connect
        nodes = {0: "0"}
        for i_node, node_info in enumerate(self.nodes):
            cur_node = []
            for op, xin in node_info:
                if consider_zero is None:
                    x = "(" + nodes[xin] + ")" + "@{:}".format(op)
                elif consider_zero:
                    if op == "none" or nodes[xin] == "#":
                        x = "#"  # zero
                    elif op == "skip_connect":
                        x = nodes[xin]
                    else:
                        x = "(" + nodes[xin] + ")" + "@{:}".format(op)
                else:
                    if op == "skip_connect":
                        x = nodes[xin]
                    else:
                        x = "(" + nodes[xin] + ")" + "@{:}".format(op)
                cur_node.append(x)
            nodes[i_node + 1] = "+".join(sorted(cur_node))
        return nodes[len(self.nodes)]

    def check_valid_op(self, op_names):
        for node_info in self.nodes:
            for inode_edge in node_info:
                # assert inode_edge[0] in op_names, 'invalid op-name : {:}'.format(inode_edge[0])
                if inode_edge[0] not in op_names:
                    return False
        return True

    def __repr__(self):
        return "{name}({node_num} nodes with {node_info})".format(
            name=self.__class__.__name__, node_info=self.tostr(), **self.__dict__
        )

    def __len__(self):
        return len(self.nodes) + 1

    def __getitem__(self, index):
        return self.nodes[index]

    @staticmethod
    def str2structure(xstr):
        if isinstance(xstr, TopologyStructure):
            return xstr
        assert isinstance(xstr, str), "must take string (not {:}) as input".format(type(xstr))
        nodestrs = xstr.split("+")
        genotypes = []
        for i, node_str in enumerate(nodestrs):
            inputs = list(filter(lambda x: x != "", node_str.split("|")))
            for xinput in inputs:
                assert len(xinput.split("~")) == 2, "invalid input length : {:}".format(xinput)
            inputs = (xi.split("~") for xi in inputs)
            input_infos = tuple((op, int(IDX)) for (op, IDX) in inputs)
            genotypes.append(input_infos)
        return TopologyStructure(genotypes)

    @staticmethod
    def str2fullstructure(xstr, default_name="none"):
        assert isinstance(xstr, str), "must take string (not {:}) as input".format(type(xstr))
        nodestrs = xstr.split("+")
        genotypes = []
        for i, node_str in enumerate(nodestrs):
            inputs = list(filter(lambda x: x != "", node_str.split("|")))
            for xinput in inputs:
                assert len(xinput.split("~")) == 2, "invalid input length : {:}".format(xinput)
            inputs = (xi.split("~") for xi in inputs)
            input_infos = list((op, int(IDX)) for (op, IDX) in inputs)
            all_in_nodes = list(x[1] for x in input_infos)
            for j in range(i):
                if j not in all_in_nodes:
                    input_infos.append((default_name, j))
            node_info = sorted(input_infos, key=lambda x: (x[1], x[0]))
            genotypes.append(tuple(node_info))
        return TopologyStructure(genotypes)
