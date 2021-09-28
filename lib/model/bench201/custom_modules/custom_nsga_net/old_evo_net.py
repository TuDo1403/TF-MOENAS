import torch
from torch.nn import Module
from .decoder import VariableGenomeDecoder
import torch.nn as nn
import numpy as np

class OldEvoNet(Module):
    def __init__(self, 
                 genome, 
                 input_size, 
                 output_size, 
                 n_nodes,
                 n_bits,
                 target_val,
                 **kwargs):
        
        super(OldEvoNet, self).__init__()
        self.__name__ = 'EvoNet'

        genome = genome if type(genome) == type(np.ndarray) else np.array([bit for bit in genome.replace(' ', '')], dtype=np.int)

        genome_dict, list_indices = self.setup_model_args(n_nodes, genome, n_bits, target_val, input_size)
        print(genome_dict)

        self.model = VariableGenomeDecoder(**genome_dict, repeats=None).get_model()
        self.genome_model = list_indices

        out = None
        with torch.no_grad():
            out = self.model(torch.autograd.Variable(torch.zeros(1, *(input_size))))
        shape = out.data.shape

        self.gap = nn.AvgPool2d(kernel_size=(shape[-2], shape[-1]), stride=1)

        shape = self.gap(out).data.shape

        self.linear = nn.Linear(shape[1]*shape[2]*shape[3], output_size)

        self.model.zero_grad()

    @staticmethod
    def setup_model_args(n_nodes, 
                         genome, 
                         n_bits, 
                         target_val, 
                         input_size,
                         **kwargs):

        # indices = np.arange(len(genome))
        # connections_length = [((n*(n-1)) // 2) + 3 for n in n_nodes]
        # list_connections, list_indices = [], []
        # for i, (length, n_node) in enumerate(zip(connections_length, n_nodes)):
        #     phase = genome[i*length : (i+1)*length]
        #     phase_indices = indices[i*length : (i+1)* length]
        #     list_nodes, node_indices = [], []
        #     start = 0
        #     for i in range(1, n_node):
        #         end = start + i
        #         list_nodes += [phase[start : end].tolist()]
        #         node_indices += [phase_indices[start : end].tolist()]
        #         start = end

        #     list_nodes += [[phase[-3]], [int(''.join(str(bit) for bit in phase[-2:]), 2)]]
        #     node_indices += [*[[phase_indices[-3]], phase_indices[-2:].tolist()]]

        #     list_connections += [list_nodes]
        #     list_indices += [*node_indices]

        # genome_dict = {}
        # bit_count = 0
        # for encode_name, n in n_bits.items():
        #     encode_indices = indices[::-1][bit_count : bit_count + (n*len(n_nodes))][::-1]
        #     list_indices += [encode_indices[i:i+n].tolist() for i in range(0, len(encode_indices), n)]
            
        #     bit_length = bit_count + (n*len(n_nodes))
        #     bit_length = bit_length - 1 if encode_name == 'pool_sizes' else bit_length
        #     encode_bits = genome[::-1][bit_count : bit_length][::-1]
        #     encode_val = [int(''.join(str(bit) for bit in encode_bits[i:i+n]), 2) for i in range(0, len(encode_bits), n)]
        #     target = np.array(target_val[encode_name])
        #     encode_val = target[encode_val]
        #     bit_count += n * (len(n_nodes)-1) if encode_name == 'pool_sizes' else n * len(n_nodes)
        #     genome_dict[encode_name] = encode_val

        # channels = genome_dict['channels']
        # new_channels = [None] * len(channels)
        # for i, channel in enumerate(channels):
        #     new_channels[i] = [channels[i-1] if i != 0 else input_size[0], channel]
        # genome_dict['channels'] = new_channels

        # genome_dict['list_genome'] = list_connections
        # return genome_dict, list_indices
        
        indices = np.arange(len(genome))
        connections_length = [((n*(n-1)) // 2) + 3 for n in n_nodes]
        list_connections, list_indices = [], []
        for i, (length, n_node) in enumerate(zip(connections_length, n_nodes)):
            phase = genome[i*length : (i+1)*length]
            phase_indices = indices[i*length : (i+1)* length]
            list_nodes, node_indices = [], []
            start = 0
            for i in range(1, n_node):
                end = start + i
                list_nodes += [phase[start : end].tolist()]
                node_indices += [phase_indices[start : end].tolist()]
                start = end

            list_nodes += [[phase[-3]], [int(''.join(str(bit) for bit in phase[-2:]), 2)]]
            node_indices += [*[[phase_indices[-3]], phase_indices[-2:].tolist()]]

            list_connections += [list_nodes]
            list_indices += [*node_indices]

        genome_dict = {}
        bit_count = 0
        for encode_name, n in n_bits.items():
            encode_indices = indices[::-1][bit_count : bit_count + (n*len(n_nodes))][::-1]
            list_indices += [encode_indices[i:i+n].tolist() for i in range(0, len(encode_indices), n)]

            encode_bits = genome[::-1][bit_count : bit_count + (n*len(n_nodes))][::-1]
            encode_val = [int(''.join(str(bit) for bit in encode_bits[i:i+n]), 2) for i in range(0, len(encode_bits), n)]
            target = np.array(target_val[encode_name])
            encode_val = target[encode_val]
            bit_count += n * len(n_nodes)
            genome_dict[encode_name] = encode_val

        channels = genome_dict['channels']
        new_channels = [None] * len(channels)
        for i, channel in enumerate(channels):
            new_channels[i] = [channels[i-1] if i != 0 else input_size[0], channel]
        genome_dict['channels'] = new_channels

        genome_dict['list_genome'] = list_connections
        return genome_dict, list_indices

    def forward(self, x):
        x = self.gap(self.model(x))
        x = x.view(x.size(0), -1)
        return self.linear(x)
