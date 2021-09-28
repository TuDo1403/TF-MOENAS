import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from .nsga_net_phase import *

class Decoder(ABC):
    """
    Abstract genome decoder class.
    """

    @abstractmethod
    def __init__(self, list_genome):
        """
        :param list_genome: genome represented as a list.
        """
        self._genome = list_genome

    @abstractmethod
    def get_model(self):
        raise NotImplementedError()


class ChannelBasedDecoder(Decoder):
    """
    Channel based decoder that deals with encapsulating constructor logic.
    """

    def __init__(self, list_genome, channels, repeats=None):
        """
        Constructor.
        :param list_genome: list, genome describing the connections in a network.
        :param channels: list, list of tuples describing the channel size changes.
        :param repeats: None | list, list of integers describing how many times to repeat each phase.
        """
        super().__init__(list_genome)

        self._model = None

        # First, we remove all inactive phases.
        self._genome = self.get_effective_genome(list_genome)
        self._channels = channels[:len(self._genome)]

        # Use the provided repeats list, or a list of all ones (only repeat each phase once).
        if repeats is not None:
            # First select only the repeats that are active in the list_genome.
            active_repeats = []
            for idx, gene in enumerate(list_genome):
                if phase_active(gene):
                    active_repeats.append(repeats[idx])

            self.adjust_for_repeats(active_repeats)
        else:
            # Each phase only repeated once.
            self._repeats = [1 for _ in self._genome]

        # If we had no active nodes, our model is just the identity, and we stop constructing.
        if not self._genome:
            self._model = Identity()

        # print(list_genome)

    def adjust_for_repeats(self, repeats):
        """
        Adjust for repetition of phases.
        :param repeats:
        """
        self._repeats = repeats

        # Adjust channels and genome to agree with repeats.
        repeated_genome = []
        repeated_channels = []
        for i, repeat in enumerate(self._repeats):
            for j in range(repeat):
                if j == 0:
                    # This is the first instance of this repeat, we need to use the (in, out) channel convention.
                    repeated_channels.append((self._channels[i][0], self._channels[i][1]))
                else:
                    # This is not the first instance, use the (out, out) convention.
                    repeated_channels.append((self._channels[i][1], self._channels[i][1]))

                repeated_genome.append(self._genome[i])

        self._genome = repeated_genome
        self._channels = repeated_channels

    def build_layers(self, phases):
        """
        Build up the layers with transitions.
        :param phases: list of phases
        :return: list of layers (the model).
        """
        layers = []
        last_phase = phases.pop()
        for phase, repeat in zip(phases, self._repeats):
            for _ in range(repeat):
                layers.append(phase)
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # TODO: Generalize this, or consider a new genome.

        layers.append(last_phase)
        return layers

    @staticmethod
    def get_effective_genome(genome):
        """
        Get only the parts of the genome that are active.
        :param genome: list, represents the genome
        :return: list
        """
        return [gene for gene in genome if phase_active(gene)]

    @abstractmethod
    def get_model(self):
        raise NotImplementedError()

class ResidualGenomeDecoder(ChannelBasedDecoder):
    """
    Genetic CNN genome decoder with residual bit.
    """

    def __init__(self, list_genome, channels, preact=False, repeats=None):
        """
        Constructor.
        :param list_genome: list, genome describing the connections in a network.
        :param channels: list, list of tuples describing the channel size changes.
        :param repeats: None | list, list of integers describing how many times to repeat each phase.
        """
        super().__init__(list_genome, channels, repeats=repeats)

        if self._model is not None:
            return  # Exit if the parent constructor set the model.

        # Build up the appropriate number of phases.
        phases = []
        for idx, (gene, (in_channels, out_channels)) in enumerate(zip(self._genome, self._channels)):
            phases.append(ResidualPhase(gene, in_channels, out_channels, idx, preact=preact))

        self._model = nn.Sequential(*self.build_layers(phases))

    def get_model(self):
        """
        :return: nn.Module
        """
        return self._model


class ResidualPhase(nn.Module):
    """
    Residual Genome phase.
    """

    def __init__(self, gene, in_channels, out_channels, kernel_size, idx, preact=False):
        """
        Constructor.
        :param gene: list, element of genome describing connections in this phase.
        :param in_channels: int, number of input channels.
        :param out_channels: int, number of output channels.
        :param idx: int, index in the network.
        :param preact: should we use the preactivation scheme?
        """
        super(ResidualPhase, self).__init__()

        self.channel_flag = in_channels != out_channels  # Flag to tell us if we need to increase channel size.
        self.first_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1 if idx != 0 else 3, stride=1, bias=False)
        self.dependency_graph = ResidualPhase.build_dependency_graph(gene)

        if preact:
            node_constructor = PreactResidualNode

        else:
            node_constructor = ResidualNode

        nodes = []
        for i in range(len(gene)):
            if len(self.dependency_graph[i + 1]) > 0:
                nodes.append(node_constructor(out_channels, out_channels, kernel_size))
            else:
                nodes.append(None)  # Module list will ignore NoneType.

        self.nodes = nn.ModuleList(nodes)

        #
        # At this point, we know which nodes will be receiving input from where.
        # So, we build the 1x1 convolutions that will deal with the depth-wise concatenations.
        #
        conv1x1s = [Identity()] + [Identity() for _ in range(max(self.dependency_graph.keys()))]
        for node_idx, dependencies in self.dependency_graph.items():
            if len(dependencies) > 1:
                conv1x1s[node_idx] = \
                    nn.Conv2d(len(dependencies) * out_channels, out_channels, kernel_size=1, bias=False)

        self.processors = nn.ModuleList(conv1x1s)
        self.out = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def build_dependency_graph(gene):
        """
        Build a graph describing the connections of a phase.
        "Repairs" made are as follows:
            - If a node has no input, but gives output, connect it to the input node (index 0 in outputs).
            - If a node has input, but no output, connect it to the output node (value returned from forward method).
        :param gene: gene describing the phase connections.
        :return: dict
        """
        graph = {}
        residual = gene[-1][0] == 1

        # First pass, build the graph without repairs.
        graph[1] = []
        for i in range(len(gene) - 1):
            graph[i + 2] = [j + 1 for j in range(len(gene[i])) if gene[i][j] == 1]

        graph[len(gene) + 1] = [0] if residual else []

        # Determine which nodes, if any, have no inputs and/or outputs.
        no_inputs = []
        no_outputs = []
        for i in range(1, len(gene) + 1):
            if len(graph[i]) == 0:
                no_inputs.append(i)

            has_output = False
            for j in range(i + 1, len(gene) + 2):
                if i in graph[j]:
                    has_output = True
                    break

            if not has_output:
                no_outputs.append(i)

        for node in no_outputs:
            if node not in no_inputs:
                # No outputs, but has inputs. Connect to output node.
                graph[len(gene) + 1].append(node)

        for node in no_inputs:
            if node not in no_outputs:
                # No inputs, but has outputs. Connect to input node.
                graph[node].append(0)

        return graph

    def forward(self, x):
        if self.channel_flag:
            x = self.first_conv(x)

        outputs = [x]

        for i in range(1, len(self.nodes) + 1):
            if not self.dependency_graph[i]:  # Empty list, no outputs to give.
                outputs.append(None)

            else:
                outputs.append(self.nodes[i - 1](self.process_dependencies(i, outputs)))

        return self.out(self.process_dependencies(len(self.nodes) + 1, outputs))

    def process_dependencies(self, node_idx, outputs):
        """
        Process dependencies with a depth-wise concatenation and
        :param node_idx: int,
        :param outputs: list, current outputs
        :return: Variable
        """
        return self.processors[node_idx](torch.cat([outputs[i] for i in self.dependency_graph[node_idx]], dim=1))


class VariableGenomeDecoder(ChannelBasedDecoder):
    """
    Residual decoding with extra integer for type of node inside the phase.
    This genome decoder produces networks that are a superset of ResidualGenomeDecoder networks.
    """
    RESIDUAL = 0
    PREACT_RESIDUAL = 1
    DENSE = 2

    def __init__(self, list_genome, channels, repeats=None):
        """
        Constructor.
        :param list_genome: list, genome describing the connections in a network, and the type of phase.
        :param channels: list, list of tuples describing the channel size changes.
        :param repeats: None | list, list of integers describing how many times to repeat each phase.
        """
        phase_types = [gene.pop() for gene in list_genome]
        genome_copy = copy(list_genome)  # We can't guarantee the genome won't be changed in the parent constructor.

        super().__init__(list_genome, channels, repeats=repeats)

        if self._model is not None:
            return  # Exit if the parent constructor set the model.

        # Adjust the types for repeats and inactive phases.
        self._types = self.adjust_types(genome_copy, phase_types)

        phases = []
        for idx, (gene, (in_channels, out_channels), phase_type) in enumerate(zip(self._genome,
                                                                                  self._channels,
                                                                                  self._types)):
            if phase_type == self.RESIDUAL:
                phases.append(ResidualPhase(gene, in_channels, out_channels, idx))

            elif phase_type == self.PREACT_RESIDUAL:
                phases.append(ResidualPhase(gene, in_channels, out_channels, idx, preact=True))

            elif phase_type == self.DENSE:
                phases.append(DensePhase(gene, in_channels, out_channels, idx))

            else:
                raise NotImplementedError("Phase type corresponding to {} not implemented.".format(phase_type))

        self._model = nn.Sequential(*self.build_layers(phases))

    def adjust_types(self, genome, phase_types):
        """
        Get only the phases that are active.
        Similar to ResidualDecoder.get_effective_genome but we need to consider phases too.
        :param genome: list, list of ints
        :param phase_types: list,
        :return:
        """
        effective_types = []

        for idx, (gene, phase_type) in enumerate(zip(genome, phase_types)):
            if phase_active(gene):
                for _ in range(self._repeats[idx]):
                    effective_types.append(*phase_type)

        return effective_types

    def get_model(self):
        return self._model


class DenseGenomeDecoder(ChannelBasedDecoder):
    """
    Genetic CNN genome decoder with residual bit.
    """
    def __init__(self, list_genome, channels, repeats=None):
        """
        Constructor.
        :param list_genome: list, genome describing the connections in a network.
        :param channels: list, list of tuples describing the channel size changes.
        :param repeats: None | list, list of integers describing how many times to repeat each phase.
        """
        super().__init__(list_genome, channels, repeats=repeats)

        if self._model is not None:
            return  # Exit if the parent constructor set the model.

        # Build up the appropriate number of phases.
        phases = []
        for idx, (gene, (in_channels, out_channels)) in enumerate(zip(self._genome, self._channels)):
            phases.append(DensePhase(gene, in_channels, out_channels, idx))

        self._model = nn.Sequential(*self.build_layers(phases))

    @staticmethod
    def get_effective_genome(genome):
        """
        Get only the parts of the genome that are active.
        :param genome: list, represents the genome
        :return: list
        """
        return [gene for gene in genome if phase_active(gene)]

    def get_model(self):
        """
        :return: nn.Module
        """
        return self._model


class DensePhase(nn.Module):
    """
    Phase with nodes that operates like DenseNet's bottle necking and growth rate scheme.
    Refer to: https://arxiv.org/pdf/1608.06993.pdf
    """

    def __init__(self, gene, in_channels, out_channels, kernel_size, idx):
        """
        Constructor.
        :param gene: list, element of genome describing connections in this phase.
        :param in_channels: int, number of input channels.
        :param out_channels: int, number of output channels.
        :param idx: int, index in the network.
        """
        super(DensePhase, self).__init__()

        self.in_channel_flag = in_channels != out_channels  # Flag to tell us if we need to increase channel size.
        self.out_channel_flag = out_channels != DenseNode.t
        self.first_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1 if idx != 0 else 3, stride=1, bias=False)
        self.dependency_graph = ResidualPhase.build_dependency_graph(gene)

        channel_adjustment = 0

        for dep in self.dependency_graph[len(gene) + 1]:
            if dep == 0:
                channel_adjustment += out_channels

            else:
                channel_adjustment += DenseNode.t

        self.last_conv = nn.Conv2d(channel_adjustment, out_channels, kernel_size=1, stride=1, bias=False)

        nodes = []
        for i in range(len(gene)):
            if len(self.dependency_graph[i + 1]) > 0:
                channels = self.compute_channels(self.dependency_graph[i + 1], out_channels)
                nodes.append(DenseNode(channels))

            else:
                nodes.append(None)

        self.nodes = nn.ModuleList(nodes)
        self.out = nn.Sequential(
            self.last_conv,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def compute_channels(dependency, out_channels):
        """
        Compute the number of channels incoming to a node.
        :param dependency: list, nodes that a particular node gets input from.
        :param out_channels: int, desired number of output channels from the phase.
        :return: int
        """
        channels = 0
        for d in dependency:
            if d == 0:
                channels += out_channels

            else:
                channels += DenseNode.t

        return channels

    def forward(self, x):
        if self.in_channel_flag:
            x = self.first_conv(x)

        outputs = [x]

        for i in range(1, len(self.nodes) + 1):
            if not self.dependency_graph[i]:  # Empty dependencies, no output to give.
                outputs.append(None)

            else:
                # Call the node on the depthwise concatenation of its inputs.
                outputs.append(self.nodes[i - 1](torch.cat([outputs[j] for j in self.dependency_graph[i]], dim=1)))

        if self.out_channel_flag and 0 in self.dependency_graph[len(self.nodes) + 1]:
            # Get the last nodes in the phase and change their channels to match the desired output.
            non_zero_dep = [dep for dep in self.dependency_graph[len(self.nodes) + 1] if dep != 0]

            return self.out(torch.cat([outputs[i] for i in non_zero_dep] + [outputs[0]], dim=1))

        if self.out_channel_flag:
            # Same as above, we just don't worry about the 0th node.
            return self.out(torch.cat([outputs[i] for i in self.dependency_graph[len(self.nodes) + 1]], dim=1))

        return self.out(torch.cat([outputs[i] for i in self.dependency_graph[len(self.nodes) + 1]]))



def phase_active(gene):
    """
    Determine if a phase is active.
    :param gene: list, gene describing a phase.
    :return: bool, true if active.
    """
    # The residual bit is not relevant in if a phase is active, so we ignore it, i.e. gene[:-1].
    return sum([sum(t) for t in gene[:-1]]) != 0

class Identity(nn.Module):
    """
    Adding an identity allows us to keep things general in certain places.
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x