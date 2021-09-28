from .nsga_net_node import *

class DensePhase(Module):
    pass

class ResidualPhase(Module):
    def __init__(self,
                 supernet, 
                 encoder, 
                 in_channels, 
                 out_channels,
                 kernel_size, 
                 idx, 
                 preact=False):
        super(ResidualPhase, self).__init__()

        self.channel_flag = in_channels != out_channels
        self.first_conv = nn.Conv2d(in_channels, 
                                    out_channels, 
                                    kernel_size=kernel_size,
                                    stride=1,
                                    bias=False)

        self.dependency_graph = ResidualPhase.build_dependency_graph(encoder)

        node_type = 'res' if not preact else 'res_pre'
        nodes = []
        for i in range(len(encoder)):
            if len(self.dependency_graph[i+1]) > 0:
                nodes.append(supernet.module_dict[''])


    @staticmethod
    def build_dependency_graph(self, encoder):
        pass

    def forward(self, x):
        if self.channel_flag:
            x = self.first_conv(x)


        