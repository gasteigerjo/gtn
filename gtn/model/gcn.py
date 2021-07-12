import torch
import torch.nn as nn
from lcn.utils import scatter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import zeros


class GCNConv(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        act_fn,
        bilin_transform,
        bias=False,
        use_linear=True,
        aggr="add",
        avg_degree=1.0,
        deg_norm=False,
        **kwargs
    ):
        super(GCNConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_fn = act_fn
        self.deg_norm = deg_norm

        # Renormalization factor that prevents embedding magnitude changes
        # between layers when not using degree normalization
        if self.deg_norm:
            self.renorm_factor = 1.0
        else:
            self.renorm_factor = 1.0 / avg_degree

        if use_linear:
            self.fc = nn.Linear(in_channels, out_channels)
        else:
            assert in_channels == out_channels
            self.fc = lambda x: x

        self.bilin = bilin_transform

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        zeros(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = self.act_fn(self.fc(x))

        if self.deg_norm:
            # Use symmetric normalization for aggregation
            row, col = edge_index
            deg = scatter(x.new_ones(1).expand_as(row), row, dim=0, dim_size=x.shape[0])
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

            norm_degree = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        else:
            norm_degree = None

        return x, self.propagate(
            edge_index,
            x=x,
            norm={"edge_weight": edge_weight, "norm_degree": norm_degree},
        )

    def message(self, x_j, norm):
        if norm["norm_degree"] is not None:
            x_j = x_j * norm["norm_degree"][:, None]
        if norm["edge_weight"] is None:
            return x_j
        else:
            return self.bilin(x_j, norm["edge_weight"])

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out * self.renorm_factor

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )
