import torch.nn as nn
import torch
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_param = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(
            self.linear_proj.weight, gain=nn.init.calculate_gain("relu")
        )
        nn.init.xavier_normal_(
            self.attn_param.weight, gain=nn.init.calculate_gain("relu")
        )

    def edge_attention(self, edges):
        concat = torch.cat([edges.src["W_h"], edges.dst["W_h"]], dim=1)
        e = self.attn_param(concat)
        return {"e": F.relu(e)}

    def message_func(self, edges):
        return {"W_h": edges.src["W_h"], "e": edges.data["e"]}

    def reduce_func(self, nodes):
        alpha = F.softmax(
            nodes.mailbox["e"], dim=1
        )  # this is the attention coefficient

        h_N = torch.sum(alpha * nodes.mailbox["W_h"], dim=1)
        return {"h_N": h_N}

    def forward(self, g, h):
        with g.local_scope():
            # g.ndata['h'] = h

            W_h = self.linear_proj(h)
            g.ndata["W_h"] = W_h

            # calculate attention coefficient and store in edges
            g.apply_edges(self.edge_attention)

            # cal h_N
            g.update_all(self.message_func, self.reduce_func)

            return g.ndata["h_N"]
