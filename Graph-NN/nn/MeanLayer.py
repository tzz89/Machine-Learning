import torch
import torch.nn as nn
import dgl


class MeanLayer(nn.Module):
    """Custom mean layer that will take mean from neighbouring nodes"""

    def __init__(self) -> None:
        super().__init__()

    def message_func(self, edges: dgl.udf.EdgeBatch) -> dict:
        """

        Refer to DGL message passing api for what is an edgebatch
        Edges are
        Args:
            edges (dgl.udf.EdgeBatch): [src, dst, data]
        returns a dictionary {'msg': torch.Tensor}
        """
        mail = edges.src["h"]
        return {"m": mail}

    def reduce_func(self, nodes: dgl.udf.NodeBatch) -> dict:
        # Get the all the tensor from mailbox
        mail = nodes.mailbox[
            "m"
        ]  # mailbox shape is [number_node_nodes with X neighbours, number_neighbours, feature_dim]
        return {"h_N": torch.mean(mail, dim=1)}  # Dim 1 is the number of neighbours

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.update_all(self.message_func, self.reduce_func)
            h_N = g.ndata["h_N"]
            return h_N
