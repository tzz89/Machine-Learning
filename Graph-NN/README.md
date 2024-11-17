


## Key Concepts in DGL
To simultaneously update all node features, the following 3 functions are used
1. message_func(edges): each edge has attributes edge = [src, dst, data]. it stores everything that needs to be done in a dictionary called mailbox which can be accessed via "nodes.mailbox"
2. reduce_func(nodes): update node features using the update equation
3. g.update_all(message_func, reduce_func) sends message through all edges and update features of all nodes
4. To implement custom layer, we need to define the message_func and reduce_func and implement the forward(graph, h) method
   1. Refer to Graph-NN.nn.MeanLayer for more information for sample implementation
   2. Most of the time, we will use DGL built in functions as they are highly optimized
5. 


## Tutorial references
1. dgl basics: https://www.youtube.com/watch?v=RABd6rnI84Y&list=PL8ser0zRo_NhRIoBkBcSdUooNZdrM5WkI
   1. https://github.com/dtdo90/Deep_Graph_Library_Tutorials/blob/main/01_gcn.ipynb