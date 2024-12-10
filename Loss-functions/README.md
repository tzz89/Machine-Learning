## Imbalance dataset
1. Focal loss: https://arxiv.org/abs/1708.02002



## Pytorch
1. For cross_entropy_loss, the default class that pytorch will ignore in the target/label is -100, we can assign paddings to this class inorder to ignore the cross_entropy. This is important in NLP token prediction tasks.