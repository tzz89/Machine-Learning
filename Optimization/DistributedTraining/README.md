## Distributed training
1. In each training step, all the gradients are syncronized using all reduce
2. In multinode setup there are 2 types of ranks
   1. local rank: starts from zero for each node #int(os.environ["LOCAL_RANK"])
   2. global rank: unqiue for each GPU           #int(os.environ["RANK"])

##Questions
1. in Multinode training, all nodes have to do snapshot?

## Required code changes
1. import torch.multiprocessing as mp
2. from torch.utils.data.distributed import DistributedSampler
3. from torch.nn.parallel import DistributedDataParallel as DDP # main work horse for wrapping model
4. from torch.distributed import init_process_group, destroy_process_group

```
def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = XXXXX ## We dont need this for torchrun
    os.environ["MASTER_PORT"] = XX    ## We dont need this for torchrun
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def trainer(model, train_data, optimizer, gpu_id):
    ## Wrap model in DDP
    model = DDP(model, device_id=[gpu_id])

    ## Add condition on saving model
    if gpu_id == 0 and epoch % xx == 0:
        torch.save(model.state_dict())

def prepare_dataloader(dataset):
    return DataLoader(
        dataset,
        batch_size,
        pin_memory=True,
        shuffle=False, ## Turn shuffle to off
        sampler = DistributedSampler() ## need to this sampler
    )

def main(device->rank, world_size, total_epoch, save_every):
    ddp_setup(rank, world_size)
    dataset, model, optimzer = load_train_objs()
    train_data = prepare_dataloader()
    trainer = Trainer(model, train_data, optimizer,rank)
    trainer.train

    destroy_process_group() # use this to stop the distributed training

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size) ## mp.spawn will automatically assign rank/gpu_id to main

```

## Fault tolerance using torch_run
1. We just need to  run init_process_group(backend='nccl') # dont need the other parameters and environment variables
2. gpu_id can be accessed from environ variable using os.environ["LOCAL_RANK"]
3. torch_run will help save a snapshot of the training, we can pass a path to the trainer init and if the snapshot path exists, we can load from the snapshot path
```
def _load_snapshot(self, snapshot_path):
    snapshot = torch.load(snapshot_path)
    self.model.load_state_dict(snapshot["MODEL_STATE"])
    self.epochs_run = snapshot["EPOCH_RUNS"]
    print(f"resuming training from {epochs_run}")

def _save_snapshot(self, epoch):
    snapshot = {}
    snapshot["MODEL_STATE"] = self.model.module.state_dict()
    snapshot["EPOCHS_RUN"] = epoch
    torch.save("checkpoint.pt")
    print(f"Epoch {epoch} | Training checkpoint saved at snapshot.pt")
```
4. We can remove mp.spawn from main and let torchrun handle it 
```
example of torchrun command line 
torchrun --standalone --nproc_per_node=gpu script.py
```

5. For multinodes training
we have to run torchrun in all nodes 
Alternatively, we can use slurm
```
Sample torchrun command
torchrun --nproc_per_node=gpu --nnodes=2 --node_rank=0/1/ --rdzv_backend=c10d --rdzv_enpoint=172.xxx.xxxx.xx script.py

```


## Tutorials
1. Introduction to pytorch DDP: https://www.youtube.com/watch?v=Cvdhwx-OBBo&list=PL_lsbAsL_o2CSuhUhJIiW0IkdT5C2wGWj&index=2
2. Hands on DDP in pytorch: https://www.youtube.com/watch?v=toUSzwR0EV8&t=315s