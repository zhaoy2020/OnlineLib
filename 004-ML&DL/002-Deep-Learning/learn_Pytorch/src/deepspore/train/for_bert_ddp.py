import torch 
import torch.distributed as dist 
from torch.utils.data import DistributedSampler, DataLoader 
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.amp import autocast, GradScaler 

from Pytorch.src.deepspore.datas.dataset.bert_wiki_dataset import load_data_wiki
from Pytorch.src.deepspore.models.bert import BERT 
from Pytorch.src.deepspore.loss.bert_mlm_loss import _get_batch_loss_bert 
from Pytorch.src.deepspore.training import Toolkits, parameter_sizes

from collections import defaultdict
import os 
from loguru import logger 
import dataclasses


def train(
        net: torch.nn.Module,
        loss: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        optimizer: torch.optim.Optimizer,
        num_steps: int,
        batch_size: int,
        vocab_size: int,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        use_amp: bool = False 
):
    # 1. init
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    # 2. datas
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8, sampler=sampler, pin_memory=True)   
    # 3. model
    net = net.to(local_rank)
    net = DDP(net, device_ids=[local_rank], output_device=local_rank)

    # training
    if use_amp:
        scaler = GradScaler()
    storage = defaultdict(list)
    step, num_step_reached = 0, False
    while step < num_steps and not num_step_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X, mlm_Y, nsp_y in train_dataloader:
            # 4. DDP sampler shuffling seed
            vir_epoch = step // len(train_dataloader)
            train_dataloader.sampler.set_epoch(vir_epoch)

            # move to device
            tokens_X = tokens_X.to(local_rank)
            segments_X = segments_X.to(local_rank)
            valid_lens_x = valid_lens_x.to(local_rank)
            pred_positions_X = pred_positions_X.to(local_rank)
            mlm_weights_X = mlm_weights_X.to(local_rank)
            mlm_Y, nsp_y = mlm_Y.to(local_rank), nsp_y.to(local_rank)
            # get loss value, backward and step update
            optimizer.zero_grad()
            if use_amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    mlm_l, nsp_l, l = _get_batch_loss_bert(
                        net, loss, vocab_size, 
                        tokens_X, segments_X, valid_lens_x, 
                        pred_positions_X, mlm_weights_X, mlm_Y, nsp_y
                    )
                scaler.scale(l).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                mlm_l, nsp_l, l = _get_batch_loss_bert(
                    net, loss, vocab_size, 
                    tokens_X, segments_X, valid_lens_x, 
                    pred_positions_X, mlm_weights_X, mlm_Y, nsp_y
                )
                l.backward()
                optimizer.step()

            # update step-based scheduler
            if scheduler is not None:
                scheduler.step()

            # visualization
            if rank == 0 and step % 100 == 0:
                print(f'Step {step}, MLM Loss: {mlm_l.item()}, NSP Loss: {nsp_l.item()}, Loss: {l.item()}')
                storage['step'].append(step)
                storage['mlm_loss'].append(mlm_l.item())
                storage['nsp_loss'].append(nsp_l.item())
                storage['loss'].append(l.item())
                Toolkits.save_to_pkl(storage, f'cache/bert_large_logs/train_metrics_logs.pkl')
                
            if rank == 0 and step % 10000 == 0:
                Toolkits.save_checkpoint(net, optimizer, step, l.item(), f'cache/bert_large_logs/bert_{step}.pth')

            step += 1
            if step == num_steps:
                num_step_reached = True
                break

    # 5. destroy process group
    dist.destroy_process_group()


def main(config: dataclasses.dataclass):
    # dataset
    batch_size, max_len, num_steps = config.batch_size, config.max_len, config.num_steps
    _, vocab, train_dataset = load_data_wiki(batch_size, "./data/wikipedia_text", max_len)

    # Init the model with configure:
    net = BERT(
        vocab_size=len(vocab), 
        hidden_size=config.hidden_size, 
        num_layers=config.num_layers, 
        num_heads=config.num_heads, 
        dim_feedforward=config.dim_feedforward,
    )
    logger.info(f"Model size: {parameter_sizes(net)[2]}")

    # loss function
    loss = torch.nn.CrossEntropyLoss(reduction="mean") # For per-token loss 

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)

    # scheduler: 线性 warmup + 余弦衰减
    from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_steps), # 0.1 * 10000
        num_training_steps=num_steps
    )

    # training
    train(net, loss, train_dataset, optimizer, num_steps=num_steps, batch_size=batch_size, vocab_size=len(vocab), scheduler=scheduler, use_amp=False)


@dataclasses.dataclass
class BERTBaseConfig:
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dim_feedforward: int = 3072
    batch_size: int = 64
    max_len: int = 512
    lr: float = 1e-4
    num_steps: int = 100_000


@dataclasses.dataclass
class BERTLargeConfig:
    hidden_size: int = 1024
    num_layers: int = 24
    num_heads: int = 16
    dim_feedforward: int = 4096
    batch_size: int = 32 # avoid OOM
    max_len: int = 512
    lr: float = 1e-4
    num_steps: int = 1_000_000


if __name__ == "__main__":
    main(config= BERTLargeConfig())
    # Launch command with 2 GPUs in a single node (machine):
    # OMP_NUM_THREADS=16 torchrun --nnodes 1 --nproc-per-node 2 ./utils/deepspore/deepspore/train/for_bert_ddp.py
