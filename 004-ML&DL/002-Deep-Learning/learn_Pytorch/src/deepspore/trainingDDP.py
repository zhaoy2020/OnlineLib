
import torch
from torch import distributed 
from torch.utils.data.distributed import DistributedSampler 
from torch.nn.parallel import DistributedDataParallel 

import os
from loguru import logger


class DDPTrainer:
    # -------------------------
    # Distributed setup helpers
    # -------------------------
    @staticmethod
    def setup():
        '''Init process group and device index.'''

        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        torch.cuda.set_device(local_rank)
        distributed.init_process_group(backend='nccl', device_id=local_rank)

        logger.info(f'Initialized process group: rank/world_size: {rank}/{world_size}, local_rank: {local_rank}.')
        return rank, world_size, local_rank
    
    @staticmethod
    def cleanup():
        '''Destroy process group.'''

        distributed.barrier()
        logger.info(f'Destroyed process group: rank/world_size: {distributed.get_rank()}/{distributed.get_world_size()}.')
        distributed.destroy_process_group()

    @staticmethod
    def is_main_process():
        '''If the process is main process (local rank is 0).'''

        return distributed.get_rank() == 0
    
    @staticmethod
    def build_dataloader(dataset: torch.utils.data.Dataset, 
                         batch_size: int = 1, collate_fn = None,
                         stage: str = None, base_seed: int = 123
        ) -> tuple[torch.utils.data.Dataset, torch.utils.data.distributed.DistributedSampler]:
        '''DataLoader with sampler.'''

        logger.info(f'Building dataloader on rank[{distributed.get_rank()}].')
        
        # sampler
        if stage == 'train':
            sampler = DistributedSampler(dataset, shuffle=True, seed=base_seed) 
        elif stage == 'val':
            sampler = None
        elif stage == 'test':
            sampler = None 
        else:
            raise ValueError(f'ERROR: not exist {stage}.')
        
        # collate_fn
        if collate_fn is not None:
            collate = lambda x: collate_fn(x)
        else:
            collate = None

        # worker_init_fn
        def worker_init_fn(worker_id):
            # worker_info: id, seed, dataset, num_workers.
            worker_info = torch.utils.data.get_worker_info()
            seed = torch.initial_seed()                         # same the above seed value.
            logger.info(f'Rank[{distributed.get_rank()}], worker_id: {worker_id}/{worker_info.id} seed: {seed}/{worker_info.seed}.')

        # generator for sampler shuffling in DataLoader.
        g = torch.Generator()
        g.manual_seed(base_seed)
        
        # DataLoader
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size,
            sampler=sampler, collate_fn = collate_fn,
            num_workers=16, pin_memory=True, persistent_workers=True, 
            generator = g, worker_init_fn=worker_init_fn)
        
        return (loader, sampler)

    @staticmethod
    def build_model(model: torch.nn.Module, local_rank, device: torch.device, sync_bn: bool = False) -> torch.nn.Module:
        '''Model with DDP.'''

        logger.info(f'Building DDP model wrap on rank[{distributed.get_rank()}]')

        model = model.to(device)

        if sync_bn:
            # convert bn to sync_bn
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = DistributedDataParallel(model, device_ids=[local_rank])
        distributed.barrier()
        return model
    
    # -------------------------
    # Checkpoint utilities
    # -------------------------
    @staticmethod
    def save_checkpoint(state: dict, filename: str):
        # only save on local rank 0.
        torch.save(state, filename)
        logger.info(f'Saved checkpoint to : {filename} on rank[{distributed.get_rank()}].')

    @staticmethod
    def load_checkpoint(filename: str, device: torch.device):
        # load checkpoint for all GPU device.
        ckpt = torch.load(filename, map_location=device, weights_only=False)
        logger.info(f'Loading from checkpoint : {filename} on rank[{distributed.get_rank()}].')
        return ckpt
    

    # -------------------------
    # Training & validation
    # -------------------------
    def train_one_epoch(epoch, model, optimizer, scheduler, accumulation_steps, train_loader, train_sampler, device, loss_fn, metrics):
        # timer start
        epoch_start = time.time()
        model.train()

        # reset metrics at epoch start
        metrics.reset()

        loss_sum: float = 0.0
        total: int = 0
        optimizer.zero_grad() # for accumulation

        train_sampler.set_epoch(epoch)
        for step, (X, y) in enumerate(train_loader):
            X, y = [b.to(device, non_blocking=True) for b in [X, y]]
            out = model(inputs=X, attention_mask=None)
            loss = loss_fn(out, y)
            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # update()
            metrics.update(out, y)

            loss_sum += (loss.item() * X.size(0)) # X.size(0) is Batch size
            total += X.size(0)

        # scheduler update
        scheduler.step()

        # aggregate across processes
        tensor = torch.tensor([loss_sum, total], device=device, dtype=torch.float64)
        distributed.all_reduce(tensor, op=distributed.ReduceOp.SUM)
        loss_sum, total = tensor.tolist()

        # compute()
        avg_loss = loss_sum / total
        # metrics_dict is a mapping like {'acc': tensor(...), 'prec': tensor(...), ...}
        metrics_dict = metrics.compute()  # this will sync across processes because sync_on_compute=True
        metrics_dict = {k:v.item() for k, v in metrics_dict.items()} # tranformer tensor to number.

        # reset() 保持干净, ready for next phase
        metrics.reset()

        # timer stop.
        duration = time.time() - epoch_start

        # add other metrics to metrics_dict; time, loss and etc.
        metrics_dict.update({'train time': duration, 'train loss': avg_loss})
        
        return metrics_dict

    def validate(model, device, valid_loader, loss_fn, metrics):
        # timer start
        val_start = time.time()
        model.eval()

        # reset metrics at epoch start
        metrics.reset()

        # loss_fn = torch.nn.CrossEntropyLoss() # 默认reduction='mean'

        loss_sum: float = 0.0
        total: int = 0

        with torch.no_grad():
            for X, y in valid_loader:
                X, y = [b.to(device, non_blocking=True) for b in [X, y]]
                out = model(inputs=X, attention_mask=None)
                loss = loss_fn(out, y)

                # update(), out is logits.
                metrics.update(out, y)

                loss_sum += loss.item() * X.size(0) # batch numbers every batch may be different, owning to droplast = False
                total += X.size(0)

        # aggregate across processes
        tensor = torch.tensor([loss_sum, total], device=device, dtype=torch.float64)
        distributed.all_reduce(tensor, op=distributed.ReduceOp.SUM)
        loss_sum, total = tensor.tolist()

        # compute()
        avg_loss = loss_sum / total
        # metrics_dict is a mapping like {'acc': tensor(...), 'prec': tensor(...), ...}
        metrics_dict = metrics.compute()
        metrics_dict = {k:v.item() for k, v in metrics_dict.items()} # Transformer tensor to number.

        # reset() 保持干净
        metrics.reset()

        # timer stop.
        duration = time.time() - val_start

        # add other metrics to metrics_dict; time, loss and etc.
        metrics_dict.update({'val time': duration, 'val loss': avg_loss})
        
        return metrics_dict 
    
    def DDPtrainer(epochs: int = 3, accumulation_steps: int = 1, 
               resume: bool = False, 
               checkpoint_file_path: str = 'checkpoint.pt', log_file_path: str = 'train_logs.csv'):
        '''Trainer with PyTorch DDP.
        Example:
        >>> DDPtrainer(epochs=50, accumulation_steps=5, checkpoint_file_path='checkpoint.pt')
        '''

        # init process group and device index.
        rank, world_size, local_rank = setup()
        device = torch.device(f'cuda:{local_rank}')

        # metrics（sync_on_compute=True 自动同步各 GPU）
        num_classes = 2
        metrics = MetricCollection({
            "accuracy": Accuracy(task="multiclass", num_classes=num_classes, average='micro', sync_on_compute=True),
            "precision": Precision(task="multiclass", num_classes=num_classes, average='macro', sync_on_compute=True),
            "recall": Recall(task="multiclass", num_classes=num_classes, average='macro', sync_on_compute=True),
            "F1 score": F1Score(task="multiclass", num_classes=num_classes, average='macro', sync_on_compute=True),
        }).to(device)
        train_metrics = metrics.clone(prefix='train ') # space
        val_metrics = metrics.clone(prefix='val ') # sapce
        metrics_history: list = []

        # data
        train_loader, train_sampler = build_dataloader(train_df, batch_size=1, stage='train', is_augmentation=True)
        valid_loader, _ = build_dataloader(valid_df, batch_size=1, stage='val', is_augmentation=False)
        # test_loader, _ = build_dataloader(test_df, batch_size=1, stage='test', is_augmentation=False)

        # model
        model = build_model(local_rank=local_rank, device=device, sync_bn=True)

        # loss_fn, optimizer and scheduler
        loss_fn = torch.nn.CrossEntropyLoss() # default, reduction="mean"
        optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)

        # resume, after model, optimizer, scheduler are declared.
        init_epoch: int = 1
        if resume and (checkpoint_file_path is not None):
            checkpoint = load_checkpoint(checkpoint_file_path, device)
            model.module.load_state_dict(checkpoint['model_state_dict'])    # model
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])   # optimizer
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])   # scheduler
            init_epoch = checkpoint['epoch'] + 1 # continue next epoch      # epoch

        # grad accumulation
        accumulation_steps = max(1, accumulation_steps)

        # traing loop
        for epoch in range(init_epoch, epochs + 1):
            # training
            train_metrics_dict = train_one_epoch(
                epoch, model, optimizer, scheduler, accumulation_steps, train_loader, train_sampler, device, loss_fn, train_metrics
            )

            # validation
            val_metrics_dict = validate(model, device, valid_loader, loss_fn, val_metrics)

            # ---- ✅ 主进程写 CSV ----
            # save on local rank 0.
            if is_main_process():
                # metrics_history for subsquent visulation.
                log_entry = {}
                log_entry.update({'epoch': epoch, 'learning rate': optimizer.param_groups[0]["lr"]})
                log_entry.update(train_metrics_dict)
                log_entry.update(val_metrics_dict)
                metrics_history.append(log_entry)

                df = pd.DataFrame(metrics_history)
                df.to_csv(log_file_path, index=False)

                # print(
                #     f'Epoch {epoch}/{epochs}: train/val={train_time:.2f}/{val_time:.2f}s. '
                #     f'train_loss/val_loss={train_loss:.2f}/{val_loss:.2f}, '
                #     f'train_acc/val_acc={train_acc:.2f}/{val_acc:.2f}, '
                #     f'train_prec/val_prec={train_prec:.2f}/{val_prec:.2f}, '
                #     f'train_recall/val_recall={train_recall:.2f}/{val_recall:.2f}, '
                #     f'train_f1/val_f1={train_f1:.2f}/{val_f1:.2f}, '
                #     f'lr={optimizer.param_groups[0]["lr"]:.2e}'
                # )

                # print on screen.
                contents = [f'Epoch {epoch}/{epochs}:']
                for (tk, tv), (vk, vv) in zip(train_metrics_dict.items(), val_metrics_dict.items()):
                    content = f'{tk}/{vk}={tv:.2f}/{vv:.2f}'
                    contents.append(content)
                contents.append(f'lr={optimizer.param_groups[0]["lr"]:.2e}')
                print('\t'.join(contents))
                
                # save checkpoint.
                state = {
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                }
                save_checkpoint(state, checkpoint_file_path)
        
        # test, if needed. (test_loader with validation step code.)
        # test_metrics_dict = validate(model, device, test_loader, loss_fn, val_metrics)
        
        # destroy process group
        cleanup()
        logger.info(f'Training finished on local_rank[{local_rank}].')
