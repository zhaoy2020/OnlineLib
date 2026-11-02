import torch 
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.utils.data.distributed import DistributedSampler

from torch.amp import autocast, GradScaler

from .training import MetricTracker, Visualization

import os 
import pickle 
from loguru import logger 
from tqdm.auto import tqdm 


class Trainer:
    '''
    From Trainer, train visualization per step, valid visualization per epoch.

    Demo:
    >>> trainer = Trainer()
    >>> trainer.train(epochs=3, steps=None)  # Visualization per epoch
    >>> trainer.train(epochs=3, steps=10)    # Visualization per 10 steps
    '''

    def __init__(
            self, 
            train_dataset: torch.utils.data.Dataset | None = None, 
            val_dataset: torch.utils.data.Dataset | None = None, 
            batch_size: int = 128,
            model: torch.nn.Module | None = None, 
            loss_fn: torch.nn.Module | None = None, 
            optimizer: torch.optim.Optimizer | None = None, 
            is_tqdm: bool = True, 
            callbacks: list | None = [],
            use_amp: bool = False,
            clip_grad: bool = False,  # 是否使用梯度裁剪
    ) -> None:
        # basic sets
        self.device = self._get_device()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model = model
        self.loss_fn = loss_fn 
        self.optimizer = optimizer
        self.is_tqdm = is_tqdm
        self.callbacks = callbacks
        self.use_amp = use_amp
        self.clip_grad = clip_grad  # 是否使用梯度裁剪

        # Init with DDP 
        ## Processed group
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo', init_method='env://')
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(self.local_rank)
            logger.info(f'[rank {self.rank}] Init process group rank, world_size:{self.world_size}, local_rank:{self.local_rank}...')
        ## Model with DDP 
        self.model = self.model.to(self.local_rank)
        self.model = DDP(self.model, device_ids=[self.local_rank] if torch.cuda.is_available() else None)
        logger.info(f'[rank {self.rank}] Model with DDP ...')
        ## Dataloader with DDP
        self.train_sampler = DistributedSampler(self.train_dataset)
        self.val_sampler = DistributedSampler(self.val_dataset)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, sampler=self.train_sampler, pin_memory=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, sampler=self.val_sampler, pin_memory=True)
        logger.info(f'[rank {self.rank}] Dataloader with DDP ...')

        # mixed precision
        if self.use_amp:
            self.scaler = GradScaler() # ✅ AMP 的 GradScaler
            logger.info(f"[rank {self.rank}] Using AMP (Automatic Mixed Precision) for training.")

        # clip_grad 
        if self.clip_grad:
            logger.info(f"[rank {self.rank}] Using gradient clipping with max norm of 1.0") 

        # set metrics_tracker 
        self.metrics_tracker = MetricTracker()
        self.metrics_tracker.add_metric('loss', lambda **kw: kw['loss'])  # 直接从kwargs获取loss
        self.metrics_tracker.add_metric('acc', lambda **kw: (kw['y_hat'].argmax(1) == kw['y']).float().mean().item())

        # visualization
        self.visualization = Visualization()

    def _get_device(self):
        '''Set the deive on GPUs or CPU.'''
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Runing on {device} ...")
        return device
    
    def _disable_visualization(self):
        '''Show tqdm.'''
        if self.is_tqdm and self.rank == 0:
            return False 
        else:
            return True
        
    def _call_callbacks(self, method_name, **kwargs):
        '''Run the method of callback from callback dict with default order.'''
        for callback in self.callbacks:
            if hasattr(callback, method_name):
                method = getattr(callback, method_name, f"{method_name} is not exist!")
                return method(trainer=self, **kwargs)

    def _move_to_device(self, batch):
        '''Move value to device among different datatype (Dict, List, Tuple)'''
        if isinstance(batch, dict):
            # If batch is a dictionary, move each tensor to the device
            # This is common in NLP tasks where the batch is a dict of tensors
            # e.g., {'input_ids': tensor, 'attention_mask': tensor, 'labels': tensor}
            # Ensure that 'labels' is included in the batch
            if 'labels' not in batch:
                raise ValueError("Batch must contain 'labels' key for training.")
            return {key:value.to(self.device, non_blocking=True) for key, value in batch.items()}
        
        elif isinstance(batch, (list, tuple)):
            # If batch is a list or tuple, assume it's a tuple of tensors
            # e.g., (input_tensor, target_tensor)
            if len(batch) < 2:
                raise ValueError("Batch must contain at least two elements: input and labels.")
            return {'inputs': batch[0].to(self.device, non_blocking=True), 'labels': batch[-1].to(self.device, non_blocking=True)}
        
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}. Expected dict, list, or tuple.")

    def train(self, epochs: int=2, steps: int | None=None, **kwargs):
        '''Main loop.
        >>>train(epochs= 3, steps=None)
        >>>train(epochs= 3, steps=10)
        '''
        self._call_callbacks(method_name= "on_train_begin", **kwargs)

        with tqdm(range(epochs), desc= "Training epoch", unit= "epoch", disable= self._disable_visualization()) as pbar:
            steps_counter = 1
            for epoch in pbar:
                # train
                self._call_callbacks(method_name= "on_epoch_begin", **kwargs)

                # 在 DDP 下，sampler 每个 epoch 都需要设置 epoch 保证 shuffle 不同
                self.train_dataloader.sampler.set_epoch(epoch)

                # 训练阶段
                steps_counter = self._train_step(steps, steps_counter, pbar)

                # 获取训练的平均指标 (每个 rank)
                train_logs = self.metrics_tracker.train_average_metrics(level='epoch')
                train_logs = {"train_"+name: value for name, value in train_logs.items()}

                # 验证阶段
                if steps is None:
                    # 每个 rank 计算 y_hat 和 loss
                    self._validate_step() 
                    val_logs = self.metrics_tracker.average_metrics(stage='val', show_level='epoch')
                    val_logs = {"val_"+name: value for name, value in val_logs.items()}

                    # 聚合验证的结果到 rank=0
                    if self.rank == 0:
                        # visualization
                        ## on pbar
                        pbar.set_postfix({**train_logs, **val_logs})
                        ## on plot
                        if self._disable_visualization():
                            self.visualization.refresh_plot(history= self.metrics_tracker.get_history())

                self._call_callbacks(method_name= "on_epoch_end", **kwargs)

        self._call_callbacks(method_name= "on_train_end", **kwargs)
        
        # 同步所有进程
        dist.barrier() # 同步所有进程
        dist.destroy_process_group()  # 销毁进程组

    def _train_step(self, steps, steps_counter, pbar, **kwargs):
        '''On train step.'''
        self.model.train() 

        for batch in self.train_dataloader:
            self._call_callbacks(method_name= "on_step_begin", **kwargs)

            batch = self._move_to_device(batch)
            y = batch['labels']
            self.optimizer.zero_grad()

            # ✅ AMP 混合精度训练逻辑
            if self.use_amp:
                with autocast(device_type=self.device.type):
                    y_hat = self.model(**batch)
                    loss = self.loss_fn(y_hat, y)
                self.scaler.scale(loss).backward()

                # === 监控并剪切梯度 ===
                if self.clip_grad:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    # if grad_norm > 1.0:
                    #     logger.debug(f"[GradClip] Clipped: grad norm = {grad_norm:.4f}")
                else:
                    # grad_norm = None
                    self.scaler.step(self.optimizer)
                    self.scaler.update()  
            else:
                y_hat = self.model(**batch)
                loss = self.loss_fn(y_hat, y)
                loss.backward()

                # === 监控并剪切梯度 === 
                if self.clip_grad:   
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    if grad_norm > 1.0:
                        logger.debug(f"[GradClip] Clipped: grad norm = {grad_norm:.4f}")
                else: 
                    self.optimizer.step()

            self.metrics_tracker.train_update(y_hat= y_hat, y= y, loss= loss.item()) ## update for train

            # update show progress bar or visualization
            if (steps != None) and (steps_counter%steps == 0):
                train_logs = self.metrics_tracker.train_average_metrics(level='step') ## on step level with train
                train_logs = {"train_"+name: value for name, value in train_logs.items()}
                # val
                self._validate_step()
                val_logs = self.metrics_tracker.average_metrics(stage='val', show_level='step')
                val_logs = {"val_"+name: value for name, value in val_logs.items()}
                pbar.set_postfix({**train_logs, **val_logs})
                if self._disable_visualization():
                    self.visualization.refresh_plot(history= self.metrics_tracker.get_history())

            self._call_callbacks(method_name= "on_step_end", **kwargs)
            steps_counter += 1
        return steps_counter
    
    def _validate_step(self):
        '''On validate step.'''
        if dist.is_initialized():
            dist.barrier()  # 同步，确保所有进程都进入验证阶段

        self.model.eval()
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = self._move_to_device(batch)
                y = batch['labels']

                # ✅ optional: 在验证阶段也使用自动混合精度进行推理（更快）
                if self.use_amp:
                    with autocast(device_type=self.device.type):
                        y_hat = self.model(**batch)
                else:
                    y_hat = self.model(**batch)

                loss = self.loss_fn(y_hat, y)

                self.metrics_tracker.update(stage='val', y_hat= y_hat, y= y, loss= loss.item())
        if dist.is_initialized():
            dist.barrier()  # 验证结束再同步

    def predict(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader):
        '''Predict with the model.
        Demo:
        >>> all_labels, all_preds = trainer.predict(model=model, dataloader= dataloader)
        '''
        model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_to_device(batch)
                inputs, labels = batch['inputs'], batch['labels']
                outputs = model(inputs)
                _, preds = torch.max(outputs, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
    
        return all_labels, all_preds

    def save_metrics(self, file_path):
        '''Save the history with pickle format.'''
        if self.rank == 0:
            history = self.metrics_tracker.get_history()
            with open(file_path, 'wb') as f:
                pickle.dump(history, f)
            logger.debug(f"[Rank 0] Metrics saved to {file_path}")

    def save_checkpoint(self, file_path):
        '''Save checkpoint.'''
        if self.rank == 0:
            # DDP 模型要用 .module.state_dict() 保存（方便单卡/多卡通用）
            model_state = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()
            checkpoint = {
                'model_state_dict': model_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            torch.save(checkpoint, file_path)
            logger.debug(f"[Rank 0] Checkpoint saved to {file_path}")

    def load_checkpoint(self, file_path):
        '''Load checkpoint.'''
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.local_rank}
        if dist.get_rank() == 0:
            logger.debug(f"[Rank 0] Loading checkpoint from {file_path} ...")
            checkpoint = torch.load(file_path, map_location=map_location)
        else:
            checkpoint = None 

        # 广播 checkpoint 到所有进程
        checkpoint = dist.broadcast_object_list([checkpoint], src=0)[0]

        # 处理 DDP 的module. 前缀问题
        model_state_dict = checkpoint['model_state_dict']
        try: 
            self.model.load_state_dict(model_state_dict)
        except RuntimeError:
            # 如果保存的是 model.module.state_dict()
            from collections import OrderedDict 
            new_state_dict = OrderedDict()
            for k, v in model_state_dict.items():
                name = k.replace('module.', '')
                new_state_dict[name] = v 
            self.model.load_state_dict(new_state_dict)

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logger.debug(f"[Rank {dist.get_rank()}] Checkpoint loaded successfully.")


# trainer.metrics_tracker._history
# trainer.metrics_tracker.get_history()
# trainer.save_metrics(file_path= "./cache/metrics_tracker_history.pickle")
# trainer.save_checkpoint(file_path= './cache/checkpoint.pt')
# trainer.load_checkpoint(file_path= './cache/checkpoint.pt')


class Trainer2(Trainer):
    '''Trainer with step-based records.'''

    def __init__(self):
        super().__init__()

    def train(self, num_steps: int = 30, interval_steps: int = 10):
        '''Train the model.
        >>> trainer.train(num_steps=30, interval_steps=10)
        '''
        self.model.train()
        step, num_steps_reached = 0, False
        steps_per_epoch = len(self.train_dataloader)
        # progress bar
        progress_bar = tqdm(total=num_steps, desc='Training', unit='step', disable=self._disable_visualization(), position=1, leave=True)
        while step < num_steps and not num_steps_reached:
            for batch in self.train_dataloader:
                # ✅ 在 DDP 下，sampler 每个 epoch 都需要设置 epoch 保证 shuffle 不同
                # self.train_dataloader.sampler.set_epoch(step)
                # ✅ 使用虚拟 epoch 控制 shuffle
                virtual_epoch = step // steps_per_epoch
                self.train_dataloader.sampler.set_epoch(virtual_epoch)

                batch = self._move_to_device(batch)
                y = batch['labels']

                self.optimizer.zero_grad() 

                # ✅ AMP 混合精度训练逻辑
                if self.use_amp:
                    with autocast(device_type=self.device.type):
                        y_hat = self.model(**batch)
                        loss = self.loss_fn(y_hat, y)
                    self.scaler.scale(loss).backward()
                    # === 监控并剪切梯度 ===
                    if self.clip_grad:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    else:
                        grad_norm = None
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    y_hat = self.model(**batch)
                    loss = self.loss_fn(y_hat, y)
                    loss.backward()
                    # === 监控并剪切梯度 ===
                    if self.clip_grad:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    else:
                        grad_norm = None
                    self.optimizer.step()

                # update train metrics
                self.metrics_tracker.train_update(y_hat=y_hat, y=y, loss=loss.item())

                # validation 
                if step % interval_steps == 0:
                    train_logs = self.metrics_tracker.train_average_metrics(level='step')
                    train_logs = {'train_'+name: value for name, value in train_logs.items()}
                    self._validate_step()
                    val_logs = self.metrics_tracker.average_metrics(stage='val', show_level='step')
                    val_logs = {"val_"+name: value for name, value in val_logs.items()
                    progress_bar.set_postfix({**train_logs, **val_logs})

                # update train progress bar
                progress_bar.update(1)

                # update step
                step += 1
                if step == num_steps:
                    num_steps_reached = True
                    break
        progress_bar.close()

    def _validate_step(self):
        '''On validate step.'''
        self.model.eval()
        with torch.no_grad():
            # progress bar
            progress_bar = tqdm(total=len(self.val_dataloader), desc='Validation', unit='step', disable=self._disable_visualization(), position=1, leave=False)
            for batch in self.val_dataloader:
                batch = self._move_to_device(batch)
                y = batch['labels']
                # optional: AMP
                if self.use_amp:
                    with autocast(device_type=self.device.type):
                        y_hat = self.model(**batch)
                        loss = self.loss_fn(y_hat, y)
                else:
                    y_hat = self.model(**batch)
                    loss = self.loss_fn(y_hat, y)

                # update metrics
                self.metrics_tracker.update(state='val', y_hat=y_hat, y=y, loss=loss.item())

                # update progress bar
                progress_bar.update(1)
            progress_bar.close()