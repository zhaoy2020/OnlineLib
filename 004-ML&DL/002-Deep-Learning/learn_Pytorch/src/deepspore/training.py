import os 
import random 
import pickle 

import torch
from torch.amp import autocast, GradScaler
from torch import distributed 
from torch.utils.data.distributed import DistributedSampler 
from torch.nn.parallel import DistributedDataParallel 

import numpy as np 
import matplotlib 
import matplotlib.pyplot as plt 
import seaborn as sns 
from collections import defaultdict
from IPython import display
from tqdm.auto import tqdm 
import time
import gc 

from loguru import logger 


def set_seed(seed: int = 123)-> None:
    '''Function for setting the seed.
    Args:
        seed: int, default is 123.

    Examples:
    >>> set_seed(seed= 123)
    '''

    # Set the seed for Python's built-in random module
    random.seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)
    
    # Set the seed for PyTorch on CPU device
    torch.manual_seed(seed)

    # Set the seed for PyTorch on GPU device
    if torch.cuda.is_available():  
        # For current GPU device
        torch.cuda.manual_seed(seed)
        # For all GPUs device
        torch.cuda.manual_seed_all(seed)

    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Set seed {seed} for reproducibility.")
    
    return None


def try_gpu(i:int=0):
    '''Try get gpu.
    >>> try_gpu(i=0)
    '''
    if torch.cuda.is_available():
        return torch.device(f"cuda:{i}")
    return torch.device('cpu')


def try_all_gpus():
    '''Try all GPUs.
    >>> try_all_gpus()
    '''
    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


def clean_device_caches(values: list = None) -> None:
    if values is not None:
        for value in tqdm(values, desc='Clean for CPU/GPUs'):
            if value is not None:
                del value 

    # garbage clean for Python
    # print("Garbage clean for CPU.")
    gc.collect()

    # garbage clean for GPU caches
    # print("Garbage clean for GPUs.")
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return None 


def get_k_fold_data(k_nums:int, i:int, X:torch.Tensor, y:torch.Tensor):
    ''' 
    k_nums: 总共的 折数 K，例如 5 折交叉验证就传 5。
    i: 当前是第几折，用作划分验证集（取值范围 0 到 k_nums-1）。
    X: 特征矩阵，形状为 (样本数, 特征数)。
    y: 标签向量，形状为 (样本数,)。
    
    Demo:
    >>> Xt, yv, Xv, yv = get_k_fold_data(k_nums=10, i=1, X=X, y=y)
    '''
    assert k_nums > 1                                       # 合法性检查,确保至少进行 2 折交叉验证。否则划分毫无意义。
    fold_size = X.shape[0] // k_nums                        # 计算每一折的大小,总样本数除以折数，得到每折包含多少个样本（取整）。
    X_train, y_train = None, None
    for j in range(k_nums):
        idx = slice(j * fold_size, (j + 1) * fold_size)     # 切片，得到当前折的起止索引范围。
        X_part, y_part = X[idx, :], y[idx]                  # 当前折的特征和标签子集。
        if j == i:
            X_valid, y_valid = X_part, y_part               # 验证集选择,如果当前折号 j 等于目标折号 i，那么就将这一折作为验证集。
        elif X_train is None:
            X_train, y_train = X_part, y_part               # 如果还没开始创建训练集，就直接赋值；
        else:
            X_train = torch.cat((X_train, X_part), dim=0)   # 否则，把当前折拼接到已有训练集中。
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid 


def labels_distribute(train_labels, valid_labels) -> matplotlib.figure.Figure | None:
    '''Visualize the distribution of labels in the dataset.'''
    if train_labels is None or valid_labels is None:
        return None
    fig, axs = plt.subplots(1, 2, figsize=(5, 2.5))
    sns.histplot(train_labels, bins=10, kde=True, label='Train Labels', ax=axs[0])
    axs[0].set_title('Train Labels Distribution')
    sns.histplot(valid_labels, bins=10, kde=True, label='Valid Labels', ax=axs[1])
    axs[1].set_title('Valid Labels Distribution')
    fig.tight_layout()
    return None


class Toolkits:
    @staticmethod
    def save_checkpoint(
            model: torch.nn.Module, 
            optimizer: torch.optim.Optimizer, 
            epoch: int, 
            loss: float, 
            filepath: str
    ):
        '''Save the model checkpoint.
        >>> save_checkpoint(model=net, optimizer=trainer, epoch=step, loss=metric[0] / metric[3], filepath=f'cache/checkpoints/bert_{step}.pth')
        '''
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")

    @staticmethod
    def load_checkpoint(
            filepath: str,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
    ) -> tuple[torch.nn.Module, torch.optim.Optimizer, int, float]:
        '''Load the model checkpoint.
        >>> load_checkpoint(filepath, model, optimizer)
        '''
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        logger.info(f"Checkpoint loaded from {filepath}")
        return model, optimizer, epoch, loss

    @staticmethod 
    def save_to_pkl(values, file_path: str) -> None:
        '''Save to pkl file.
        >>> save_to_pkl(checkpoint, f'cache/bert_logs/train_metrics_logs.pkl')
        '''
        with open(file_path, 'wb') as f:
            pickle.dump(values, f)
        logger.info(f"Data saved to {file_path}")
        return None

    @staticmethod
    def load_from_pkl(file_path: str) -> dict:
        '''Load from pkl file.
        >>> checkpoint = load_from_pkl(f'cache/bert_logs/train_metrics_logs.pkl')
        '''
        with open(file_path, 'rb') as f:
            values = pickle.load(f)
        logger.info(f"Data loaded from {file_path}")
        return values 
    
    @staticmethod
    def filter_dataset_by_labels(dataset: torch.utils.data.Dataset, keep_labels: list, max_per_class: int | None = None):
        """
        Args:
            dataset: 原始 Dataset，dataset[i] = (x, label)
            keep_labels (list[int]): 想要保留的类别
            max_per_class (int | None): 每个类别最多保留多少样本
        Return:
            Subset dataset
        Example:
        >>> small_dataset = filter_dataset_by_labels(train_dataset, keep_labels=[0,1], max_per_class=500)
        """
        indices = []
        counts = {l: 0 for l in keep_labels}

        for i in range(len(dataset)):
            _, label = dataset[i]
            if label in keep_labels:
                if max_per_class is None or counts[label] < max_per_class:
                    indices.append(i)
                    counts[label] += 1

        return torch.utils.data.Subset(dataset, indices)

    @staticmethod 
    def train_valid_split(subset: torch.utils.data.Subset, valid_ratio: float = 0.2, seed: int = 42):
        '''
        Example:
        >>> small_dataset = filter_dataset_by_labels(train_dataset, keep_labels=[0,1], max_per_class=500)
        >>> train_set, valid_set = train_valid_split(small_dataset, valid_ratio=0.2)
        '''
        n = len(subset)
        indices = np.random.RandomState(seed).permutation(n)
        split = int(n * (1 - valid_ratio))
        train_idx, valid_idx = indices[:split], indices[split:]
        train_set = torch.utils.data.Subset(subset, train_idx)
        valid_set = torch.utils.data.Subset(subset, valid_idx)
        return train_set, valid_set


class Timer:
    """Record multiple running times.
    
    Demo:
    >>> timer = Timer()
    >>> timer.start()
    >>> timer.update()
    >>> timer.sum()
    >>> timer.avg()
    >>> timer.cumsum()
    >>> timer.to_date(seconds= timer.sum())
    """

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def update(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self) -> float:
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self) -> float:
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self) -> list:
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()
    
    def to_date(self, seconds) -> None:
        '''Translate seconds to date format.'''
        days = seconds // (24 * 3600)
        hours = (seconds % (24 * 3600)) // 3600
        minutes = (seconds % 3600) // 60
        remaining_seconds = seconds % 60
        print(f"{'='*20}\n Total: \n {days} d \n {hours} h \n {minutes} m \n {remaining_seconds} s")
        return None
        
    

class Callback:
    '''
    callback template
    '''

    def on_train_begin(self, **kwargs):
        pass
    
    def on_train_end(self, **kwargs):
        pass
    
    def on_epoch_begin(self, **kwargs):
        pass
    
    def on_epoch_end(self, **kwargs):
        pass
    
    def on_step_begin(self, **kwargs):
        pass
    
    def on_step_end(self, **kwargs):
        pass


class DemoCallback(Callback):
    '''Call function demo.'''

    def on_train_begin(self, **kwargs):
        '''Runing on train stage begin.'''
        print("Runing on_train_begin ...")


## callback
class LRScheduler(Callback):
    def on_epoch_end(self, **kwargs):
        scheduler.step()
        print('LR: ', trainer.optimizer.param_groups[0]['lr'])
        print(val_logs)


class ReduceLROnPlateauCallback:
    '''Adjust the lr according to the metric.'''

    def __init__(self, scheduler, monitor='val_loss', verbose=True):
        self.scheduler = scheduler
        self.monitor = monitor
        self.verbose = verbose

    def on_epoch_end(self, trainer=None, **kwargs):
        history = trainer.metrics_tracker.get_history()

        # # 如果还没有 val_loss，跳过
        # if self.monitor not in history['epoch'] or len(history[self.monitor]) == 0:
        #     if self.verbose:
        #         print(f"[ReduceLROnPlateau] Skipping scheduler step (no {self.monitor} yet)")
        #     return

        current = history['epoch'][self.monitor][-1]
        self.scheduler.step(current)

        if self.verbose:
            current_lr = trainer.optimizer.param_groups[0]['lr']
            print(f"[ReduceLROnPlateau] {self.monitor}: {current:.6f}, lr → {current_lr:.6f}")


class CosineAnnealingLRCallback:
    def __init__(self, scheduler, verbose=True):
        self.scheduler = scheduler
        self.verbose = verbose
        logger.info('Using CosineAnnealingLRCallback.')

    def on_epoch_end(self, trainer=None, **kwargs):
        self.scheduler.step()
        if self.verbose:
            current_lr = trainer.optimizer.param_groups[0]['lr']
            print(f"[CosineAnnealing] lr → {current_lr:.6f}")


class StepLRCallback:
    def __init__(self, scheduler, verbose=True):
        self.scheduler = scheduler
        self.verbose = verbose

    def on_epoch_end(self, trainer=None, **kwargs):
        self.scheduler.step()
        if self.verbose:
            current_lr = trainer.optimizer.param_groups[0]['lr']
            print(f"[StepLR] lr → {current_lr:.6f}")


class GetModelSize:
    '''
    Calculate the parameter numbers and sizes of model.

    Demo:
    >>>get_model_size = GetModelSize()
    >>>get_model_size.parameter_numbers(model= model)
    >>>get_model_size.parameter_sizes(model= model)
    '''

    def parameter_numbers(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def parameter_sizes(self, model, dtype=torch.float32):
        bytes_per_param = torch.tensor([], dtype=dtype).element_size()
        total_params = self.parameter_numbers(model)
        total_size = total_params * bytes_per_param
        print(f'{total_params/1000000} M parameters')
        print(f'{total_size/(1024*1024):.2f} MB')
        # return total_params, total_size


def parameter_numbers(model: torch.nn.Module):
    '''Calculate the parameter numbers of model.'''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameter_sizes(model: torch.nn.Module, dtype=torch.float32):
    '''Calculate the parameter numbers and sizes of model.'''
    bytes_per_param = torch.tensor([], dtype=dtype).element_size()
    total_params = parameter_numbers(model)
    total_size = total_params * bytes_per_param
    # print(f'{total_params/1_000_000:.2f} M parameters')
    # print(f'{total_size/(1024*1024):.2f} MB')
    return (total_params/1_000_000, total_size/(1024*1024), f"Model size: {total_params/1_000_000:.6f} M parameters, {total_size/(1024*1024):.6f} MB")


class Visualization:
    '''
    The Visualization class is designed for visualization of history.

    Demo:
    >>>visualization = Visualization()
    >>>visualization.refresh_plot(history=history)
    '''

    def _get_config(self, history):
        levels = set()
        stages = set()
        metrics = set()
        for level in history.keys():
            levels.add(level)
        for i in history.values():
            if i.__len__():
                for j in i.keys():
                    stage, metric = j.split("_")
                    stages.add(stage)
                    metrics.add(metric)
        return levels, stages, metrics
    
    def _show(self, history):
        '''根据实验：train、val、test等，指标：loss、acc、f1等自动绘图'''
        levels, stages, metrics = self._get_config(history)
        if len(metrics) == 0:
            print("⚠️ No metrics to plot. Skipping visualization.")
        fig, axess = plt.subplots(nrows= len(levels), ncols= len(metrics), figsize=(5, 5))        
        for i, level in enumerate(levels):
            for j, metric in enumerate(metrics):
                for stage in stages:
                    if stage == "train":
                        kw = {'color': '#BF00BF', 'linestyle': '-'}
                    elif stage == "val":
                        kw = {'color': '#008000', 'linestyle': '--'}
                    axess[i][j].plot(history[level][f"{stage}_{metric}"], label= f"{stage}_{metric}", **kw)
                    axess[i][j].legend(fontsize="small")
                    axess[i][j].set_xlabel(level)
                    axess[i][j].set_ylabel(metric)
                    axess[i][j].set_title(f"{metric} curve")
        fig.tight_layout()         
        return fig
    
    def refresh_plot(self, history):
        '''再jupyter中持续刷新展示图片'''
        plt.close()                                 # close figure （推荐）
        fig = self._show(history)
        display.display(fig)                        # 在jupyter中展示 （推荐）
        display.clear_output(wait= True)             # 等待 （必须）


class MetricTracker:
    '''
    The MetricTracker class is designed to track and average metrics during training and validation.
    It allows for the addition of custom metrics, updates during training and validation steps,
    and provides methods to compute averages and store results in a history dictionary.
    
    Demo:
    >>>tracker = MetricTracker()
    >>>tracker.add_metric('loss', lambda **kw: kw['loss'])  # Example metric function
    >>>tracker.update(stage='val', y_hat=y_hat, y=y, loss=loss.item())
    >>>tracker.average_metrics(stage='val', show_level='step')
    >>>tracker.average_metrics(stage='val', show_level='epoch')
    >>>tracker.train_update(y_hat=y_hat, y=y, loss=loss.item())
    >>>tracker.train_average_metrics(level='step')
    >>>tracker.train_average_metrics(level='epoch')
    >>>tracker.get_history()
    >>>tracker.get_buffer()
    '''

    def __init__(self):
        '''
       _buffers = {
            'val': {
                'loss': [],
                'acc': [],
            },
            'test': {
                'loss': [],
                'acc': []
            }
        }
        _train_buffers = {
            'epoch': {
                'loss': [],
                'acc': [],
            },
            'step': {
                'loss': [],
                'acc': []
            }
        }
        history = {
            'epoch': {
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': [],
            },
            'step': {
                'train_loss': [],
                'trian_acc': [],
                'val_loss': [],
                'val_acc': [],
            },
        }
        '''
        self._metrics = {}
        # for val, test, infer just foucs on epoch level result.
        self._buffers = {}
        # for trian, has two models include epoch or step level.
        self._train_buffers = {
            'epoch': defaultdict(list),
            'step': defaultdict(list)
        }
        self.history = {
            'epoch': defaultdict(list),
            'step': defaultdict(list),
        }

    def add_metric(self, metric_name, metric_fn):
        '''Add metric.'''
        if metric_name not in self._metrics:
            self._metrics[metric_name] = metric_fn 

    def update(self, stage:str, **kwargs):
        '''Record metric results per batch step for val/test/infer and etc.'''
        # for val/test/infer which only foucs on epoch result.
        current_buffer = self._buffers.setdefault(stage, defaultdict(list))

        for metric_name, metric_fn in self._metrics.items():
            metric_result = metric_fn(**kwargs)
            current_buffer[metric_name].append(metric_result)

    def average_metrics(self, stage:str, show_level:str):
        '''Average the _buffer.'''
        current_buffer = self._buffers[stage]
        metric_results = {}
        for metric_name, values in current_buffer.items():
            value_avg = self._compute_avg(values)
            metric_results[metric_name] = value_avg
            # self.history[show_level][f'{stage}_{metric_name}_{show_level}'].append(value_avg)
            self.history[show_level][f'{stage}_{metric_name}'].append(value_avg)
        current_buffer.clear() # clear the buffer.
        return metric_results

    def train_update(self, **kwargs):
        for metric_name, metric_fn in self._metrics.items():
            metric_result = metric_fn(**kwargs)
            for level in ['epoch', 'step']: # 'epoch' or 'step'
                self._train_buffers[level][metric_name].append(metric_result)

    def train_average_metrics(self, level):
        '''Average.'''
        metric_results = {}
        for metric_name, values in self._train_buffers[level].items():
            value_avg = self._compute_avg(values)
            metric_results[metric_name] = value_avg 
            self.history[level][f'train_{metric_name}'].append(value_avg)
        self._train_buffers[level].clear() # clear the buffer.
        return metric_results 

    def _compute_avg(self, values):
        '''Compute the average.'''
        if not values:
            return 0.0  # 避免空列表
        if isinstance(values[0], (int, float)):
            return sum(values) / len(values)
        elif isinstance(values[0], torch.Tensor):
            return torch.stack(values).mean(dim=0)
        else:
            raise TypeError(f"Unsupported data type: {type(values[0])}")
        
    def get_history(self):
        '''Get the history.'''
        return self.history 
    
    def get_buffer(self):
        '''Get the buffer.'''
        return self._buffers


class Trainer:
    '''
    From Trainer, train visualization per step, valid visualization per epoch.

    Demo:
    >>>trainer = Trainer()
    >>>trainer.train(epochs=3, steps=None)  # Visualization per epoch
    >>>trainer.train(epochs=3, steps=10)    # Visualization per 10 steps
    '''

    def __init__(
            self, 
            device: str = "auto",
            train_dataloader: torch.utils.data.DataLoader | None = None, 
            val_dataloader: torch.utils.data.DataLoader | None = None, 
            model: torch.nn.Module | None = None, 
            loss_fn: torch.nn.Module | None = None, 
            optimizer: torch.optim.Optimizer | None = None, 
            is_tqdm: bool = True, 
            callbacks: list | None = [],
            use_amp: bool = False,
            clip_grad: bool = False,  # 是否使用梯度裁剪
    ) -> None:
        # basic sets
        self.device = self._get_device(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = self._get_model(device, model)
        self.loss_fn = loss_fn 
        self.optimizer = optimizer
        self.is_tqdm = is_tqdm
        self.callbacks = callbacks
        self.use_amp = use_amp
        self.clip_grad = clip_grad  # 是否使用梯度裁剪

        # mixed precision
        self.scaler = GradScaler() if self.use_amp else None  # ✅ AMP 的 GradScaler
        if self.use_amp:
            logger.info("Using AMP (Automatic Mixed Precision) for training.")

        # clip_grad 
        if self.clip_grad:
            logger.info("Using gradient clipping with max norm of 1.0") 

        # set metrics_tracker 
        self.metrics_tracker = MetricTracker()
        self.metrics_tracker.add_metric('loss', lambda **kw: kw['loss'])  # 直接从kwargs获取loss
        self.metrics_tracker.add_metric('acc', lambda **kw: (kw['y_hat'].argmax(1) == kw['y']).float().mean().item())

        # visualization
        self.visualization = Visualization()

    def _get_device(self, device):
        '''CPU or GPUs.'''
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        # print("=" * 100)
        # print(f"Runing on {device} ...")
        # print("=" * 100)
        logger.info(f"Runing on {device} ...")
        return device
    
    def _get_model(self, device, model):
        '''Move the mode to device.'''
        if device == 'auto':
            model = torch.nn.DataParallel(model).to(self.device)
            logger.info(f"Model on {self.device} with DataParallel ...")
        else:
            model = model.to(device)
            logger.info(f"Model on {device} ...")
        return model
    
    def _disable_visualization(self):
        '''Show tqdm.'''
        if self.is_tqdm:
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
            return {key:value.to(self.device) for key, value in batch.items()}
        
        elif isinstance(batch, (list, tuple)):
            # If batch is a list or tuple, assume it's a tuple of tensors
            # e.g., (input_tensor, target_tensor)
            if len(batch) < 2:
                raise ValueError("Batch must contain at least two elements: input and labels.")
            return {'inputs': batch[0].to(self.device), 'labels': batch[-1].to(self.device)}
        
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
                steps_counter = self._train_step(steps, steps_counter, pbar)
                train_logs = self.metrics_tracker.train_average_metrics(level='epoch')
                train_logs = {"train_"+name: value for name, value in train_logs.items()}

                # update show progress bar or visualization
                if steps == None:
                    # val
                    self._validate_step()
                    val_logs = self.metrics_tracker.average_metrics(stage='val', show_level='epoch')
                    val_logs = {"val_"+name: value for name, value in val_logs.items()}

                    # visualization
                    ## on pbar
                    pbar.set_postfix({**train_logs, **val_logs})
                    ## on plot
                    if self._disable_visualization():
                        self.visualization.refresh_plot(history= self.metrics_tracker.get_history())

                self._call_callbacks(method_name= "on_epoch_end", **kwargs)

        self._call_callbacks(method_name= "on_train_end", **kwargs)

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
                    # self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    if grad_norm > 1.0:
                        logger.debug(f"[GradClip] Clipped: grad norm = {grad_norm:.4f}")
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

    def predict(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader):
        '''Predict with the model.
        Example:
        >>> all_labels, all_preds = trainer.predict(model=model, dataloader=dataloader)
        '''
        model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_to_device(batch)
                labels = batch['labels']
                outputs = model(**batch)
                _, preds = torch.max(outputs, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
    
        return all_labels, all_preds

    def save_metrics(self, file_path):
        '''Save the history with pickle format.'''
        history = self.metrics_tracker.get_history()
        with open(file_path, 'wb') as f:
            pickle.dump(history, f)
        logger.debug(f"Metrics saved to {file_path}")

    def save_checkpoint(self, file_path):
        '''Save checkpoint.'''
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, file_path)
        logger.debug(f"Checkpoint saved to {file_path}")

    def load_checkpoint(self, file_path):
        '''Load checkpoint.'''
        checkpoint = torch.load(file_path)
        self.model.load_state_dict(state_dict= checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.debug(f"Checkpoint loaded from {file_path}")


# trainer.metrics_tracker._history
# trainer.metrics_tracker.get_history()
# trainer.save_metrics(file_path= "./cache/metrics_tracker_history.pickle")
# trainer.save_checkpoint(file_path= './cache/checkpoint.pt')
# trainer.load_checkpoint(file_path= './cache/checkpoint.pt')


