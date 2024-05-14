from contextlib import contextmanager
from collections import defaultdict
import math
import os
import sys
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
import time
import datetime

import torch
import torch.distributed as dist

import threading
import inspect
import copy

from torch.iidp.utils.distributed import get_allgather_value, print_one_rank
from torch.iidp.utils.json_utils import read_json
from torch.iidp.utils.cost_utils import estimate_cost

from .elastic.runtime.rpc import trainer_client as trainer_client
from .config.configurator import IIDPConfig, IIDPConfigurator, IIDPFutureConfigurator
from .cluster.cluster_manager import IIDPClusterManager

from torch.iidp.config.model.global_batch_size.gaussian_process import GaussianProcessRegressionModel
from torch.iidp.config.model.global_batch_size.exponential_smoothing import ExponentialSmoothing
from torch.iidp.config.model.global_batch_size.ensemble_method import EnsembleMethod

from torch.iidp.config.config_utils import check_user_config_is_valid
from torch.iidp.profiler.profiler_utils import get_mem_profile_data_summary

from torch.iidp.test.utils.timer import Timer

import gc


REGISTERED_WEIGHT_SYNC_METHODS = [
    'recommend',
    'overlap',
    'sequential'
]


class EpochIterator(object):
    def __init__(self):
        self.epoch_idx = -1
        self.final_epochs = -1

    @property
    def epoch(self):
        #print(f'[DEBUG][EpochIterator] property self.epoch_idx: {self.epoch_idx}')
        if self.final_epochs == -1:
            return 0
        else:
            return self.epoch_idx

    @epoch.setter
    def epoch(self, epoch):
        self.epoch_idx = epoch
        #print(f'[DEBUG][EpochIterator] self.epoch_idx: {self.epoch_idx}')

    def __iter__(self):
        return self

    def __next__(self):
        if self.final_epochs == -1:
            raise ValueError(
                f'[ERROR][torch/iidp/trainer.py] final_epochs must be > 0 in EpochIterator().__next__()')
        if self.epoch_idx < self.final_epochs-1:
            self.epoch_idx += 1
            #print(f'[DEBUG][EpochIterator] self.epoch_idx: {self.epoch_idx}')
            return self.epoch_idx
        else:
            self.epoch_idx = -1
            raise StopIteration

    def __len__(self):
        return self.final_epochs


class GlobalTrainerState(object):
    def __init__(self):
        self.partition_size = []
        self.is_accum_mode = False
        self.global_batch_size = 0


class LocalTrainerState(object):
    def __init__(self):
        self.local_batch_size = 0
        self.num_models = 0
        self.accum_step = 0


GLOBAL_TRAINER_STATE = GlobalTrainerState()
LOCAL_TRAINER_STATE = LocalTrainerState()


class TrainerHelper(object):
    def __init__(self, gpu, local_batch_size, num_models, accum_step=1, weight_sync_method='recommend'):
        self.gpu = gpu
        self.local_batch_size = local_batch_size
        self.num_models = num_models
        self.accum_step = accum_step
        self.max_accum_step = -1
        self.batch_size_per_gpu = self.local_batch_size * self.num_models
        if self.batch_size_per_gpu % local_batch_size != 0:
            raise ValueError('Local batch size must be dividied by batch size per GPU')
        if weight_sync_method not in REGISTERED_WEIGHT_SYNC_METHODS:
            raise ValueError(f'Not support unregisted weight_sync_method: {weight_sync_method}')
        self.weight_sync_method = weight_sync_method
        if os.getenv("EASYSCALE") == "1" or os.getenv("SIMIGRAD") == "1":
            if self.weight_sync_method != 'sequential':
                raise ValueError(
                    f'With EASYSCALE or SIMIGRAD weight sync method must be sequential, '
                    f'but {self.weight_sync_method}')

        self.model_streams = []

        self.original_local_models = []
        self.local_models = []
        self.local_optimizers = []
        self.local_schedulers = []
        self.criterion = None
        self.output_as_loss = False
        self.losses = {}
        self.sampler = None

        # For overlapping optimizer
        self.prepared_for_ddp = False
        self.hooks = []
        self.optimizer_stream = None
        # One DDP model's bucket_indices (All of local model's bucket indices is same)
        self.ddp_bucket_indices = []
        self.is_rebuilt_ddp_bucket_indices = False

        self._get_total_num_models()
        self.global_batch_size = self.local_batch_size * self.total_num_models

        self.all_partition_size_in_process_group = []
        self._get_all_partition_size_in_process_group()

        self.all_accum_step_in_process_group = []
        self._get_all_accum_step_in_process_group()
        self.max_accum_step = max(self.all_accum_step_in_process_group)

        self.is_accum_mode = True if self.max_accum_step > 1 else False
        # Used in seq_parallel_compute() for being block different number of VSWs on inter-node
        self.sync_accum_barrier = threading.Barrier(self.num_models)

        # It is used for _sync_params() in torch/nn/parallel/distributed.py
        self._sync_buffer_barrier = [None, None]
        if self.num_models > 1:
            self._sync_buffer_barrier = [threading.Barrier(self.num_models) for i in range(2)]

        self._set_trainer_state()

        self.local_accum_step = 0
        self.sync_step = 0
        self.epoch_iterator = EpochIterator()
        self.elapsed_time = 0
        self.total_epoch_time = 0

    def _set_trainer_state(self):
        GLOBAL_TRAINER_STATE.partition_size = self.all_partition_size_in_process_group
        GLOBAL_TRAINER_STATE.is_accum_mode = self.is_accum_mode
        GLOBAL_TRAINER_STATE.global_batch_size = self.global_batch_size

        LOCAL_TRAINER_STATE.local_batch_size = self.local_batch_size
        LOCAL_TRAINER_STATE.num_models = self.num_models
        LOCAL_TRAINER_STATE.accum_step = self.accum_step

    def _get_total_num_models(self):
        if dist.is_initialized():
            tensor = torch.tensor([self.num_models * self.accum_step], dtype=torch.int64).to(self.gpu)
            dist.all_reduce(tensor) # Default op is SUM
            self.total_num_models = tensor.item()
            tensor.cpu()
            del tensor
        else:
            self.total_num_models = self.num_models * self.accum_step

    def _get_all_partition_size_in_process_group(self):
        local_partition_size = (self.batch_size_per_gpu * self.accum_step) / self.global_batch_size
        self.all_partition_size_in_process_group = get_allgather_value(local_partition_size, self.gpu)

    def _get_all_accum_step_in_process_group(self):
        self.all_accum_step_in_process_group = get_allgather_value(self.accum_step, self.gpu)

    def set_original_local_models(self, models):
        """Set the compelete local models by users"""
        if models is None:
            raise ValueError(f"Argument is None: {models}")
        else:
            if not isinstance(models, (list, tuple)):
                raise ValueError(
                    f"Argument models must be list or tuple type: {type(models)}")
        self.original_local_models = models
        assert len(self.original_local_models) == self.num_models

    def set_local_optimizers(self, optimizers):
        """Set the compelete local optimizers by users"""
        if optimizers is None:
            raise ValueError(f"Argument is None: {optimizers}")
        else:
            if not isinstance(optimizers, (list, tuple)):
                raise ValueError(
                    f"Argument optimizers must be list or tuple type: {type(optimizers)}")
        self.local_optimizers = optimizers
        assert len(self.local_optimizers) == self.num_models

    def set_local_schedulers(self, schedulers=None):
        """Set the compelete local schedulers by users"""
        # LR scheduler is optional
        if schedulers is not None:
            if not isinstance(schedulers, (list, tuple)):
                raise ValueError(
                    f"Argument optimizers must be list or tuple type: {type(schedulers)}")
            self.local_schedulers = schedulers
            assert len(self.local_schedulers) == self.num_models

    def _set_original_local_models(self, model):
        if model is None:
            raise ValueError(f"Argument is None: {model}")
        is_set_by_user = (len(self.original_local_models) == self.num_models)
        if not is_set_by_user:
            self.original_local_models = [model]
            for _ in range(1, self.num_models):
                copied_model = copy.deepcopy(model)
                self.original_local_models.append(copied_model)

    def _set_criterion(self, criterion):
        if criterion is None:
            raise ValueError(f"Argument is None: {criterion}")
        self.criterion = criterion
        if hasattr(self.criterion, 'forward'):
            args_of_criterion = inspect.getfullargspec(getattr(self.criterion, 'forward')).args
        else:
            args_of_criterion = inspect.getfullargspec(self.criterion).args
        if 'self' in args_of_criterion:
            args_of_criterion.remove('self')
        num_args = len(args_of_criterion)
        if num_args == 1:
            self.output_as_loss = True
        elif num_args == 2: # We expect arguments as output (y) and target (y^)
            self.output_as_loss = False
        else:
            raise ValueError(
                f"Not support number of arguments in criterion function > 2: {num_args}")

        self.trainer_print(f"Criterion has {num_args} argument(s): {','.join(args_of_criterion)} "
              f"=> self.output_as_loss: {self.output_as_loss}")

    def _get_required_args_value(self, instance):
        """Helper function for _set_local_optimizers() and _set_local_schedulers()"""
        removable_args = ['self', 'optimizer', 'params', 'lr']
        args_inspect = inspect.getfullargspec(instance.__init__)
        args_of_instace = args_inspect.args
        filtered_args_of_instance = [x for x in args_of_instace if x not in removable_args]
        is_defaults_exists = (args_inspect.defaults is not None and len(args_inspect.defaults) > 1)
        if is_defaults_exists:
            required_args = filtered_args_of_instance[:-len(args_inspect.defaults)]
        else:
            required_args = filtered_args_of_instance
        args = []
        for arg_name in required_args:
            try:
                # NOTE: In torch/optim/lr_scheduler.py, ```LambdaLR``` class has self.lr_lambdas,
                # but argument is lr_lambda
                if arg_name == 'lr_lambda':
                    args.append(instance.__dict__['lr_lambdas'][0])
                else:
                    args.append(instance.__dict__[arg_name])
            except KeyError:
                raise KeyError(f'[ERROR] instance.__dict__: {instance.__dict__} \n'
                               f'This might happen if argument is not registered by '
                               f'member variable of instance.')
        return args

    def _set_local_optimizers(self, optimizer, param_groups_func=None):
        """
        NOTE: Even main optimizer only updates globally aggregated gradients,
        optimizer.zero_grad() is efficient for parallel_compute().
        That's why we keep the individual optimizer for each local model.
        """
        if not issubclass(type(optimizer), torch.optim.Optimizer):
            raise TypeError(
                f'To set local optimizers for copy (use _set_local_optimizers()), original optimizer type: '
                f'{type(optimizer)} '
                f'must be sub-class of torch.optim.Optimizer')
        if optimizer is None:
            raise TypeError(f"Argument optimizer must be configured, but {optimizer}")

        self.param_groups_func = param_groups_func
        is_set_by_user = (len(self.local_optimizers) == self.num_models)
        if not is_set_by_user:
            self.local_optimizers = [optimizer]
            for idx in range(1, self.num_models):
                if self.param_groups_func:
                    params = self.param_groups_func(self.original_local_models[idx])
                else:
                    params = self.original_local_models[idx].parameters()
                args = self._get_required_args_value(optimizer)
                # https://stackoverflow.com/questions/21060073/dynamic-inheritance-in-python
                cls = type(optimizer.__class__.__name__, (optimizer.__class__,), dict(optimizer.__dict__))
                copied_optimizer = cls(params, lr=optimizer.defaults['lr'], *args)
                copied_optimizer.load_state_dict(optimizer.state_dict())
                self.local_optimizers.append(copied_optimizer)
        for optimizer in self.local_optimizers:
            optimizer.zero_grad()

    def _set_local_schedulers(self, scheduler=None):
        # LR scheduler is optional
        if scheduler is not None:
            is_set_by_user = (len(self.local_schedulers) == self.num_models)
            if not is_set_by_user:
                self.local_schedulers = [scheduler]

    @property
    def main_stream(self):
        return self.model_streams[0]

    @property
    def eval_model(self):
        if type(self.local_models[0]) == torch.nn.parallel.DistributedDataParallel:
            return self.local_models[0].module
        else:
            return self.local_models[0]

    @property
    def main_model(self):
        return self.local_models[0]

    @property
    def main_optimizer(self):
        return self.local_optimizers[0]

    @property
    def main_scheduler(self):
        return self.local_schedulers[0] if self.local_schedulers is not None else None

    @property
    def num_local_models(self):
        return self.num_models

    @property
    def epoch(self):
        #self.trainer_print(f'trainer.epoch property: {self.epoch_iterator.epoch}')
        return self.epoch_iterator.epoch

    @epoch.setter
    def epoch(self, epoch):
        self.epoch_iterator.epoch = epoch

    def remaining_epochs(self, final_epochs):
        #self.trainer_print(f'Number of epochs to train: {final_epochs}')
        self.epoch_iterator.final_epochs = final_epochs
        try:
            for epoch in self.epoch_iterator.__iter__():
                yield epoch
        finally:
            self.print_final_results()

    def print_final_results(self):
        self.trainer_print(f'Total epoch time (sec): {self.total_epoch_time}')
        self.trainer_print(f'Total epoch time: {datetime.timedelta(seconds=self.total_epoch_time)}')

    def set_model_train(self):
        for local_model in self.local_models:
            local_model.train()

    def _create_model_streams(self):
        for _ in range(self.num_models):
            self.model_streams.append(torch.cuda.Stream())

    def _create_stream_for_optimizer(self):
        self.optimizer_stream = torch.cuda.Stream()

    def _prepare_hooks_for_local_models(self, hook):
        def dummy_hook(state, bucket):
            fut = torch.futures.Future()
            fut.set_result(bucket.get_tensors())
            return fut

        for model_idx in range(self.num_models):
            if model_idx == 0:
                self.hooks.append(hook)
            else:
                self.hooks.append(dummy_hook)

    def _check_overlap_with_ddp(self):
        if not self.prepared_for_ddp:
            raise ValueError(
                "DDP instance must be prepared with self.weight_sync_method = overlap")

        for param_indices in self.ddp_bucket_indices:
            if param_indices != sorted(param_indices):
                raise RuntimeError(
                    "Parameter indices in each bucket must be sorted with self.weight_sync_method = overlap")

        if self.hooks is None:
            raise ValueError(
                "hooks must be prepared with self.weight_sync_method = overlap")
        elif not isinstance(self.hooks, (list, tuple)):
            raise ValueError(
                f"Argument hooks must be list or tuple type: {type(self.hooks)}")
        elif len(self.hooks) != self.num_models:
            raise ValueError(f"Number of hooks: {len(self.hooks)} "
                             f"must be equal to "
                             f"number of local models : {self.num_models}")
        for hook in self.hooks:
            if not callable(hook):
                raise TypeError("hook must be callable.")

        if self.optimizer_stream is None:
            raise ValueError(
                "optimizer_stream must be assigned with self.weight_sync_method = overlap")

    def trainer_print(self, message, status='info'):
        print_msg = f'[{status.upper()}][{self.__class__.__name__}] {message}'
        if dist.is_initialized():
            if dist.get_rank() == 0:
                print(print_msg)
        else:
            print(print_msg)

    @contextmanager
    def measure_epoch_time(self):
        try:
            start_time = time.time()
            yield
        finally:
            self.elapsed_time = int(time.time() - start_time)
            self.trainer_print(f'Epoch time: {self.elapsed_time}')
            self.total_epoch_time += self.elapsed_time

    @contextmanager
    def record_epoch_data(self):
        try:
            yield
        finally:
            self.trainer_print(f'record at epoch: {self.epoch} | iterations: {self.sync_step} | loss: {self.losses[0]:.3f}')


class IIDPTrainer(TrainerHelper):
    def __init__(self, gpu, local_batch_size, num_models, accum_step, weight_sync_method='recommend'):
        super().__init__(gpu, local_batch_size, num_models, accum_step, weight_sync_method)

    def prepare_stream_parallel(self, model, criterion, **kwargs):
        gradient_as_bucket_view = kwargs.get('gradient_as_bucket_view') or False
        find_unused_parameters = kwargs.get('find_unused_parameters') or False
        self._create_model_streams()
        self._set_original_local_models(model)
        self._set_criterion(criterion)
        for idx, original_model in enumerate(self.original_local_models):
            # Assign buckets for DDP to model's stream to synchronize copy in all-reduce
            with torch.cuda.stream(self.model_streams[idx]):
                local_ddp_module = torch.nn.parallel.DistributedDataParallel(
                                    original_model, device_ids=[self.gpu], output_device=[self.gpu],
                                    find_unused_parameters=find_unused_parameters,
                                    gradient_as_bucket_view=gradient_as_bucket_view,
                                    model_index=idx, num_local_models=self.num_models,
                                    total_num_models=self.total_num_models,
                                    sync_buffer_barrier=self._sync_buffer_barrier)
                self.local_models.append(local_ddp_module)
            # It is used for overlapping optimizer with torch.nn.parallel.DistributedDataParallel
            self.ddp_bucket_indices = self.main_model.bucket_indices
            self.prepared_for_ddp = True

            if self.weight_sync_method == 'recommend':
                self._recommend_weight_update()
            else: # Show bucket distribution even weight sync method is not 'recommend' option
                self._calculate_bucket_distribution()

    def prepare_weight_sync_method(self, optimizer, scheduler=None, param_groups_func=None):
        self._set_local_optimizers(optimizer, param_groups_func)
        self._set_local_schedulers(scheduler)
        if self.weight_sync_method == 'overlap':
            if self.prepared_for_ddp:
                self._prepare_overlap_optimizer_with_ddp()
            else:
                raise RuntimeError("[ERROR] Without DDP, overlap optimizer cannot work")

    def _recommend_weight_update(self):
        """
        If the distribution of bucket size is not uniform,
        overlapping optimizer is not recommended
        due to interference with all-reduce NCCL kernel
        """
        def get_avg(val):
            if isinstance(val, (list, tuple)):
                return int(sum(val) / len(val))
        bukcet_capacity = self.main_model.bucket_bytes_cap / (1024 * 1024) # MB
        bucket_size_distribution = self._calculate_bucket_distribution()
        # Last bucket doesn't overlap with all-reduce kernel
        allreduce_overlap_bucket_distribution = bucket_size_distribution[:-1]
        #self.trainer_print(f'bucket : {allreduce_overlap_bucket_distribution}', 'debug')
        """ Buckets to the size specified by the user is larger than a threshold
        potential_interference_bucket = [
            bucket for bucket in allreduce_overlap_bucket_distribution \
                 if bucket > bukcet_capacity
        ]
        """
        potential_interference_bucket = allreduce_overlap_bucket_distribution
        #self.trainer_print(f'potential_interference_bucket: {potential_interference_bucket}', 'debug')
        avg_outlier_bucket_size = get_avg(potential_interference_bucket)
        #self.trainer_print(f'avg_outlier_bucket_size: {avg_outlier_bucket_size}', 'debug')
        norm_avg_outlier_bucket_size = avg_outlier_bucket_size / bukcet_capacity
        if norm_avg_outlier_bucket_size > 1.5: # Threshold is heuristic
            self.weight_sync_method = 'sequential'
        else:
            self.weight_sync_method = 'overlap'
        self.trainer_print(f'Recommend [{self.weight_sync_method}] as weight sync method '
              f'as uniformity of bucket size is {norm_avg_outlier_bucket_size}')

    def _calculate_bucket_distribution(self):
        bucket_size_distribution = []
        parameter_size_distribution = []
        for _, param in enumerate(self.main_model.ddp_register_params):
            if hasattr(param, 'index'):
                param_mem_value = round(param.nelement() * param.element_size() / (1024 ** 2), 2)
                parameter_size_distribution.append(param_mem_value)

        for bucket in self.ddp_bucket_indices:
            bucket_size = 0
            for param_index in bucket:
                param_size = parameter_size_distribution[param_index]
                bucket_size += param_size
            bucket_size_distribution.append(round(bucket_size, 2))
        self.trainer_print(f'bucket_size_distribution (backward order): {bucket_size_distribution}', 'debug')
        return bucket_size_distribution

    @contextmanager
    def accum_processing(self):
        if dist.is_initialized() and self.local_accum_step == 0:
            self.prev_require_forward_param_sync = self.main_model.require_forward_param_sync
            def _forward_model(model, stream):
                with torch.cuda.stream(stream):
                    if model.require_forward_param_sync:
                        model._sync_params()
                        model.require_forward_param_sync = False

            threads = []
            for idx in range(self.num_models):
                threads.append(threading.Thread(target=_forward_model,
                                        args=(self.local_models[idx], self.model_streams[idx],))
                            )
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        try:
            yield
        finally:
            self.local_accum_step += 1
            if dist.is_initialized() and self.local_accum_step == self.accum_step:
                for model in self.local_models:
                    model.require_forward_param_sync = self.prev_require_forward_param_sync

    def _compute_forward_and_loss(self, model, criterion, input, target):
        #print(f'[DEBUG] _compute_forward_and_loss() - input.size(): {input.size()} | target.size(): {target.size()}')
        if self.output_as_loss:
            model_to_inspect = model.module if type(model) == torch.nn.parallel.DistributedDataParallel else model
            args_of_model = inspect.getfullargspec(getattr(model_to_inspect, 'forward')).args
            if 'self' in args_of_model:
                args_of_model.remove('self')
            num_args = len(args_of_model)
            if num_args > 2:
                loss = criterion(model(*input, target))
            else:
                loss = criterion(model(input, target))
        else:
            if isinstance(input, (tuple, list)):
                output = model(*input)
            else:
                output = model(input)
            loss = criterion(output, target)
        return loss

    # Use to computation profiler
    def parallel_forward(self, scatter_input, scatter_target):
        if scatter_input is None or scatter_target is None:
            raise RuntimeError("scatter_input and scatter_target must be configured "
                               "to arguments of parallel_compute()")
        elif len(scatter_input) != self.num_models or len(scatter_target) != self.num_models:
            raise RuntimeError(f"Length of scatter_input: {len(scatter_input)} "
                               f"and scatter_target: {len(scatter_target)} "
                               f"must be equal to "
                               f"number of local models : {self.num_models}")

        lock = threading.Lock()
        def _local_worker(index, model, stream, input, target, criterion):
            with torch.cuda.stream(stream):
                loss = self._compute_forward_and_loss(model, criterion, input, target)
                with lock:
                    self.losses[index] = loss
                    #self.trainer_print(f'loss at index: {index}: {self.losses[index]}', 'debug')

        if self.num_models > 1:
            threads = []
            for idx in range(self.num_models):
                threads.append(threading.Thread(target=_local_worker,
                                        args=(idx, self.local_models[idx], self.model_streams[idx],
                                            scatter_input[idx], scatter_target[idx], self.criterion))
                            )
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            idx = 0
            _local_worker(idx, self.local_models[idx], self.model_streams[idx],
                          scatter_input[idx], scatter_target[idx], self.criterion)

    # Use to computation profiler
    def parallel_backward(self, scatter_input, scatter_target):
        if scatter_input is None or scatter_target is None:
            raise RuntimeError("scatter_input and scatter_target must be configured "
                               "to arguments of parallel_compute()")
        elif len(scatter_input) != self.num_models or len(scatter_target) != self.num_models:
            raise RuntimeError(f"Length of scatter_input: {len(scatter_input)} "
                               f"and scatter_target: {len(scatter_target)} "
                               f"must be equal to "
                               f"number of local models : {self.num_models}")

        lock = threading.Lock()
        def _local_worker(index, model, stream, input, target, criterion, optimizer):
            with torch.cuda.stream(stream):
                loss = self.losses[index]
                if not self.is_accum_mode:
                    optimizer.zero_grad()
                loss.backward(model_index=index)

        if self.num_models > 1:
            threads = []
            for idx in range(self.num_models):
                threads.append(threading.Thread(target=_local_worker,
                                        args=(idx, self.local_models[idx], self.model_streams[idx],
                                            scatter_input[idx], scatter_target[idx],
                                            self.criterion, self.local_optimizers[idx]))
                            )
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            idx = 0
            _local_worker(idx, self.local_models[idx], self.model_streams[idx],
                          scatter_input[idx], scatter_target[idx],
                          self.criterion, self.local_optimizers[idx])

    def parallel_compute(self, scatter_input, scatter_target, accum_step=-1):
        if scatter_input is None or scatter_target is None:
            raise RuntimeError("scatter_input and scatter_target must be configured "
                               "to arguments of parallel_compute()")
        elif len(scatter_input) != self.num_models or len(scatter_target) != self.num_models:
            raise RuntimeError(f"Length of scatter_input: {len(scatter_input)} "
                               f"and scatter_target: {len(scatter_target)} "
                               f"must be equal to "
                               f"number of local models : {self.num_models}")
        if self.is_accum_mode and self.prepared_for_ddp:
            self.seq_parallel_compute(scatter_input, scatter_target, accum_step)
            return

        lock = threading.Lock()
        def _local_worker(index, model, stream, input, target, criterion, optimizer):
            with torch.cuda.stream(stream):
                #print(f'[DEBUG] rank: {dist.get_rank()} | step: {self.sync_step} | _local_worker() - idx: {index}')
                loss = self._compute_forward_and_loss(model, criterion, input, target)
                #print(f'[DEBUG] rank: {dist.get_rank()} | step: {self.sync_step} | after _compute_forward_and_loss - idx: {index}')
                with lock:
                    self.losses[index] = loss
                    #print(f'[DEBUG] rank: {dist.get_rank()} | step: {self.sync_step} | loss at index: {index}: {self.losses[index]}')
                if not self.is_accum_mode:
                    optimizer.zero_grad()
                #print(f'[DEBUG] rank: {dist.get_rank()} | step: {self.sync_step} | before loss.backward() - idx: {index}')
                loss.backward(model_index=index)
                #print(f'[DEBUG] rank: {dist.get_rank()} | step: {self.sync_step} | @@@@@@@@@@@@@@@@@@@@@@@@@ after loss.backward() - idx: {index}')

        #print(f'[DEBUG] rank: {dist.get_rank()} | parallel_compute() - self.num_models: {self.num_models} | '
        #      f'len(self.local_models): {len(self.local_models)} | '
        #      f'len(self.model_streams): {len(self.model_streams)}')
        if self.num_models > 1:
            threads = []
            for idx in range(self.num_models):
                threads.append(threading.Thread(target=_local_worker,
                                        args=(idx, self.local_models[idx], self.model_streams[idx],
                                            scatter_input[idx], scatter_target[idx],
                                            self.criterion, self.local_optimizers[idx]))
                            )
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            idx = 0
            _local_worker(idx, self.local_models[idx], self.model_streams[idx],
                          scatter_input[idx], scatter_target[idx],
                          self.criterion, self.local_optimizers[idx])

    def seq_parallel_compute(self, scatter_input, scatter_target, accum_step=-1):
        if scatter_input is None or scatter_target is None:
            raise RuntimeError("scatter_input and scatter_target must be configured "
                               "to arguments of parallel_compute()")
        elif len(scatter_input) != self.num_models or len(scatter_target) != self.num_models:
            raise RuntimeError(f"Length of scatter_input: {len(scatter_input)} "
                               f"and scatter_target: {len(scatter_target)} "
                               f"must be equal to "
                               f"number of local models : {self.num_models}")
        if self.max_accum_step <= 1 or accum_step < 0:
            raise RuntimeError('If self.max_accum_step <= 1 or accum_step < 0, '
                               'seq_parallel_compute() must not be called')

        lock = threading.Lock()
        def _local_accum_worker(index, model, stream, input, target, criterion):
            #print(f'[_local_accum_worker] rank: {dist.get_rank()} | step: {self.sync_step} | model index: {index}')
            with torch.cuda.stream(stream):
                with model.no_sync():
                    loss = self._compute_forward_and_loss(model, criterion, input, target)
                    #print(f'[DEBUG] rank: {dist.get_rank()} | step: {self.sync_step} | after _compute_forward_and_loss - idx: {index}')
                    with lock:
                        self.losses[index] = loss
                        #print(f'[DEBUG] rank: {dist.get_rank()} | step: {self.sync_step} | loss at index: {index}: {self.losses[index]}')
                    loss.backward(model_index=index)
                    #print(f'[_local_accum_worker] rank: {dist.get_rank()} | step: {self.sync_step} | model index: {index} | after backward()')

        def _local_sync_worker(index, model, stream, input, target, criterion):
            #print(f'[_local_sync_worker] rank: {dist.get_rank()} | step: {self.sync_step} | model index: {index}')
            with torch.cuda.stream(stream):
                loss = self._compute_forward_and_loss(model, criterion, input, target)
                #print(f'[DEBUG] rank: {dist.get_rank()} | step: {self.sync_step} | after _compute_forward_and_loss - idx: {index}')
                with lock:
                    self.losses[index] = loss
                    #print(f'[DEBUG] rank: {dist.get_rank()} | step: {self.sync_step} | loss at index: {index}: {self.losses[index]}')
                self.sync_accum_barrier.wait()
                loss.backward(model_index=index)
                #print(f'[_local_sync_worker] rank: {dist.get_rank()} | step: {self.sync_step} | model index: {index} | after backward()')

        if self.num_models > 1:
            if accum_step < self.accum_step - 1:
                #print(f'[TEST] _assert_equal_params() | accum_step: {accum_step} | self.accum_step: {self.accum_step}')
                #self._assert_equal_params()
                threads = []
                for idx in range(self.num_models):
                    threads.append(threading.Thread(target=_local_accum_worker,
                                            args=(idx, self.local_models[idx], self.model_streams[idx],
                                                scatter_input[idx], scatter_target[idx],
                                                self.criterion))
                                )
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
            else:
                #print(f'[TEST] _assert_equal_params() | accum_step: {accum_step} | self.accum_step: {self.accum_step}')
                #self._assert_equal_params()
                threads = []
                for idx in range(self.num_models):
                    threads.append(threading.Thread(target=_local_sync_worker,
                                            args=(idx, self.local_models[idx], self.model_streams[idx],
                                                scatter_input[idx], scatter_target[idx],
                                                self.criterion))
                                )
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
        else:
            idx = 0
            if accum_step < self.accum_step - 1:
                _local_accum_worker(idx, self.local_models[idx], self.model_streams[idx],
                                    scatter_input[idx], scatter_target[idx],
                                    self.criterion)
            else:
                _local_sync_worker(idx, self.local_models[idx], self.model_streams[idx],
                                    scatter_input[idx], scatter_target[idx],
                                    self.criterion)

    def compute(self, data):
        if self.is_accum_mode and self.prepared_for_ddp:
            """
            parallel_accum_inputs = [[data[i+self.num_models*j][0] for i in range(self.num_models)] for j in range(self.accum_step)]
            parallel_accum_targets = [[data[i+self.num_models*j][1] for i in range(self.num_models)] for j in range(self.accum_step)]
            with self.accum_processing():
                for seq_step, (parallel_input, parallel_target) in \
                        enumerate(zip(parallel_accum_inputs, parallel_accum_targets)):
                    #print(f'[DEBUG] *****> seq step: {seq_step}')
                    self.parallel_compute(parallel_input, parallel_target, seq_step)
            """
            parallel_input = [data[i][0] for i in range(self.num_models)]
            parallel_target = [data[i][1] for i in range(self.num_models)]
            with self.accum_processing():
                self.parallel_compute(parallel_input, parallel_target, self.local_accum_step)
        else:
            parallel_input = [data[i][0] for i in range(self.num_models)]
            parallel_target = [data[i][1] for i in range(self.num_models)]
            self.parallel_compute(parallel_input, parallel_target)

    def profile_parallel_forward(self, scatter_input, scatter_target):
        if scatter_input is None or scatter_target is None:
            raise RuntimeError("scatter_input and scatter_target must be configured "
                               "to arguments of parallel_compute()")
        elif len(scatter_input) != self.num_models or len(scatter_target) != self.num_models:
            raise RuntimeError(f"Length of scatter_input: {len(scatter_input)} "
                               f"and scatter_target: {len(scatter_target)} "
                               f"must be equal to "
                               f"number of local models : {self.num_models}")

        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)

        lock = threading.Lock()
        def _local_worker(index, model, stream, input, target, criterion):
            with torch.cuda.stream(stream):
                if index == 0:
                    fwd_start.record()
                loss = self._compute_forward_and_loss(model, criterion, input, target)
                with lock:
                    self.losses[index] = loss

        if self.num_models > 1:
            threads = []
            for idx in range(self.num_models):
                threads.append(threading.Thread(target=_local_worker,
                                        args=(idx, self.local_models[idx], self.model_streams[idx],
                                            scatter_input[idx], scatter_target[idx], self.criterion))
                            )
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            idx = 0
            _local_worker(idx, self.local_models[idx], self.model_streams[idx],
                          scatter_input[idx], scatter_target[idx], self.criterion)
        if idx == self.num_models-1:
            fwd_end.record()
        torch.cuda.synchronize()
        return fwd_start.elapsed_time(fwd_end)

    def profile_parallel_backward(self, scatter_input, scatter_target):
        if scatter_input is None or scatter_target is None:
            raise RuntimeError("scatter_input and scatter_target must be configured "
                               "to arguments of parallel_compute()")
        elif len(scatter_input) != self.num_models or len(scatter_target) != self.num_models:
            raise RuntimeError(f"Length of scatter_input: {len(scatter_input)} "
                               f"and scatter_target: {len(scatter_target)} "
                               f"must be equal to "
                               f"number of local models : {self.num_models}")

        bwd_start = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)

        lock = threading.Lock()
        def _local_worker(index, model, stream, input, target, criterion, optimizer):
            with torch.cuda.stream(stream):
                if index == 0:
                    bwd_start.record()
                loss = self.losses[index]
                if not self.is_accum_mode:
                    optimizer.zero_grad()
                loss.backward(model_index=index)

        if self.num_models > 1:
            threads = []
            for idx in range(self.num_models):
                threads.append(threading.Thread(target=_local_worker,
                                        args=(idx, self.local_models[idx], self.model_streams[idx],
                                            scatter_input[idx], scatter_target[idx],
                                            self.criterion, self.local_optimizers[idx]))
                            )
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            idx = 0
            _local_worker(idx, self.local_models[idx], self.model_streams[idx],
                          scatter_input[idx], scatter_target[idx],
                          self.criterion, self.local_optimizers[idx])
        if idx == self.num_models-1:
            bwd_end.record()
        torch.cuda.synchronize()
        return bwd_start.elapsed_time(bwd_end)

    def profile_parallel_compute(self, scatter_input, scatter_target):
        if scatter_input is None or scatter_target is None:
            raise RuntimeError("scatter_input and scatter_target must be configured "
                               "to arguments of parallel_compute()")
        elif len(scatter_input) != self.num_models or len(scatter_target) != self.num_models:
            raise RuntimeError(f"Length of scatter_input: {len(scatter_input)} "
                               f"and scatter_target: {len(scatter_target)} "
                               f"must be equal to "
                               f"number of local models : {self.num_models}")

        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        bwd_start = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)

        # reference: https://stackoverflow.com/questions/2829329/catch-a-threads-exception-in-the-caller-thread
        class ParallelComputeThread(threading.Thread):
            def run(self):
                self._exc = None
                try:
                    super().run()
                except Exception as e:
                    self._exc = e

            def join(self, timeout=None):
                super().join(timeout=timeout)
                if self._exc:
                    raise self._exc

        lock = threading.Lock()
        def _local_worker(index, model, stream, input, target, criterion, optimizer):
            if index == 0:
                fwd_start.record()
            with torch.cuda.stream(stream):
                loss = self._compute_forward_and_loss(model, criterion, input, target)
                with lock:
                    self.losses[index] = loss
                    #print(f'[DEBUG] loss at index: {index}: {self.losses[index]}')
                if not self.is_accum_mode:
                    optimizer.zero_grad()
                if index == self.num_models-1:
                    fwd_end.record()
                if index == 0:
                    bwd_start.record()
                loss.backward(model_index=index)

        if self.num_models > 1:
            threads = []
            for idx in range(self.num_models):
                threads.append(ParallelComputeThread(target=_local_worker,
                                        args=(idx, self.local_models[idx], self.model_streams[idx],
                                            scatter_input[idx], scatter_target[idx],
                                            self.criterion, self.local_optimizers[idx]))
                            )
            for thread in threads:
                thread.start()
            for thread in threads:
                try:
                    thread.join()
                except RuntimeError as e:
                    raise RuntimeError(e)
        else:
            idx = 0
            _local_worker(idx, self.local_models[idx], self.model_streams[idx],
                          scatter_input[idx], scatter_target[idx],
                          self.criterion, self.local_optimizers[idx])
        if idx == self.num_models-1:
            bwd_end.record()
        torch.cuda.synchronize()
        return fwd_start.elapsed_time(fwd_end), bwd_start.elapsed_time(bwd_end)

    def register_comm_hook(self, hook):
        dummy_hook = torch.iidp.ddp_comm_hooks.dummy_hook

        for model_idx in range(self.num_models):
            if model_idx == 0:
                self.hooks.append(hook)
            else:
                self.hooks.append(dummy_hook)
        for local_model, hook in zip(self.local_models, self.hooks):
            local_model.register_comm_hook(state=None, hook=hook)

    def _create_optimizer_hook(self, hook):
        def hook_rebuild_bucket_setup():
            if self.main_model._has_rebuilt_buckets and not self.is_rebuilt_ddp_bucket_indices:
                self.ddp_bucket_indices = self.main_model.bucket_indices
                self._calculate_bucket_distribution()
                #self.trainer_print(f'hook_rebuild_bucket_setup() - rebult ddp_bucket_indices: {self.ddp_bucket_indices}', 'debug')
                self.is_rebuilt_ddp_bucket_indices = True

        def hook_with_optimizer_step(state, bucket):
            future_work = hook(state, bucket)
            hook_rebuild_bucket_setup()
            def optimizer_step(fut: torch.futures.Future):
                bucket_index = bucket.get_index()
                param_indices = self.ddp_bucket_indices[bucket_index]
                nccl_stream = torch.cuda.current_stream()
                self.optimizer_stream.wait_stream(nccl_stream)
                with torch.cuda.stream(self.optimizer_stream):
                    gradients = bucket.get_gradients()
                    #self.trainer_print(f'gradients: {gradients}', 'debug')
                    for index, grad in zip(param_indices, gradients):
                        grad.index = index
                    self._optimizer_step(gradients, param_indices)
                return bucket.get_tensors()

            return future_work.then(optimizer_step)
        return hook_with_optimizer_step

    def _prepare_overlap_optimizer_with_ddp(self):
        hook = torch.iidp.ddp_comm_hooks.iidp_allreduce_hook
        self._create_stream_for_optimizer()
        self._prepare_hooks_for_local_models(hook)
        self._check_overlap_with_ddp()
        for i, (local_model, hook) in enumerate(zip(self.local_models, self.hooks)):
            if i == 0:
                state = torch.iidp.ddp_comm_hooks.IIDPState(None, self.total_num_models)
                hook = self._create_optimizer_hook(hook)
            else:
                state = None
            local_model.register_comm_hook(state=state, hook=hook)

    def _optimizer_step(self, gradients, param_indices):
        if not self.weight_sync_method == 'overlap':
            raise RuntimeError("This function must be called if weight_sync_method is overlap")

        self.main_optimizer.step(gradients)
        # Partial weight copy
        partial_src_params_to_copy = [self.main_model.ddp_register_params[i] for i in param_indices]
        for idx in range(1, self.num_models):
            partial_dst_params_to_copy = [self.local_models[idx].ddp_register_params[i] for i in param_indices]
            for src_param, dst_param in \
                    zip(partial_src_params_to_copy, partial_dst_params_to_copy):
                dst_param.data.copy_(src_param.data)

    def is_sync_step(self):
        if self.is_accum_mode and self.local_accum_step < self.accum_step:
            return False
        else:
            return True

    def step(self):
        if self.is_accum_mode and self.local_accum_step < self.accum_step:
            # NOTE: Synchronize multi-stream before next computation
            # to avoid RuntimeError: CUDA error: device-side assert triggered
            torch.cuda.synchronize()
            return False
        if self.weight_sync_method == 'overlap':
            if self.is_accum_mode:
                with torch.cuda.stream(self.optimizer_stream):
                    self.main_optimizer.zero_grad()
                for idx in range(1, self.num_models):
                    stream = self.model_streams[idx]
                    optimizer = self.local_optimizers[idx]
                    with torch.cuda.stream(stream):
                        optimizer.zero_grad()
            torch.cuda.synchronize()

        elif self.weight_sync_method == 'sequential':
            with torch.cuda.stream(self.main_stream):
                self.main_optimizer.step()
                if self.is_accum_mode:
                    self.main_optimizer.zero_grad()
            torch.cuda.synchronize()
            for idx in range(1, self.num_models):
                stream = self.model_streams[idx]
                optimizer = self.local_optimizers[idx]
                with torch.cuda.stream(stream):
                    for src_param, dst_param in \
                            zip(self.main_model.parameters(), self.local_models[idx].parameters()):
                        dst_param.data.copy_(src_param.data)
                    if self.is_accum_mode:
                        optimizer.zero_grad()
            torch.cuda.synchronize()
        else:
            raise RuntimeError(f'Not support weight_sync_method: {self.weight_sync_method}')
        self.sync_step += 1
        self.local_accum_step = 0
        return True

    def profile_step(self):
        update_start = torch.cuda.Event(enable_timing=True)
        update_end = torch.cuda.Event(enable_timing=True)
        copy_start = torch.cuda.Event(enable_timing=True)
        copy_end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        with torch.cuda.stream(self.main_stream):
            update_start.record()
            self.main_optimizer.step()
            update_end.record()
        torch.cuda.synchronize()
        if self.num_models == 1:
            copy_start.record()
        for idx in range(1, self.num_models):
            stream = self.model_streams[idx]
            with torch.cuda.stream(stream):
                if idx == 1:
                    copy_start.record()
                for src_param, dst_param in \
                        zip(self.main_model.parameters(), self.local_models[idx].parameters()):
                    dst_param.data.copy_(src_param.data)
        if self.num_models == 1 or idx == self.num_models-1:
            copy_end.record()
        torch.cuda.synchronize()
        return update_start.elapsed_time(update_end), copy_start.elapsed_time(copy_end)

    def scheduler_step(self):
        if self.local_schedulers:
            if self.weight_sync_method == 'overlap':
                with torch.cuda.stream(self.optimizer_stream):
                    self.main_scheduler.step()
            elif self.weight_sync_method == 'sequential':
                with torch.cuda.stream(self.main_stream):
                    self.main_scheduler.step()
            else:
                raise RuntimeError(f'Not support weight_sync_method: {self.weight_sync_method}')


class ElasticTrainTimer(object):
    def __init__(self, start_time):
        self.start_time = start_time
        self.elapsed_time = 0

    def update(self, measured_time):
        self.elapsed_time = measured_time - self.start_time


class AdaptiveIIDPTrainer(IIDPTrainer):
    def __init__(self, gpu, local_batch_size, num_models, accum_step,
                 weight_sync_method='recommend', adaptive_batch_params=None,
                 checkpoint_dir=None, elastic_train_timer=None):
        super().__init__(gpu, local_batch_size, num_models, accum_step, weight_sync_method)
        self.data_loader = None
        self.is_elastic_training = False
        self.is_resource_reallocated = False
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_path = None
        if self._checkpoint_dir is not None:
            self.is_elastic_training = True
            self._trainer_id = int(os.environ['IIDP_JOB_ID'])
            self._local_rank = int(os.environ['IIDP_LOCAL_RANK'])
            self._worker_id = int(os.environ['IIDP_WORKER_ID'])
            self._sched_addr = os.environ['IIDP_SCHED_ADDR']
            self._sched_port = int(os.environ['IIDP_SCHED_PORT'])
            self._rpc_client = trainer_client.TrainerRpcClient(
                    self._trainer_id, self._worker_id, self._sched_addr, self._sched_port)
            if not os.path.isdir(self._checkpoint_dir):
                raise ValueError(f'self._checkpoint_dir must be directory: {self._checkpoint_dir}')
            self._checkpoint_path = os.path.join(self._checkpoint_dir, 'checkpoint.pth')

            self.is_resource_reallocated = (os.path.exists(self._checkpoint_path) is True)
            self.trainer_print(f'self.is_resource_reallocated: {self.is_resource_reallocated}', 'debug')
            # NOTE: Include the below elapsed time
            # 1) re-initialize and train components setup (model, optimizer, data loader, etc)
            # 2) save & load checkpoint overhead
            self.reallocation_overhead = -1
            self.elastic_train_timer = elastic_train_timer

        self.adaptive_batch_params = adaptive_batch_params
        self._check_user_config_is_valid()
        self._init_adaptive_batch_params()
        # For IIDP dynamic configuration
        self.prev_max_num_models = self.num_models
        # NOTE: Even if elastic training is not configured, this is used for the system analysis
        self.global_batch_size_trajectory = []
        if self.is_elastic_training is True:
            if os.getenv('NO_BATCH_SIZE_PREDICTION_EXP') == "1":
                self.trainer_print('**********************************************', 'experimental')
                self.trainer_print('Elastic training without batch size prediction', 'experimental')
                self.trainer_print('**********************************************', 'experimental')
            else:
                self._prepare_global_batch_size_prediction()

        self.total_overhead_dict = {
            'dynamic config': 0,
            'forecasting': 0,
            'dp solver': {
                'adaptive batching': 0,
                'auto-scaling': 0
            }
        }
        self.total_epoch_cost = 0

    def prepare_stream_parallel(self, model, criterion, **kwargs):
        if not kwargs.get('gradient_as_bucket_view'):
            kwargs['gradient_as_bucket_view'] = True
            self.trainer_print('gradient_as_bucket_view must be True')
        super().prepare_stream_parallel(model, criterion, **kwargs)

    def prepare_weight_sync_method(self, optimizer, scheduler=None, param_groups_func=None):
        self._set_local_optimizers(optimizer, param_groups_func)
        self._set_local_schedulers(scheduler)
        self._prepare_gradient_based_metric()
        if self.prepared_for_ddp:
            if self.weight_sync_method == 'overlap':
                self._prepare_overlap_optimizer_with_ddp()
            if self.weight_sync_method == 'sequential':
                self.register_comm_hook()
        else:
            raise RuntimeError("[ERROR] Without DDP, AdaptiveIIDPTrainer cannot work")

    def prepare_adaptive_data_loader(self, data_loader):
        if not isinstance(data_loader, torch.iidp.data.AdaptiveDataLoader):
            raise ValueError(f'Only support torch.iidp.data.AdaptiveDataLoader, but {type(data_loader)}')
        self.data_loader = data_loader

    def _prepare_overlap_optimizer_with_ddp(self):
        self._create_stream_for_optimizer()
        self._check_overlap_with_ddp()
        for i, (local_model, state, hook) in enumerate(zip(self.local_models, self.states, self.hooks)):
            if i == 0:
                hook = self._create_optimizer_hook(hook)
            local_model.register_comm_hook(state=state, hook=hook)

    def register_comm_hook(self):
        #self.trainer_print(f'register_comm_hook() - self.states: {self.states} | self.hooks: {self.hooks}', 'debug')
        for _, (local_model, state, hook) in enumerate(zip(self.local_models, self.states, self.hooks)):
            local_model.register_comm_hook(state=state, hook=hook)

    def prepare_adaptive_training(self):
        # NOTE: Resource configuration is set up by the assumption of checkpoint-based restart
        # TODO: self.available_server_name_list is given by elastic agent by registered GPU servers (workers)
        if self.adaptive_batch_params["enable_adjust"] is True:
            self.available_server_name_list = self.adaptive_batch_params["available_servers"]
            self.cluster_manager = IIDPClusterManager(
                    self.adaptive_batch_params["gpu_cluster_info"],
                    self.available_server_name_list, self.gpu,
                    homo_servers=self.adaptive_batch_params["homo_servers"],
                    resource_alloc_unit=self.adaptive_batch_params["resource_alloc_unit"])
        self._prepare_iidp_configurator()

        if self.local_models == [] or self.local_optimizers == [] or self.data_loader is None:
            raise ValueError(
                f'Before calling prepare_adaptive_training(), model, optimizer and data loader must be configured')
        if self.is_elastic_training is True:
            self._prepare_checkpoint_based_restart()
            # NOTE: Pre-build configuratos for all candidate global resource
            self.future_configurator.prepare()

    def _prepare_checkpoint_based_restart(self):
        if self.is_elastic_training is False:
            raise ValueError(
                f'_prepare_checkpoint_based_restart() must be called if elf.is_elastic_training is True'
            )
        if self.is_resource_reallocated:
            self.trainer_print(f'Load checkpoint: {self._checkpoint_path}')
            self.load_checkpoint()

        # If self.reallocation_overhead > 0 after loading checkpoint,
        # it indicates the overhead has already been measured.
        if self.reallocation_overhead < 0:
            if dist.get_rank() == 0:
                ckpt_dir = 'profile_ckpt_overhead_dir'
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_file_path = os.path.join(ckpt_dir, 'checkpoint.pth')
                self.save_checkpoint(ckpt_file_path)
                self.load_checkpoint(ckpt_file_path)
                os.system(f'rm -rf {ckpt_dir}')
            dist.barrier()
            self.elastic_train_timer.update(time.time())
            self.reallocation_overhead = self.elastic_train_timer.elapsed_time
            self.trainer_print(f'Reallocation overhead = {self.reallocation_overhead:.2f} sec')

        self._rpc_client.init()

    def _prepare_gradient_based_metric(self):
        self.trainer_print('Prepare for adaptive training!')
        if self.adaptive_batch_params["metric"] == 'similarity':
            self._prepare_cosine_similarity()
        else: # TODO: suuport various metrics - e.g, GNS, Norm
            raise ValueError(f'Not support other gradient-based metric except similarity: {self.adaptive_batch_params["metric"]}')

    def _prepare_iidp_configurator(self):
        # NOTE
        # 1) Profile data dir must be placed on each local server even GPU type is same among another servers
        # 2) Porfile data on all of local servers must be placed on every servers (e.g, NFS)
        self.local_config = IIDPConfig(self.local_batch_size, self.num_models, self.accum_step, self.weight_sync_method)
        if self.adaptive_batch_params["enable_adjust"] is True:
            self.configurator = IIDPConfigurator(
                self.adaptive_batch_params["comp_profile_dir"],
                self.adaptive_batch_params["comm_profile_dir"],
                self.adaptive_batch_params["bucket_profile_dir"],
                self.adaptive_batch_params["memory_profile_dir"],
                self.local_config,
                self.cluster_manager.global_server_info,
                self.adaptive_batch_params["batch_size_upper_bound"],
                self.adaptive_batch_params["enable_adjust_lbs"],
                self.gpu
            )
            self.print_initial_config()
            if self.is_elastic_training is True:
                self.future_configurator = IIDPFutureConfigurator(
                    self.adaptive_batch_params["comp_profile_dir"],
                    self.adaptive_batch_params["comm_profile_dir"],
                    self.adaptive_batch_params["bucket_profile_dir"],
                    self.adaptive_batch_params["memory_profile_dir"],
                    self.local_config,
                    self.cluster_manager.candidate_server_infos,
                    self.adaptive_batch_params["batch_size_upper_bound"],
                    self.adaptive_batch_params["enable_adjust_lbs"],
                    self.gpu
                )

    def print_initial_config(self):
        if self.epoch == 0 and self.is_resource_reallocated is False:
            cluster_config_str = ''
            for server_info in self.cluster_manager.global_server_info:
                cluster_config_str += (server_info.__repr__() + '\n')
            self.trainer_print(
                f'\n====================== Initial configuration ======================\n'
                f'-------------------------- Cluster --------------------------------\n'
                f'{cluster_config_str}'
                f'-------------------------------------------------------------------\n'
                f'GBS: {self.global_batch_size} | LBS: {self.local_batch_size} | '
                f'IIDP config: {self.configurator.iidp_config_map_in_cluster}\n'
                f'===================================================================='
            )
            self.trainer_print(
                f'\n========== Memory Profile Data Summary ==========\n'
                f'   LBS\t|\tGPU\t|\tMax number of VSWs\n'
                f'---------------------------------------------------\n'
                f'{get_mem_profile_data_summary(self.adaptive_batch_params["memory_profile_dir"])}'
                f'==================================================='
            )

    def _prepare_global_batch_size_prediction(self):
        self.batch_size_model = None
        if self.adaptive_batch_params["batch_size_predict_model"] == 'gaussian':
            self.trainer_print('Global batch size prediction model = Gaussian Process Regression (GPR)')
            self.batch_size_model = GaussianProcessRegressionModel()
        elif self.adaptive_batch_params["batch_size_predict_model"] == 'exp_smoothing':
            self.trainer_print('Global batch size prediction model = ExponentialSmoothing')
            self.batch_size_model = ExponentialSmoothing()
        elif self.adaptive_batch_params["batch_size_predict_model"] == 'ensemble':
            self.trainer_print('Global batch size prediction model = Ensemble learning with ExponentialSmoothing + Gaussian Process Regression (GPR)')
            models = [GaussianProcessRegressionModel(), ExponentialSmoothing()]
            rates = [0.5, 0.5]
            self.batch_size_model = EnsembleMethod(models, rates)
        else:
            raise ValueError(
                f'Not support such batch size prediction model: '
                f'{self.adaptive_batch_params["batch_size_predict_model"]}')

    def _check_user_config_is_valid(self):
        if self.epoch == 0 and self.is_resource_reallocated is False:
            check_user_config_is_valid(self.adaptive_batch_params)

    def _init_adaptive_batch_params(self):
        if self.adaptive_batch_params is not None:
            if os.path.isfile(self.adaptive_batch_params):
                self.adaptive_batch_params = read_json(self.adaptive_batch_params)
            else:
                raise ValueError(f'Adaptive config param file: {self.adaptive_batch_params} must exist')
            self.adaptive_batch_params=defaultdict(lambda: None, self.adaptive_batch_params)
            if self.is_elastic_training is True:
                if self.adaptive_batch_params["batch_size_lower_bound"] is None:
                    raise ValueError(
                        f'If is_elastic_training = True, adaptive_batch_params["batch_size_lower_bound"] must be configured')
                else:
                    if self.is_resource_reallocated is False:
                        if self.global_batch_size < self.adaptive_batch_params["batch_size_lower_bound"]:
                            raise ValueError(
                                f'Within elastic training, initial global batch size: {self.global_batch_size} '
                                f'must be < batch_size_lower_bound: {self.adaptive_batch_params["batch_size_lower_bound"]}'
                            )
                        self.adaptive_batch_params["original_batch_size"] = self.global_batch_size
                if self.adaptive_batch_params["available_servers"] is None:
                    raise ValueError(
                        f'If is_elastic_training = True, adaptive_batch_params["available_servers"] must be configured')
            else:
                self.adaptive_batch_params["original_batch_size"] = self.global_batch_size
                if self.adaptive_batch_params["batch_size_lower_bound"] is None:
                    self.adaptive_batch_params["batch_size_lower_bound"] = self.adaptive_batch_params["original_batch_size"]
                    self.trainer_print(f'batch_size_lower_bound is configured by initial global batch size: {self.global_batch_size}')
            if self.adaptive_batch_params["batch_size_lower_bound"] is not None and self.adaptive_batch_params["batch_size_upper_bound"] is not None:
                assert self.adaptive_batch_params["batch_size_upper_bound"]>=self.adaptive_batch_params["batch_size_lower_bound"]
            self.adaptive_batch_params["global_lr_modifier"]=1.0
            if self.adaptive_batch_params["enable_decrease_batch_size"] is None:
                self.adaptive_batch_params["enable_decrease_batch_size"] = True
            if self.adaptive_batch_params["enable_adjust"] is None:
                self.adaptive_batch_params["enable_adjust"] = True
            else:
                if isinstance(self.adaptive_batch_params["enable_adjust"], str): # handle to parse from json file
                    self.adaptive_batch_params["enable_adjust"] = bool(self.adaptive_batch_params["enable_adjust"] == "True")
            if self.adaptive_batch_params["enable_adjust_lbs"] is None:
                self.adaptive_batch_params["enable_adjust_lbs"] = True
            else:
                if isinstance(self.adaptive_batch_params["enable_adjust_lbs"], str): # handle to parse from json file
                    self.adaptive_batch_params["enable_adjust_lbs"] = bool(self.adaptive_batch_params["enable_adjust_lbs"] == "True")
            if self.adaptive_batch_params["verbose"] is None:
                self.adaptive_batch_params["verbose"] = True
            else:
                if isinstance(self.adaptive_batch_params["verbose"], str): # handle to parse from json file
                    self.adaptive_batch_params["verbose"] = bool(self.adaptive_batch_params["verbose"] == "True")
            if self.adaptive_batch_params["metric"] is None:
                self.adaptive_batch_params["metric"] = 'similarity'
            # TODO: sub-group partitioning - default: horizontal
            if self.adaptive_batch_params["subgroup_partitioning"] is None:
                self.adaptive_batch_params["subgroup_partitioning"] = 'horizontal'
            if self.adaptive_batch_params["batch_size_adjust_perc"] is None:
                self.adaptive_batch_params["batch_size_adjust_perc"] = 0.1 # 10%
            if self.adaptive_batch_params["batch_size_adjust_interval"] is None:
                self.adaptive_batch_params["batch_size_adjust_interval"] = 100
            else:
                # handle to parse from json file
                if isinstance(self.adaptive_batch_params["batch_size_adjust_interval"], str):
                    self.adaptive_batch_params["batch_size_adjust_interval"] = int(self.adaptive_batch_params["batch_size_adjust_interval"])
            if self.adaptive_batch_params["utility_threshold"] is None:
                self.adaptive_batch_params["utility_threshold"] = 0
            if self.adaptive_batch_params["available_servers"] is None:
                self.adaptive_batch_params["available_servers"] = []
            if self.adaptive_batch_params["batch_size_predict_model"] is None:
                self.adaptive_batch_params["batch_size_predict_model"] = 'ensemble'
        else:
            raise ValueError(f'Adaptive config param must be configured')

        self.trainer_print(f'Adaptive training parameters: {self.adaptive_batch_params}')

    def save_checkpoint(self, checkpoint_file_path=None):
        # NOTE: If checkpoint_file_path is configured, this is for testing checkpoint
        if checkpoint_file_path:
            save_ckpt_path = checkpoint_file_path
            epoch = -1 # NOTE: epoch idx starts from -1, refer to [EpochIterator]
        else:
            if not os.path.exists(self._checkpoint_dir):
                os.makedirs(self._checkpoint_dir)
                self.trainer_print(f'Make a checkpoint dir: {self._checkpoint_dir}')
            save_ckpt_path = self._checkpoint_path
            epoch = self.epoch
        self.trainer_print(f'Save checkpoint path: {save_ckpt_path}')
        self.trainer_print(f'Saved epoch: {self.epoch}')
        if self.local_schedulers:
            scheduler_state = self.main_scheduler.state_dict()
            # Scheduler for adaptive training may need data loader's get_progress()
            if scheduler_state.get('data_loader'):
                scheduler_state.pop('data_loader')
        else:
            scheduler_state = None
        trainer_state_dict = {
            'epoch': epoch,
            'total_epoch_time': self.total_epoch_time,
            'total_epoch_cost': self.total_epoch_cost,
            'step': self.sync_step,
            'gbs_trajectory': self.global_batch_size_trajectory,
            'model' : self.main_model.module.state_dict(),
            'optimizer'  : self.main_optimizer.state_dict(),
            'scheduler'  : scheduler_state,
            'data': self.data_loader.state_dict()
        }
        if self.adaptive_batch_params["metric"] == 'similarity':
            trainer_state_dict['simigrad_step'] = self.simigrad_state.step
            trainer_state_dict['global_lr_modifier'] = self.adaptive_batch_params["global_lr_modifier"]
        trainer_state_dict['total_overhead_dict'] = self.total_overhead_dict
        if os.getenv('GBS_INTERVAL_AS_EPOCH') == "1":
            trainer_state_dict['simigrad_interval'] = self.simigrad_state.interval
        if self.is_elastic_training is True:
            trainer_state_dict['initial_global_batch_size'] = self.adaptive_batch_params["original_batch_size"]
            trainer_state_dict['reallocation_overhead'] = self.reallocation_overhead
            trainer_state_dict['future_configurator'] = self.future_configurator.state_dict()
            if os.getenv('NO_BATCH_SIZE_PREDICTION_EXP') == "1":
                self.trainer_print('No need to save checkpoint of batch size prediction model', 'experimental')
            else:
                self.batch_size_model.save(self._checkpoint_dir)

        torch.save(trainer_state_dict, save_ckpt_path)

    def load_checkpoint(self, checkpoint_file_path=None):
        # NOTE: If checkpoint_file_path is configured, this is for testing checkpoint
        if checkpoint_file_path:
            load_ckpt_path = checkpoint_file_path
        else:
            load_ckpt_path = self._checkpoint_path
        self.trainer_print(f'Load checkpoint path: {load_ckpt_path}')
        loc = 'cuda:{}'.format(self.gpu) if type(self.gpu) == int else self.gpu
        checkpoint = torch.load(load_ckpt_path, map_location=loc)
        for local_model in self.local_models:
            local_model.module.load_state_dict(checkpoint['model'])
        for local_optimizer in self.local_optimizers:
            local_optimizer.load_state_dict(checkpoint['optimizer'])
        for local_scheduler in self.local_schedulers:
            local_scheduler.load_state_dict(checkpoint['scheduler'])
        self.data_loader.load_state_dict(checkpoint['data'])
        self.epoch = checkpoint['epoch']
        self.total_epoch_time = checkpoint['total_epoch_time']
        self.total_epoch_cost = checkpoint['total_epoch_cost']
        self.sync_step = checkpoint['step']
        self.global_batch_size_trajectory = checkpoint['gbs_trajectory']
        if self.adaptive_batch_params["metric"] == 'similarity':
            self.simigrad_state.step = checkpoint['simigrad_step']
            self.adaptive_batch_params["global_lr_modifier"] = checkpoint['global_lr_modifier']
        self.total_overhead_dict = checkpoint['total_overhead_dict']
        if os.getenv('GBS_INTERVAL_AS_EPOCH') == "1":
            self.simigrad_state.interval = checkpoint['simigrad_interval']
        if self.is_elastic_training is True:
            if 'initial_global_batch_size' in checkpoint.keys():
                self.adaptive_batch_params["original_batch_size"] = checkpoint['initial_global_batch_size']
            self.reallocation_overhead = checkpoint['reallocation_overhead']
            self.future_configurator.all_candidate_server_configurators = checkpoint['future_configurator']
            if os.getenv('NO_BATCH_SIZE_PREDICTION_EXP') == "1":
                self.trainer_print('No need to load checkpoint of batch size prediction model', 'experimental')
            else:
                self.batch_size_model.load(self._checkpoint_dir)

        self.trainer_print(f'Loaded epoch: {checkpoint["epoch"]+1} | iterations: {self.sync_step}')

    def _build_vertical_subgroup(self): # TODO
        num_gpus_in_server = torch.cuda.device_count()
        total_num_gpus = dist.get_world_size()
        tensor_list = [
            torch.tensor([0], dtype=torch.float32).to(self.gpu) for _ in range(total_num_gpus)
        ]
        tensor = torch.tensor([self.num_models * self.accum_step], dtype=torch.float32).to(self.gpu)
        dist.all_gather(tensor_list, tensor)
        num_models_per_gpu =  [int(tensor.item()) for tensor in tensor_list]
        num_models_per_server = num_models_per_gpu[::num_gpus_in_server]
        self.trainer_print(f'_prepare_cosine_similarity() - num_gpus_in_server: {num_gpus_in_server}', 'debug')
        self.trainer_print(f'_prepare_cosine_similarity() - num_models_per_server: {num_models_per_server}', 'debug')
        assert self.total_num_models % 2 == 0, \
            '[ERROR] To support cosine similarity, total number of models is even'
        total_num_models_in_subgroup = self.total_num_models / 2

    def _build_horizontal_subgroup(self):
        # Original simigrad group-making -> horizontal slice
        # Advantage: simple and robust for various (vsw, ga) configurations
        assert torch.distributed.get_world_size() % 2 == 0
        for i in range(torch.distributed.get_world_size()):
            self.sub_groups_idx[i%2].append(i)

    def _prepare_cosine_similarity(self):
        #self.trainer_print('_prepare_cosine_similarity()', 'debug')
        num_sub_groups = 2 # SimiGrad builds two all-reduce sub-groups
        self.sub_groups_idx = [[] for _ in range(num_sub_groups)]
        self.sub_groups = []

        if self.adaptive_batch_params["subgroup_partitioning"] == 'vertical':
            """
            try:
                self._build_vertical_subgroup()
            except:
                self.trainer_print('Partitioning sub-group vertically is impossible! => Horizontal way', 'warning')
                self._build_horizontal_subgroup()
            """
            raise NotImplementedError('[TODO] Not support subgroup_partitioning == vetical')
        else:
            self._build_horizontal_subgroup()
        self.trainer_print(f'sub group id: {self.sub_groups_idx}')
        assert len(self.sub_groups_idx) == 2
        for idx in self.sub_groups_idx:
            self.sub_groups.append(torch.distributed.new_group(idx))
        self.first_subgroup_src_rank, self.second_subgroup_src_rank = self.sub_groups_idx[0][0], self.sub_groups_idx[1][0]
        # To compare gradients of the representative rank in each sub-group => used in compute_cosine_similarity()
        self.sub_groups.append(torch.distributed.new_group([self.first_subgroup_src_rank, self.second_subgroup_src_rank]))

        self.grad_placeholders = [[] for _ in range(num_sub_groups)]
        self.cos_placeholder = torch.rand(1).to(self.gpu)

        self._prepare_simigrad_allreduce_hooks()

    def _prepare_simigrad_allreduce_hooks(self):
        self.states, self.hooks = [], []
        #self.trainer_print('_prepare_simigrad_allreduce_hooks()', 'debug')
        self.simigrad_state = torch.iidp.ddp_comm_hooks.SimiGradState(
            dist.group.WORLD, self.total_num_models,
            self.sub_groups[dist.get_rank()%2], self.grad_placeholders[dist.get_rank()%2],
            self.adaptive_batch_params["batch_size_adjust_interval"]
        )
        subgroup_allreduce_hook = torch.iidp.ddp_comm_hooks.subgroup_allreduce_hook
        main_hook = torch.iidp.ddp_comm_hooks.simigrad_allreduce_hook(subgroup_allreduce_hook)
        dummy_hook = torch.iidp.ddp_comm_hooks.dummy_hook
        for i in range(self.num_models):
            if i == 0:
                self.states.append(self.simigrad_state)
                self.hooks.append(main_hook)
            else:
                self.states.append(None)
                self.hooks.append(dummy_hook)

    def compute_cosine_similarity(self):
        if dist.get_rank() == self.first_subgroup_src_rank or dist.get_rank() == self.second_subgroup_src_rank:
            #print(f'rank: {dist.get_rank()} [DEBUG] compute_cosine_similarity() - self.simigrad_state.grad_placeholder: {self.simigrad_state.grad_placeholder}')
            self.allgather_grad_placeholders = [
                torch.cat([torch.zeros_like(grad) for grad in self.simigrad_state.grad_placeholder]) for _ in range(2)
            ]
            grad_placeholder = torch.cat([grad for grad in self.simigrad_state.grad_placeholder])
            dist.all_gather(self.allgather_grad_placeholders, grad_placeholder, group=self.sub_groups[-1])
            if dist.get_rank() == self.first_subgroup_src_rank:
                self.cos_placeholder = torch.nn.functional.cosine_similarity(self.allgather_grad_placeholders[0], self.allgather_grad_placeholders[1], dim=0)
        dist.broadcast(self.cos_placeholder, self.first_subgroup_src_rank)
        self.trainer_print(f"cosine similarity: {self.cos_placeholder}")
        self.simigrad_state.grad_placeholder = []
        # NOTE: Memory deallocation when number of VSWs changes by change_local_models_state()
        if dist.get_rank() == self.first_subgroup_src_rank or dist.get_rank() == self.second_subgroup_src_rank:
            del self.allgather_grad_placeholders
        """
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
        """

    def change_local_models_state(self, adjust_num_models_diff):
        """
        torch.cuda.synchronize()
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        """
        if self.adaptive_batch_params["enable_adjust_lbs"] is False:
            self.change_local_models_state_by_reuse(adjust_num_models_diff)
        else: # dynamic LBS
            """
            print(f'================= start of change model state =====================')
            print(f'[DEBUG] memory (MB): {round(torch.cuda.memory_allocated()/(1024*1024))}')
            print(f'[DEBUG] cached memory (MB): {round(torch.cuda.memory_cached()/(1024*1024))}')
            print(f'[DEBUG] max memory (MB): {round(torch.cuda.max_memory_allocated()/(1024*1024))}')
            """
            if adjust_num_models_diff > 0:
                #start_time = time.time()
                gc.collect()
                with torch.no_grad():
                    torch.cuda.empty_cache()
                torch.cuda.synchronize()
                #print(f'[DEBUG][trainer.py]  **************************** adjust_num_models_diff > 0')
                # Create new components for training
                # 1) stream
                #if self.num_models > len(self.model_streams):
                #    for _ in range(len(self.model_streams), self.num_models):
                #        self.model_streams.append(torch.cuda.Stream())
                #for _ in range(self.num_models):
                #    self.model_streams.append(torch.cuda.Stream())
                for _ in range(adjust_num_models_diff):
                    # 1) stream
                    self.model_streams.append(torch.cuda.Stream())
                    # 2) model
                    copied_model = copy.deepcopy(self.main_model.module)
                    self.original_local_models.append(copied_model)
                    # 3) optimizer
                    # For .zero_grad(), optimizer should be added
                    # TODO: remove optimizer except main model
                    cls = type(self.main_optimizer.__class__.__name__, (self.main_optimizer.__class__,), dict(self.main_optimizer.__dict__))
                    if self.param_groups_func:
                        params = self.param_groups_func(copied_model)
                    else:
                        params = copied_model.parameters()
                    args = self._get_required_args_value(self.main_optimizer)
                    copied_optimizer = cls(params, lr=self.main_optimizer.defaults['lr'], *args)
                    copied_optimizer.load_state_dict(self.main_optimizer.state_dict())
                    copied_optimizer.zero_grad()
                    self.local_optimizers.append(copied_optimizer)
                find_unused_parameters = self.main_model.find_unused_parameters
                gradient_as_bucket_view = self.main_model.gradient_as_bucket_view
                for idx in range(self.prev_num_models, self.num_models):
                    with torch.cuda.stream(self.model_streams[idx]):
                        local_ddp_module = torch.nn.parallel.DistributedDataParallel(
                            self.original_local_models[idx], device_ids=[self.gpu], output_device=[self.gpu],
                            find_unused_parameters=find_unused_parameters,
                            gradient_as_bucket_view=gradient_as_bucket_view,
                            model_index=idx, num_local_models=self.num_models,
                            total_num_models=self.total_num_models,
                            sync_buffer_barrier=self._sync_buffer_barrier)
                        if self.main_model._has_rebuilt_buckets:
                            local_ddp_module.reducer.initialize_buckets(self.main_model.bucket_indices)
                            local_ddp_module._has_rebuilt_buckets = True
                        self.local_models.append(local_ddp_module)
                assert (len(self.local_models) == self.num_models) and (len(self.local_optimizers) == self.num_models)
                for i in range(self.num_models):
                    self.local_models[i].reconfigure(self.num_models, self.total_num_models, self._sync_buffer_barrier)
                # Synchornize previous models
                for i in range(self.prev_num_models, self.num_models):
                    with torch.cuda.stream(self.model_streams[i]):
                        for src_param, dst_param in \
                                zip(self.main_model.parameters(), self.local_models[i].parameters()):
                            dst_param.data.copy_(src_param.data)

                # hook - total num models
                assert self.total_num_models % 2 == 0
                dummy_hook = torch.iidp.ddp_comm_hooks.dummy_hook
                for i in range(self.prev_num_models, self.num_models):
                    self.states.append(None)
                    self.hooks.append(dummy_hook)
                    self.local_models[i].register_comm_hook(state=None, hook=dummy_hook)
                self.states[0].total_num_models = self.total_num_models
                self.states[0].subgroup_total_num_models = self.total_num_models / 2
                #print(f'[DEBUG] change_local_models_state() - rank: {dist.get_rank()} | Increase VSW overhead: {time.time()-start_time:.3f} sec')
            else:
                # Remove unused streams, models and optimizers
                """
                print(f'[DEBUG] self.num_models: {self.num_models} | '
                    f'self.prev_num_models: {self.prev_num_models} | '
                    f'adjust_num_models_diff: {adjust_num_models_diff} | '
                    f'len(self.local_models): {len(self.local_models)} | '
                    f'len(self.original_local_models): {len(self.original_local_models)} | '
                    f'len(self.local_optimizers): {len(self.local_optimizers)} | '
                    f'len(self.model_streams): {len(self.model_streams)}')
                """
                #start_time = time.time()
                for _ in range(self.num_models, self.prev_num_models):
                    # NOTE: Moving models to CPU tensor and removing it enables GPU memory to be decreased
                    # reference: https://discuss.pytorch.org/t/deleting-tensors-in-context-save-for-backward/122917/11
                    self.local_models[-1].zero_grad(set_to_none=True)
                    self.local_models[-1].cpu()
                    self.original_local_models[-1].zero_grad(set_to_none=True)
                    self.original_local_models[-1].cpu()
                    self.local_optimizers[-1].zero_grad(set_to_none=True)
                    del self.local_models[-1]
                    del self.original_local_models[-1]
                    del self.local_optimizers[-1]
                    del self.model_streams[-1]
                    del self.states[-1]
                    del self.hooks[-1]
                assert (len(self.local_models) == self.num_models) and (len(self.local_optimizers) == self.num_models)
                for i in range(self.num_models):
                    self.local_models[i].reconfigure(self.num_models, self.total_num_models, self._sync_buffer_barrier)
                if adjust_num_models_diff < 0:
                    gc.collect()
                    with torch.no_grad():
                        torch.cuda.empty_cache()
                # hook - total num models
                assert self.total_num_models % 2 == 0
                self.states[0].total_num_models = self.total_num_models
                self.states[0].subgroup_total_num_models = self.total_num_models / 2
                #print(f'[DEBUG] change_local_models_state() - rank: {dist.get_rank()} | Decrease or Remain VSW overhead: {time.time()-start_time:.3f} sec')
            if self.is_accum_mode:
                for i in range(self.num_models):
                    with torch.cuda.stream(self.model_streams[i]):
                        self.local_optimizers[i].zero_grad()
            """
            torch.cuda.synchronize()
            gc.collect()
            with torch.no_grad():
                torch.cuda.empty_cache()
            torch.cuda.synchronize()
            """
            """
            print(f'================= end of change model state =====================')
            print(f'[DEBUG] memory (MB): {round(torch.cuda.memory_allocated()/(1024*1024))}')
            print(f'[DEBUG] cached memory (MB): {round(torch.cuda.memory_cached()/(1024*1024))}')
            print(f'[DEBUG] max memory (MB): {round(torch.cuda.max_memory_allocated()/(1024*1024))}')
            print(torch.cuda.memory_summary())
            #print(torch.cuda.memory_snapshot())
            print('sleep 5 sec ..')
            time.sleep(5)
            """
            """ TEST code
            for idx in range(1, self.num_models):
                for i, j in zip(self.main_model.parameters(), self.local_models[idx].parameters()):
                    self.assert_equal(i, j)
                    if self.is_accum_mode:
                        i_grad = i.grad
                        j_grad = j.grad
                        if i_grad is not None and j_grad is not None:
                            self.assert_equal(i_grad, j_grad)
                    assert i.data.ne(j.data).sum() == 0, \
                        f"rank {dist.get_rank()}: local model {idx} does not share same parameter with main model"
            if adjust_num_models_diff > 0:
               assert self.num_models == len(self.hooks), \
                   f"when adjust_num_models_diff > 0, self.num_models == len(self.hooks), but {self.num_models} != {len(self.hooks)}"
            print(f'end of change_local_models_state() '
                  f'rank: {dist.get_rank()} | '
                  f'self.num_models: {self.num_models} | '
                  f'len(self.local_models): {len(self.local_models)} | '
                  f'len(self.model_streams): {len(self.model_streams)}', 'debug')
            """

    def change_local_models_state_by_reuse(self, adjust_num_models_diff):
        if adjust_num_models_diff > 0:
            if self.num_models > self.prev_max_num_models: # To reuse previous models
                actual_adjust_num_models_diff = self.num_models - self.prev_max_num_models
                """
                self.trainer_print(f'change_local_models_state() if adjust_num_models_diff > 0 '
                      f'rank: {dist.get_rank()} | self.num_models: {self.num_models} | '
                      f'actual_adjust_num_models_diff: {actual_adjust_num_models_diff}', 'debug')
                """
                # Create new streams for model
                for _ in range(actual_adjust_num_models_diff):
                    self.model_streams.append(torch.cuda.Stream())
                # Create new models
                for _ in range(actual_adjust_num_models_diff):
                    copied_model = copy.deepcopy(self.main_model.module)
                    self.original_local_models.append(copied_model)
                    # For .zero_grad(), optimizer should be added
                    # TODO: remove optimizer except main model
                    cls = type(self.main_optimizer.__class__.__name__, (self.main_optimizer.__class__,), dict(self.main_optimizer.__dict__))
                    if self.param_groups_func:
                        params = self.param_groups_func(copied_model)
                    else:
                        params = copied_model.parameters()
                    args = self._get_required_args_value(self.main_optimizer)
                    copied_optimizer = cls(params, lr=self.main_optimizer.defaults['lr'], *args)
                    copied_optimizer.load_state_dict(self.main_optimizer.state_dict())
                    copied_optimizer.zero_grad()
                    self.local_optimizers.append(copied_optimizer)
                find_unused_parameters = self.main_model.find_unused_parameters
                gradient_as_bucket_view = self.main_model.gradient_as_bucket_view
                for idx in range(self.prev_max_num_models, self.num_models):
                    with torch.cuda.stream(self.model_streams[idx]):
                        local_ddp_module = torch.nn.parallel.DistributedDataParallel(
                            self.original_local_models[idx], device_ids=[self.gpu], output_device=[self.gpu],
                            find_unused_parameters=find_unused_parameters,
                            gradient_as_bucket_view=gradient_as_bucket_view,
                            model_index=idx, num_local_models=self.num_models,
                            total_num_models=self.total_num_models,
                            sync_buffer_barrier=self._sync_buffer_barrier)
                        if self.main_model._has_rebuilt_buckets:
                            local_ddp_module.reducer.initialize_buckets(self.main_model.bucket_indices)
                            local_ddp_module._has_rebuilt_buckets = True
                        self.local_models.append(local_ddp_module)
                for i in range(self.prev_max_num_models):
                    self.local_models[i].reconfigure(self.num_models, self.total_num_models, self._sync_buffer_barrier)
                # Synchornize previous models
                for i in range(self.prev_num_models, self.num_models):
                    with torch.cuda.stream(self.model_streams[i]):
                        for src_param, dst_param in \
                                zip(self.main_model.parameters(), self.local_models[i].parameters()):
                            dst_param.data.copy_(src_param.data)
                self.prev_max_num_models = self.num_models
            else:
                for i in range(self.num_models):
                    self.local_models[i].reconfigure(self.num_models, self.total_num_models, self._sync_buffer_barrier)
                # Synchornize previous models
                for i in range(self.prev_num_models, self.num_models):
                    with torch.cuda.stream(self.model_streams[i]):
                        for src_param, dst_param in \
                                zip(self.main_model.parameters(), self.local_models[i].parameters()):
                            dst_param.data.copy_(src_param.data)

            # hook - total num models
            assert self.total_num_models % 2 == 0
            dummy_hook = torch.iidp.ddp_comm_hooks.dummy_hook
            if self.num_models > len(self.hooks):
                for i in range(len(self.hooks), self.num_models):
                    self.states.append(None)
                    self.hooks.append(dummy_hook)
                    self.local_models[i].register_comm_hook(state=None, hook=dummy_hook)
            self.states[0].total_num_models = self.total_num_models
            self.states[0].subgroup_total_num_models = self.total_num_models / 2
        else:
            for i in range(self.num_models):
                self.local_models[i].reconfigure(self.num_models, self.total_num_models, self._sync_buffer_barrier)
            # hook - total num models
            assert self.total_num_models % 2 == 0
            self.states[0].total_num_models = self.total_num_models
            self.states[0].subgroup_total_num_models = self.total_num_models / 2
        if self.is_accum_mode:
            for i in range(self.num_models):
                with torch.cuda.stream(self.model_streams[i]):
                    self.local_optimizers[i].zero_grad()
        torch.cuda.synchronize()
        """ TEST code
        for idx in range(1, self.num_models):
            for i, j in zip(self.main_model.parameters(), self.local_models[idx].parameters()):
                self.assert_equal(i, j)
                if self.is_accum_mode:
                    i_grad = i.grad
                    j_grad = j.grad
                    if i_grad is not None and j_grad is not None:
                        self.assert_equal(i_grad, j_grad)
                assert i.data.ne(j.data).sum() == 0, \
                    f"rank {dist.get_rank()}: local model {idx} does not share same parameter with main model"
        # if adjust_num_models_diff > 0:
        #   assert self.num_models == len(self.hooks), \
        #       f"when adjust_num_models_diff > 0, self.num_models == len(self.hooks), but {self.num_models} != {len(self.hooks)}"
        """
        """
        self.trainer_print(f'end of change_local_models_state() '
              f'rank: {dist.get_rank()} | '
              f'self.num_models: {self.num_models} | '
              f'len(self.local_models): {len(self.local_models)} | '
              f'len(self.model_streams): {len(self.model_streams)}', 'debug')
        """

    def assert_equal(self, tensor1, tensor2):
        if not isinstance(tensor1, torch.Tensor) or not isinstance(tensor2, torch.Tensor):
            raise TypeError(
                f'Both tensor1: {tensor1} and tensor2: {tensor2} must be torch.tensor')
        if not torch.equal(tensor1, tensor2):
            if dist.get_rank() == 0:
                print(f'****************** {self.__class__} | {self.weight_sync_method} test fail! ******************')
            raise ValueError(
                f'[ERROR] Equal test failed - tensor1: {tensor1[0][0]} | tensor2: {tensor2[0][0]}')

    def change_local_trainer_state(self, new_num_models, new_accum_step):
        """This method is called by only selective rank (GPU)"""
        self.prev_num_models = self.num_models
        if new_num_models != 0:
            self.num_models = self.prev_num_models + new_num_models
            #print(f'[DEBUG][{self.__class__.__name__}] change_local_trainer_state() | rank: {dist.get_rank()} | self.num_models: {self.num_models}')
            assert self.num_models > 0, f"self.num_models must be > 0"
            # Used in seq_parallel_compute() for being block different number of VSWs on inter-node
            self.sync_accum_barrier = threading.Barrier(self.num_models)
            # It is used for _sync_params() in torch/nn/parallel/distributed.py
            self._sync_buffer_barrier = [None, None]
            if self.num_models > 1:
                self._sync_buffer_barrier = [threading.Barrier(self.num_models) for _ in range(2)]

        if new_accum_step != 0:
            self.accum_step = self.accum_step + new_accum_step

        # Data loading
        self.batch_size_per_gpu = self.local_batch_size * self.num_models
        print(f'[INFO][{self.__class__.__name__}] rank: {dist.get_rank()} | self.num_models: {self.num_models} | self.accum_step: {self.accum_step}')
        print(f'[INFO][{self.__class__.__name__}] rank: {dist.get_rank()} | batch size per GPU: {self.batch_size_per_gpu}')
        self.data_loader.update_local_state(self.batch_size_per_gpu, self.num_models, self.accum_step)

    def change_global_trainer_state(self, solved_iidp_config_map):
        self._get_total_num_models()
        assert self.global_batch_size == self.local_batch_size * self.total_num_models, \
            f"GBS: {self.global_batch_size} | LBS: {self.local_batch_size} | " \
            f"total num models: {self.total_num_models} | " \
            f"rank: {dist.get_rank()} - num_models: {self.num_models} | accum_step: {self.accum_step} | " \
            f"self.configurator.iidp_config_map_in_cluster  {self.configurator.iidp_config_map_in_cluster} | " \
            f"solved_iidp_config_map: {solved_iidp_config_map} " \
            f"=> If solved_iidp_config_map is different among rank, please check config JSON file"

        self._get_all_accum_step_in_process_group()
        self.max_accum_step = max(self.all_accum_step_in_process_group)
        self.is_accum_mode = True if self.max_accum_step > 1 else False

        self._get_all_partition_size_in_process_group()
        self.data_loader.update_global_state(
                self.global_batch_size, self.all_partition_size_in_process_group)

    def change_configuration_for_iidp(self, solved_iidp_config_map):
        if len(solved_iidp_config_map) == 0:
            # NOTE: Even though solved_iidp_config_map is empty, local batch size can be changed.
            self.batch_size_per_gpu = self.local_batch_size * self.num_models
            print(f'[INFO][{self.__class__.__name__}] rank: {dist.get_rank()} | self.num_models: {self.num_models} | self.accum_step: {self.accum_step}')
            print(f'[INFO][{self.__class__.__name__}] rank: {dist.get_rank()} | batch size per GPU: {self.batch_size_per_gpu}')
            self.data_loader.update_local_state(self.batch_size_per_gpu, self.num_models, self.accum_step)
            self._set_trainer_state()
            dist.barrier()
            return
        # == step 1) Change local trainer state ==
        # 1-1) numer of local models, 1-2) accum step, 1-3) batch size per GPU
        rank = dist.get_rank()
        if rank in solved_iidp_config_map:
            new_num_models, new_accum_step = solved_iidp_config_map[rank]
            self.change_local_trainer_state(new_num_models, new_accum_step)
        else:
            new_num_models, new_accum_step = 0, 0
            self.change_local_trainer_state(new_num_models, new_accum_step)
        dist.barrier()
        # == step 2) Change global trainer state ==
        # 2-1) total number of models in process group
        # 2-2) all accum step in process group -> determine is_accum_mode
        # 2-3) data loader state
        self.change_global_trainer_state(solved_iidp_config_map)
        self._set_trainer_state()
        # == step 3) Change local VSW state ==
        # 3-1) Change (create / remove) streams, models and optimizers
        # 3-2) Change communication hook state of local models
        # NOTE [IMPORTANT] Even new number of models is zero,
        # 3-2) must be done in change_local_models_state()
        self.change_local_models_state(new_num_models)
        if len(solved_iidp_config_map) > 0:
            self.configurator.update()
        dist.barrier()

    def change_batch_size_for_iidp(self, new_global_batch_size_by_simigrad):
        solved_iidp_config_map = {} # return
        if self.adaptive_batch_params["batch_size_upper_bound"] is not None:
            new_global_batch_size_by_simigrad = min(new_global_batch_size_by_simigrad, self.adaptive_batch_params["batch_size_upper_bound"])
        if self.adaptive_batch_params["batch_size_lower_bound"] is not None:
            new_global_batch_size_by_simigrad = max(new_global_batch_size_by_simigrad, self.adaptive_batch_params["batch_size_lower_bound"])

        if self.global_batch_size == new_global_batch_size_by_simigrad:
            return solved_iidp_config_map

        self.trainer_print(f'new_global_batch_size_by_simigrad: {new_global_batch_size_by_simigrad}', 'debug')

        with Timer(f'[INFO][{self.__class__.__name__}] DP solver overhead') as dp_solver_timer:
            solved_iidp_config_map, new_local_batch_size, new_global_batch_size = \
                    self.configurator.solve_placement(
                        new_global_batch_size_by_simigrad,
                        self.global_batch_size
                    )
        self.total_overhead_dict['dp solver']['adaptive batching'] += dp_solver_timer.elapsed
        self.trainer_print(f'solved_iidp_config_map: {solved_iidp_config_map}', 'debug')
        self.trainer_print(f'new_local_batch_size: {new_local_batch_size}', 'debug')
        self.trainer_print(f'new_global_batch_size: {new_global_batch_size}', 'debug')
        if new_local_batch_size == 0 and new_global_batch_size == 0:
            self.trainer_print(f'Candidate global batch size by SimiGrad = {new_global_batch_size_by_simigrad}, '
                               f'but no virtual worker placement solution by DP', 'warning')
            return solved_iidp_config_map
        if new_global_batch_size//new_local_batch_size < dist.get_world_size():
            self.trainer_print(f'Candidate global batch size by SimiGrad = {new_global_batch_size_by_simigrad}, '
                               f'but cannot support on current numer of GPUs: {dist.get_world_size()}', 'warning')
            return solved_iidp_config_map

        # == Change local, global batch size == #
        self.local_batch_size = new_local_batch_size
        self.global_batch_size = new_global_batch_size
        assert self.global_batch_size % self.local_batch_size == 0, \
            f"New global batch size {self.global_batch_size} must be preserved local batch size: {self.local_batch_size}"
        return solved_iidp_config_map

    def change_global_batch_size_for_iidp(self, new_global_batch_size_by_simigrad):
        solved_iidp_config_map = {} # return
        if self.adaptive_batch_params["batch_size_upper_bound"] is not None:
            new_global_batch_size_by_simigrad = min(new_global_batch_size_by_simigrad, self.adaptive_batch_params["batch_size_upper_bound"])
        if self.adaptive_batch_params["batch_size_lower_bound"] is not None:
            new_global_batch_size_by_simigrad = max(new_global_batch_size_by_simigrad, self.adaptive_batch_params["batch_size_lower_bound"])

        new_adjust_gbs_amount = new_global_batch_size_by_simigrad - self.global_batch_size
        #self.trainer_print(f'change_global_batch_size_for_iidp() - new_global_batch_size: {new_global_batch_size_by_simigrad}', 'debug')
        #self.trainer_print(f'change_global_batch_size_for_iidp() - new_adjust_gbs_amount: {new_adjust_gbs_amount}', 'debug')
        if new_adjust_gbs_amount == 0:
            return solved_iidp_config_map
        if new_adjust_gbs_amount < 0:
            iidp_adjust_new_num_models = min(-1, round(new_adjust_gbs_amount / self.local_batch_size))
            iidp_adjust_new_num_models -= (iidp_adjust_new_num_models % 2)
        else:
            iidp_adjust_new_num_models = max(1, round(new_adjust_gbs_amount / self.local_batch_size))
            iidp_adjust_new_num_models += (iidp_adjust_new_num_models % 2)

        assert iidp_adjust_new_num_models % 2 == 0, \
                f"With SimiGrad constraint, new adjust num models: {iidp_adjust_new_num_models} are must be even"

        iidp_adjust_gbs_amount = iidp_adjust_new_num_models * self.local_batch_size
        #self.trainer_print(f'iidp_adjust_gbs_amount: {iidp_adjust_gbs_amount}', 'debug')
        candidate_global_batch_size = self.global_batch_size + iidp_adjust_gbs_amount
        total_num_workers = candidate_global_batch_size // self.local_batch_size
        if self.total_num_models == total_num_workers: # No global batch size change
            return solved_iidp_config_map
        if total_num_workers < dist.get_world_size():
            self.trainer_print(f'Candidate global batch size by SimiGrad = {candidate_global_batch_size}, '
                               f'but cannot support on current numer of GPUs: {dist.get_world_size()}', 'warning')
            return solved_iidp_config_map
        solved_iidp_config_map = self.configurator.solve_placement(total_num_workers, self.global_batch_size)
        #self.trainer_print(f'solved_iidp_config_map: {solved_iidp_config_map}', 'debug')
        if solved_iidp_config_map == {}:
            self.trainer_print(f'Candidate global batch size by SimiGrad = {candidate_global_batch_size}, '
                               f'but no virtual worker placement solution by DP', 'warning')
            return solved_iidp_config_map

        # == Change Global batch size == #
        self.global_batch_size += iidp_adjust_gbs_amount
        #self.trainer_print(f'Change Global batch size ==> self.global_batch_size: {self.global_batch_size}', 'debug')
        assert self.global_batch_size % self.local_batch_size == 0, \
            f"New global batch size {self.global_batch_size} must be preserved local batch size: {self.local_batch_size}"
        return solved_iidp_config_map

    def scale_lr(self, new_global_batch_size):
        if self.adaptive_batch_params["metric"] == 'similarity':
            if self.is_elastic_training is True:
                initial_batch_size = self.adaptive_batch_params["batch_size_lower_bound"]
            else:
                # NOTE: For some reasons, initial batch size is smaller than batch_size_lower_bound
                # Then, learning rate for batch_size_lower_bound cannot work well with smaller initial batch size
                initial_batch_size = max(
                    self.adaptive_batch_params["original_batch_size"], self.adaptive_batch_params["batch_size_lower_bound"])
            # Square-root scaling
            new_ratio = math.sqrt(new_global_batch_size / initial_batch_size)
            if self.adaptive_batch_params["lr_adjust_factor"] is not None:
                new_ratio = (new_ratio-1) * self.adaptive_batch_params["lr_adjust_factor"] + 1
            self.adaptive_batch_params["global_lr_modifier"] = new_ratio
            self.trainer_print(f"scale_lr() - The learning rate modifier was updated to {self.adaptive_batch_params['global_lr_modifier']}")

    def adjust_adaptive_lr(self, intialize=False):
        if self.adaptive_batch_params["metric"] == 'similarity':
            if not intialize:
                """
                self.trainer_print(f'adjust_adaptive_lr() - global lr modifier: {self.adaptive_batch_params["global_lr_modifier"]} '
                                   f'initial lr: {self.main_optimizer.param_groups[0]["lr"]} | '
                                   f'at local accum step: {self.local_accum_step} | sync step: {self.simigrad_state.step}')
                """
                for param_group in self.main_optimizer.param_groups:
                    param_group['lr']*=self.adaptive_batch_params["global_lr_modifier"]
                    lr = param_group['lr']
                if self.simigrad_state.step % self.simigrad_state.interval == 0:
                    self.trainer_print(f"adjust_adaptive_lr() - scaled LR: {lr} at step: {self.simigrad_state.step}")
                """
                self.trainer_print(f'adjust_adaptive_lr() - global lr modifier: {self.adaptive_batch_params["global_lr_modifier"]} '
                                   f'param groups lr: {self.main_optimizer.param_groups[0]["lr"]} | '
                                   f'at local accum step: {self.local_accum_step} | sync step: {self.simigrad_state.step}')
                self.trainer_print(f"adjust_adaptive_lr() - scaled LR: {lr} at local accum step: {self.local_accum_step} | "
                                   f"sync step: {self.simigrad_state.step}")
                """
            else:
                #for param_group in self.main_optimizer.param_groups:
                #    lr = param_group['lr']
                #self.trainer_print(f"step() - prev scaled LR: {lr} | lr modifier: {self.adaptive_batch_params['global_lr_modifier']} at step: {self.simigrad_state.step}")
                for param_group in self.main_optimizer.param_groups:
                    param_group['lr']/=self.adaptive_batch_params["global_lr_modifier"]
                    lr = param_group['lr']
                if self.simigrad_state.step % self.simigrad_state.interval == 0:
                    self.trainer_print(f"adjust_adaptive_lr() - initial LR: {lr} at step: {self.simigrad_state.step}")

    def adjust_adaptive_global_batch_size(self):
        if self.adaptive_batch_params["metric"] == 'similarity':
            self.trainer_print(f"step {self.simigrad_state.step} - Current cos similiarity {self.cos_placeholder} for batch size {self.global_batch_size} ")
            # [SimiGrad] Algorithm 1 - 2) If Φ < γ, target batch size B = 1.1B, else, B = 0.9B
            # [IIDP] - target batch size with constraint to preserve local batch size
            if self.cos_placeholder < self.adaptive_batch_params["similarity_target"]:
                new_global_batch_size_by_simigrad = self.global_batch_size * (1 + self.adaptive_batch_params["batch_size_adjust_perc"])
                if self.adaptive_batch_params["enable_adjust_lbs"] is True:
                    solved_iidp_config_map = self.change_batch_size_for_iidp(new_global_batch_size_by_simigrad)
                else:
                    solved_iidp_config_map = self.change_global_batch_size_for_iidp(new_global_batch_size_by_simigrad)
                self.trainer_print(f'step {self.simigrad_state.step} - current similarity < target ==> increase batch size !!! - new batch size: {self.global_batch_size}')
            elif self.cos_placeholder > self.adaptive_batch_params["similarity_target"] and self.global_batch_size > 1:
                new_global_batch_size_by_simigrad = self.global_batch_size * (1 - self.adaptive_batch_params["batch_size_adjust_perc"])
                if self.adaptive_batch_params["enable_adjust_lbs"] is True:
                    solved_iidp_config_map = self.change_batch_size_for_iidp(new_global_batch_size_by_simigrad)
                else:
                    solved_iidp_config_map = self.change_global_batch_size_for_iidp(new_global_batch_size_by_simigrad)
                self.trainer_print(f'step {self.simigrad_state.step} - current similarity > target  ==> decrease batch size !!! - new batch size: {self.global_batch_size}')
            with Timer(f'[INFO][{self.__class__.__name__}] Dynamic VSW config overhead') as dynamic_config_timer:
                self.change_configuration_for_iidp(solved_iidp_config_map)
            self.total_overhead_dict['dynamic config'] += dynamic_config_timer.elapsed
            self.scale_lr(self.global_batch_size)
        #self.trainer_print(f'self.epoch: {self.epoch} | self.global_batch_size_trajectory: {self.global_batch_size_trajectory}', 'debug')
        self.global_batch_size_trajectory[self.epoch].append([self.simigrad_state.step, self.global_batch_size])

    def compute(self, data):
        if self.adaptive_batch_params["enable_adjust"] and self.local_accum_step == 0:
        #if self.adaptive_batch_params["enable_adjust"]:
            self.adjust_adaptive_lr()
        super().compute(data)

    def step(self):
        # NOTE: As IIDP has overlapping backward pass and optimizer.step(),
        # scaled LR must be adopted before forward pass in compute() method
        if super().step() is False:
            return False

        if os.getenv('GBS_INTERVAL_AS_EPOCH') == "1":
            if self.adaptive_batch_params["enable_adjust"]:
                self.adjust_adaptive_lr(intialize=True)
            #self.trainer_print(f'self.simigrad_state.step: {self.simigrad_state.step} | self.simigrad_state.interval: {self.simigrad_state.interval}', 'debug')
            # == Measure aggregated gradients in sub-group only once in an epoch ==
            if self.simigrad_state.step == self.simigrad_state.interval:
                self.simigrad_state.done_epoch = True # allreduce hook for SimiGrad will work at next step
                if self.adaptive_batch_params["metric"] == 'similarity':
                    self.simigrad_state.step += 1
                return True
            if self.simigrad_state.done_epoch is True:
                self.simigrad_state.done_epoch = False
            # =====================================================================
            if self.adaptive_batch_params["metric"] == 'similarity':
                self.simigrad_state.step += 1
            return True

        if self.adaptive_batch_params["enable_adjust"]:
            self.adjust_adaptive_lr(intialize=True)
        if self.adaptive_batch_params["metric"] == 'similarity' and \
                self.simigrad_state.step % self.simigrad_state.interval == 0:
            self.compute_cosine_similarity()
        if self.adaptive_batch_params["enable_adjust"] and \
                self.simigrad_state.step % self.simigrad_state.interval == 0:
            self.adjust_adaptive_global_batch_size()
        if self.adaptive_batch_params["enable_adjust"] and \
                self.simigrad_state.step % self.adaptive_batch_params["batch_size_adjust_interval"] == 0:
            self.trainer_print(f"step() New local batch size {self.local_batch_size}")
            self.trainer_print(f"step() New train batch size {self.global_batch_size}\n==============================")

        if self.adaptive_batch_params["metric"] == 'similarity':
            self.simigrad_state.step += 1

        return True

    @contextmanager
    def measure_epoch_time(self):
        try:
            start_time = time.time()
            yield
        finally:
            self.elapsed_time = int(time.time() - start_time)
            if self.is_resource_reallocated:
                self.elapsed_time += int(self.reallocation_overhead)
                self.trainer_print(f'Epoch time: {self.elapsed_time} ' \
                                   f'(include reallocation time: {self.reallocation_overhead:.2f} sec)')
            else:
                self.trainer_print(f'Epoch time: {self.elapsed_time}')
            self.total_epoch_time += self.elapsed_time

    def current_resource_info_parser(self, global_server_info):
        resource_info_dict = {}
        resource_info_dict['total_num_gpus'] = global_server_info.total_num_gpus
        for server_info in global_server_info:
            if server_info.resource_info.device_name in resource_info_dict.keys():
                resource_info_dict[server_info.resource_info.device_name] += server_info.resource_info.num_gpus_in_server
            else:
                resource_info_dict[server_info.resource_info.device_name] = server_info.resource_info.num_gpus_in_server
        self.trainer_print(f'current resource info: {resource_info_dict}')

    def measure_epoch_cost(self):
        total_cost_per_epoch = 0
        for server_info in self.cluster_manager.global_server_info:
            total_cost_per_epoch += estimate_cost(
                    server_info.resource_info.tfplos,
                    server_info.resource_info.num_gpus_in_server,
                    self.elapsed_time / 3600 # convert to sec to hour
                )
        self.trainer_print(f'Epoch cost: {total_cost_per_epoch:.2f}')
        self.total_epoch_cost += total_cost_per_epoch

    def remaining_epochs(self, final_epochs):
        self.epoch_iterator.final_epochs = final_epochs
        try:
            for epoch in self.epoch_iterator.__iter__():
                self.global_batch_size_trajectory.append([])
                if os.getenv('GBS_INTERVAL_AS_EPOCH') == "1":
                    self.trainer_print('*************** GBS_INTERVAL_AS_EPOCH = 1 ***************', 'experiment')
                    # [GBS prediction] ES model requires at least two data points
                    if epoch == 0:
                        self.global_batch_size_trajectory[epoch].append([0, self.global_batch_size])
                    # As GBS does not change during epoch, interval can be calculated by the below
                    self.simigrad_state.interval += int(len(self.data_loader.dataset)/self.global_batch_size)-1
                    #self.trainer_print(f'self.global_batch_size: {self.global_batch_size} | self.data_loader.dataset: {len(self.data_loader.dataset)}',  'experiment')
                    self.trainer_print(f'GBS_INTERVAL_AS_EPOCH - interval: {self.simigrad_state.interval}',  'experiment')
                #self.trainer_print(f'remaining_epochs() - epoch: {epoch}', 'debug')
                # [IIDP] For result parsing
                if self.is_elastic_training is True:
                    self.current_resource_info_parser(self.cluster_manager.global_server_info)
                yield epoch
                if os.getenv('GBS_INTERVAL_AS_EPOCH') == "1":
                    self.trainer_print('*************** GBS_INTERVAL_AS_EPOCH = 1 ***************', 'experiment')
                    if self.adaptive_batch_params["metric"] == 'similarity':
                        self.compute_cosine_similarity()
                    if self.adaptive_batch_params["enable_adjust"]:
                        self.adjust_adaptive_global_batch_size()
                    if self.adaptive_batch_params["enable_adjust"]:
                        self.trainer_print(f"step() New local batch size {self.local_batch_size}")
                        self.trainer_print(f"step() New train batch size {self.global_batch_size}\n==============================")
                #print(f'[DEBUG][rank: {dist.get_rank()}] =============> after yield epoch {epoch} ..')
                is_last_epoch = (epoch == len(self.epoch_iterator)-1)
                if self.is_elastic_training is True:
                    if self._trainer_id == 0 and not is_last_epoch:
                        # NOTE: If auto-scaling is requested,
                        # the below code after the method at rank 0 do not reach
                        self.prepare_joint_adaptive_training_by_forecasting()
                    if is_last_epoch:
                        # NOTE: If not last epoch, it is called at prepare_joint_adaptive_training_by_forecasting()
                        self.measure_epoch_cost()
                dist.barrier()
                self.is_resource_reallocated = False
        finally:
            self.print_final_results()

        dist.barrier()
        #self.trainer_print(f'Initial global batch size: {self.adaptive_batch_params["original_batch_size"]}', 'debug')
        if self.is_elastic_training is True:
            # NOTE: _local_rank is defined only if is_elastic_training = True
            if self._local_rank == 0:
                self._rpc_client.shutdown()

    def print_final_results(self):
        if self.is_elastic_training is True:
            self.trainer_print(f'Total epoch time (sec): {self.total_epoch_time+int(self.total_overhead_dict["forecasting"])}')
            self.trainer_print(f'Total epoch cost (dollar): {self.total_epoch_cost:.2f}')
            self.trainer_print(f'Total train + reallocation time (sec): {self.total_epoch_time}')
            self.trainer_print(f'Total epoch time: {datetime.timedelta(seconds=self.total_epoch_time+int(self.total_overhead_dict["forecasting"]))}')
            self.trainer_print(f'Total forecasting overhead (sec): {self.total_overhead_dict["forecasting"]:.3f}', 'experimental')
        else:
            self.trainer_print(f'Total epoch time (sec): {self.total_epoch_time}')
            self.trainer_print(f'Total epoch time: {datetime.timedelta(seconds=self.total_epoch_time)}')
        self.trainer_print(f'Total dynamic VSW config overhead (sec): {self.total_overhead_dict["dynamic config"]:.3f}', 'experimental')
        total_dp_solver_overhead = 0
        for component, time in self.total_overhead_dict['dp solver'].items():
            total_dp_solver_overhead += time
            self.trainer_print(f'Total DP solver overhead for {component} (sec): {time:.3f}', 'experimental')
        self.trainer_print(f'Total DP solver overhead (sec): {total_dp_solver_overhead:.3f}', 'experimental')

    def prepare_joint_adaptive_training_by_forecasting(self):
        with Timer(f'[INFO][{self.__class__.__name__}] Forecasting overhead') as forecast_timer:
            if os.getenv('NO_BATCH_SIZE_PREDICTION_EXP') == "1":
                    self.trainer_print('Without batch size prediction, prepare joint-adaptve training', 'experimental')
                    predicted_gbs_trajectory = []
            else:
                #print(f'[DEBUG][rank: {dist.get_rank()}] [epoch {self.epoch}] =============> before self.predict_global_batch_size()')
                predicted_gbs_trajectory = self.predict_global_batch_size()
                #print(f'[DEBUG][rank: {dist.get_rank()}] [epoch {self.epoch}] =============> after self.predict_global_batch_size()')
            gbs_trajectory = [self.global_batch_size] + predicted_gbs_trajectory
            # [IIDP] For result parsing
            self.trainer_print(f'Naive predicted GBS trajectory before estimate_efficient_resource: {gbs_trajectory}', 'info')
            #self.trainer_print(f'predicted gbs_trajectory: {gbs_trajectory}', 'info')
            #print(f'[DEBUG][rank: {dist.get_rank()}] [epoch {self.epoch}] =============> before self.estimate_efficient_resource()')
            best_server_info, best_iidp_config_map, expected_avg_utility, expected_gbs_trajectory \
                = self.estimate_efficient_resource(gbs_trajectory)
        self.total_overhead_dict['forecasting'] += forecast_timer.elapsed
        # NOTE: self.elapsed_time is used to measure cost, so forecasting time is added to it
        self.elapsed_time += forecast_timer.elapsed
        self.measure_epoch_cost()
        # [IIDP] For result parsing
        assert len(expected_gbs_trajectory) > 0, f'len(expected_gbs_trajectory) == 0 - {expected_gbs_trajectory}'
        self.trainer_print(f'predicted gbs_trajectory: {expected_gbs_trajectory}', 'info')
        self.trainer_print(f'[epoch {self.epoch}] =============> after self.estimate_efficient_resource()')
        self.trainer_print(f'best server info: {best_server_info} | '
                           f'best IIDP config map: {best_iidp_config_map} | '
                           f'expected avg utility: {expected_avg_utility}', 'debug')
        self.trainer_print(f'best_server_info != self.cluster_manager.global_server_info => '
                           f'{best_server_info != self.cluster_manager.global_server_info}', 'debug')
        self.trainer_print(f'[epoch {self.epoch}] =======================================================')
        if best_server_info is not None and len(best_iidp_config_map) > 0 and \
                    best_server_info != self.cluster_manager.global_server_info:
            self.trainer_print('********************** Resource auto-scaling !!! **********************')
            self.trainer_print(f'[epoch: {self.epoch}] best server info: {best_server_info} | LBS: {self.local_batch_size}'
                f'best IIDP config map: {best_iidp_config_map} | '
                f'expected avg utility: {expected_avg_utility}', 'info')
            self.trainer_print('***********************************************************************')
            #print(f'[DEBUG][rank: {dist.get_rank()}] [epoch {self.epoch}] =============> before self.request_resource_scaling()')
            self.request_resource_scaling(best_iidp_config_map)

    def predict_global_batch_size(self):
        def train_batch_size_model():
            x_train_list, y_train_list = [], []
            self.trainer_print(
                f'train_batch_size_model() - self.epoch: {self.epoch} | ' \
                f'self.global_batch_size_trajectory[self.epoch]: {self.global_batch_size_trajectory[self.epoch]}',
                'debug')
            for step, gbs in self.global_batch_size_trajectory[self.epoch]:
                x_train_list.append(step)
                y_train_list.append(gbs)
            self.batch_size_model.train(x_train_list, y_train_list)
        # Train
        if len(self.global_batch_size_trajectory[self.epoch]) > 0:
            train_batch_size_model()
        # Prepare around next steps for prediction
        x_pred_list, iidp_gbs_trajectory = [], []
        default_step = self.adaptive_batch_params["batch_size_adjust_interval"]
        num_dataset = len(self.data_loader.dataset)
        total_steps_at_next_epoch = num_dataset // self.global_batch_size
        number_of_adjust_batch_size_at_next_epoch = total_steps_at_next_epoch // default_step
        around_steps_at_next_epoch = []
        for i in range(1, number_of_adjust_batch_size_at_next_epoch+1):
            around_steps_at_next_epoch.append(self.sync_step + default_step*i)
        self.trainer_print(
            f'predict_global_batch_size() - total_steps_at_next_epoch: {total_steps_at_next_epoch} | ' \
            f'number_of_adjust_batch_size_at_next_epoch: {number_of_adjust_batch_size_at_next_epoch} | ' \
            f'around_steps_at_next_epoch: {around_steps_at_next_epoch}',
            'debug')
        for step in around_steps_at_next_epoch:
            x_pred_list.append(step)
        # Predict
        if len(x_pred_list) > 0:
            try:
                y_pred_mean = self.batch_size_model.evaluate(x_pred_list)
            except Exception as e:
                self.trainer_print(
                    f'predict_global_batch_size() - x_pred_list: {x_pred_list} | '
                    f'self.epoch: {self.epoch} | '
                    f'global_batch_size_trajectory at epoch: {self.global_batch_size_trajectory[self.epoch]}\n'
                    f'total global_batch_size_trajectory: {self.global_batch_size_trajectory}', 'error')
                raise e
            predicted_gbs_trajectory = y_pred_mean.ravel()
            # For static local batch size
            if self.adaptive_batch_params["enable_adjust_lbs"] is False:
                for predicted_global_batch_size in predicted_gbs_trajectory:
                    total_num_virtual_workers = (int(predicted_global_batch_size) // self.local_batch_size)
                    total_num_virtual_workers += (total_num_virtual_workers % 2)
                    iidp_global_batch_size = total_num_virtual_workers * self.local_batch_size
                    iidp_gbs_trajectory.append(iidp_global_batch_size)
            else:
                iidp_gbs_trajectory = list(predicted_gbs_trajectory)
        return iidp_gbs_trajectory

    def estimate_efficient_resource(self, gbs_trajectory):
        best_server_info = None
        best_iidp_config_map = {}
        expected_avg_utility = -1
        best_expected_gbs_trajectory = gbs_trajectory
        default_step = self.adaptive_batch_params["batch_size_adjust_interval"]
        min_epoch_time = math.inf

        # For logging
        total_num_candidate_servers = len(self.cluster_manager.candidate_server_infos)
        print_freq = 10**(len(str(total_num_candidate_servers))-1)
        self.trainer_print(
            f'[{self.__class__.__name__}] Total number of candidate severs to estimate: '
            f'{total_num_candidate_servers}'
        )
        for server_id, candidate_server_info in enumerate(self.cluster_manager.candidate_server_infos):
            verbose = (server_id % print_freq == 0)
            if verbose:
                log_str = f'[{server_id} / {total_num_candidate_servers}] ' \
                    f'Start to estimate efficient server config with predicted GBS at next epoch'
                length = len(log_str) + 1
                self.trainer_print('=' * length)
                self.trainer_print(log_str)
                self.trainer_print('=' * length)
            #self.trainer_print(f'candidate_server_info: {candidate_server_info}', 'debug')
            # [EXPERIMENTAL] - Not decrease total number of GPUs
            """
            if candidate_server_info.total_num_gpus < dist.get_world_size():
                self.trainer_print(
                    f'Not decrease total number of GPUs: {candidate_server_info.total_num_gpus} | '
                    f'{dist.get_world_size()}', 'experimental')
                continue
            """
            self.future_configurator.update(server_id, self.local_batch_size, self.global_batch_size)
            #self.trainer_print(f'self.future_configurator.global_server_info: {self.future_configurator.global_server_info}', 'debug')
            epoch_duration = 0
            # [EXPERIMENTAL] - Reallocation penalty
            """
            if candidate_server_info != self.cluster_manager.global_server_info:
                epoch_duration += self.reallocation_overhead
            else: # debug
                self.trainer_print(
                    f'Current global server: {self.cluster_manager.global_server_info} | '
                    f'candidate: {candidate_server_info}', 'debug'
                )
            """
            total_utility = 0
            init_config_map_next_epoch = {}
            remaining_num_dataset = len(self.data_loader.dataset)
            expected_gbs_trajectory = []
            for gbs_idx, gbs in enumerate(gbs_trajectory):
                step = default_step
                if gbs_idx == 0: # current global batch size
                    step = default_step - (self.sync_step % default_step)
                # With last GBS, (iteration * GBS) must process all remaining dataset
                if gbs_idx == len(gbs_trajectory)-1:
                    step = (remaining_num_dataset // gbs) + 1
                """
                self.trainer_print(
                    f'estimate_time_and_utility() - argument => '
                    f'gbs: {gbs} | step: {step} | number of remaining dataset: {remaining_num_dataset}', 'debug')
                """
                with Timer(f'[INFO][{self.__class__.__name__}] DP solver overhead for auto-scaling', verbose) as dp_solver_timer:
                    time, utility, iidp_config_map, expected_gbs, expected_step \
                        = self.future_configurator.estimate_time_and_utility(gbs, step, remaining_num_dataset)
                self.total_overhead_dict['dp solver']['auto-scaling'] += dp_solver_timer.elapsed
                if time == math.inf:
                    self.trainer_print(f'configuration is impossble for expected gbs: {expected_gbs}')
                    break
                if gbs_idx == 0: # current global batch size
                    init_config_map_next_epoch = iidp_config_map
                if expected_step <= 0 and time != math.inf:
                    continue
                """
                self.trainer_print(
                    f'estimate_time_and_utility() - return => '
                    f'gbs: {gbs} | step: {step} | time: {time} | utility: {utility} | iidp config map: {iidp_config_map} | '
                    f'expected_gbs: {expected_gbs} | expected_step: {expected_step}', 'debug')
                """
                remaining_num_dataset -= expected_gbs*expected_step
                expected_gbs_trajectory.append(expected_gbs)
                epoch_duration += time
                total_utility += utility
                avg_utility = total_utility / len(gbs_trajectory)
                if epoch_duration == math.inf:
                    break
                """
                self.trainer_print(
                    f'number of remaining dataset: {remaining_num_dataset} | '
                    f'expected_gbs_trajectory: {expected_gbs_trajectory} | '
                    f'time: {time} | utility: {utility}', 'debug')
                """
            #self.trainer_print(f'epoch duration: {epoch_duration} | avg utility: {avg_utility}', 'debug')
            if (epoch_duration > 0 and epoch_duration <= min_epoch_time) \
                    and avg_utility >= self.adaptive_batch_params["utility_threshold"]:
                min_epoch_time = epoch_duration
                #self.trainer_print(f'*************************************************************', 'debug')
                #self.trainer_print(f'===> min epoch time (sec): {epoch_duration:.2f} | utility (<= 1): {avg_utility:.2f}', 'debug')
                #self.trainer_print(f'*************************************************************', 'debug')
                best_server_info = candidate_server_info
                best_iidp_config_map = init_config_map_next_epoch
                expected_avg_utility = avg_utility
                best_expected_gbs_trajectory = expected_gbs_trajectory
        self.trainer_print(f'*********************** estimate_efficient_resource() ***********************', 'debug')
        self.trainer_print(f'===> min epoch time (sec): {epoch_duration:.2f} | utility (<= 1): {avg_utility:.2f}', 'debug')
        self.trainer_print(f'*****************************************************************************', 'debug')
        return best_server_info, best_iidp_config_map, \
                expected_avg_utility, best_expected_gbs_trajectory

    def request_resource_scaling(self, rank_to_config_map):
        """NOTE: [Important] This function is called by only one rank"""
        #print(f'[DEBUG][rank: {dist.get_rank()}] [epoch {self.epoch}] =============> start self.request_resource_scaling()')
        def convert_config_map_to_proto(config_map):
            # NOTE: Message type is defined by torch/iidp/elastic/runtime/protobuf/trainer_to_scheuder.proto
            config_map_proto = {}
            for rank, (num_models, accum_step) in config_map.items():
                config_map_proto[rank] = f"{num_models},{accum_step}"
            return config_map_proto

        if self.data_loader.done is False:
            raise AssertionError('Resource re-allocation must be at the end of epoch')

        self.save_checkpoint()
        # NOTE: Only one rank communicates with agent to update configuration
        config_map_proto = convert_config_map_to_proto(rank_to_config_map)
        self._rpc_client.update_config(config_map_proto, self.local_batch_size)
        # NOTE: As update_config() requests asynchronously, one rank should stop here
        while True:
            pass
