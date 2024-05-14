import os
import json
import socket
import gc
import time
import asyncio
from contextlib import contextmanager

import matplotlib.pyplot as plt

import torch
import torch.distributed as dist

from torch.iidp.utils.json_utils import read_json, write_json
from torch.iidp.config.examples.config_utils import MODEL_TITLE_MAP

MAX_MEM_PROFILE_FILE_NAME = 'max_memory_profile_info.json'

# NOTE: Not change import order to avoid import module error
from torch.iidp.profiler.profiler_utils import nvidia_smi_memory_monitoring, async_run_command


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class CompProfileData(object):
    def __init__(self):
        # NOTE: If new member variable is added, ProfileJSONData().update() should be updated
        self.avg_total_time = AverageMeter('Total', ':6.3f')
        self.avg_fwd_time = AverageMeter('Fwd', ':6.3f')
        self.avg_bwd_time = AverageMeter('Bwd', ':6.3f')
        self.avg_update_time = AverageMeter('Update', ':6.3f')
        self.avg_copy_time = AverageMeter('Copy', ':6.3f')

    def __str__(self):
        return f'[Profile time (ms)] {self.avg_fwd_time} | {self.avg_bwd_time} | {self.avg_update_time} | {self.avg_copy_time} | {self.avg_total_time}'

    def update(self, fwd_time, bwd_time, update_time, copy_time, total_time):
        self.avg_fwd_time.update(fwd_time)
        self.avg_bwd_time.update(bwd_time)
        self.avg_update_time.update(update_time)
        self.avg_copy_time.update(copy_time)
        self.avg_total_time.update(total_time)


class IIDPTrainerHelper(object):
    def __init__(self, lbs, num_models):
        """
        The below member variables are required for ComputationProfiler
            ```model_name```
            ```lbs```
            ```num_models```
            ```profile_data```
        """
        self.gpu = 0
        self.model_name = None
        self.lbs = lbs
        self.num_models = num_models
        self.accum_step = 1
        self.weight_sync_method = 'sequential'

        self.trainer = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.param_groups_func = None
        self.warmup_step = 10
        self.num_minibatches = 90

        # Profile Data
        self.profile_data = CompProfileData()

    def set_optimizer(self):
        raise NotImplementedError

    def prepare(self):
        torch.manual_seed(31415)
        self.trainer = torch.iidp.IIDPTrainer(
            self.gpu, self.lbs, self.num_models, self.accum_step, self.weight_sync_method)
        self.trainer.prepare_stream_parallel(self.model, self.criterion, no_local_aggregation=True)
        self.set_optimizer()
        self.trainer.prepare_weight_sync_method(self.optimizer, None, self.param_groups_func)

    def run(self):
        raise NotImplementedError


class ProfileJSONData(object):
    def __init__(self, model_name, gpu_type, lbs, num_models):
        self.dict = {
            'model': model_name,
            'gpu_type': gpu_type,
            'lbs': lbs,
            'num_models': num_models
        }

    def update(self, runtime_profile_data):
        self.dict.update({
            'total_time': runtime_profile_data.avg_total_time.avg,
            'fwd_time': runtime_profile_data.avg_fwd_time.avg,
            'bwd_time': runtime_profile_data.avg_bwd_time.avg,
            'update_time': runtime_profile_data.avg_update_time.avg,
            'copy_time': runtime_profile_data.avg_copy_time.avg,
        })


class ComputationProfiler(object):
    def __init__(self, profiler_instance, profile_dir=None, plot_dir=None):
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()

        if profiler_instance is None:
            raise ValueError('Argument profiler_instance must be configured.')
        self.profiler_instance = profiler_instance
        if not hasattr(self.profiler_instance, 'model_name') or \
            not hasattr(self.profiler_instance, 'lbs') or \
                not hasattr(self.profiler_instance, 'num_models') or \
                    not hasattr(self.profiler_instance, 'profile_data'):
            raise AttributeError(f'[{self.__class__.__name__}] {self.profiler_instance.__dict__}')
        if not isinstance(self.profiler_instance.profile_data, CompProfileData):
            raise TypeError(f'[{self.__class__.__name__}] '
                            f'Type of self.profiler_instance.profile_data must be CompProfileData, '
                            'but {type(self.profiler_instance.profile_data)}')
        if self.profiler_instance.model_name is None or self.profiler_instance.lbs is None or \
            self.profiler_instance.num_models is None:
                raise ValueError(f'[{self.__class__.__name__}] '
                                 f'model_name: {self.profiler_instance.model_name} | '
                                 f'lbs: {self.profiler_instance.lbs} | '
                                 f'num_models: {self.profiler_instance.num_models}')
        self.model_name = self.profiler_instance.model_name
        self.lbs = self.profiler_instance.lbs
        self.num_models = self.profiler_instance.num_models
        self.accum_step = 1
        self.weight_sync_method = 'sequential'

        self.hostname = socket.gethostname()
        self.gpu_type = torch.cuda.get_device_name()
        self.profile_dir = profile_dir
        self.plot_dir = plot_dir

        self.profile_json_data = ProfileJSONData(
            self.model_name, self.gpu_type, self.lbs, self.num_models)

    def record_profile_data(self):
        self.profile_dir = os.path.join(
                self.profile_dir, self.model_name, str(self.lbs), self.hostname)
        os.makedirs(self.profile_dir, exist_ok=True)
        json_file = os.path.join(
            self.profile_dir,
            f'{self.hostname}_{self.model_name}_{self.lbs}_{self.num_models}_comp_profile.json'
        )
        try:
            with open(json_file, 'w') as jf:
                json_str = json.dumps(self.profile_json_data.dict)
                jf.write(json_str)
        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
            exit(1)

        # Test to confirm write json object to file
        json_data = read_json(json_file)
        print(json_data)

    def run(self):
        self.profiler_instance.run()
        self.profile_json_data.update(self.profiler_instance.profile_data)
        if self.profile_dir:
            self.record_profile_data()
        if self.plot_dir:
            self.plot_comp_profile_data()

    def plot_comp_profile_data(self, file_path='comp_profile_data_breakdown.png'):
        try:
            model_name_for_plot = MODEL_TITLE_MAP[self.model_name]
        except:
            print(f'[WARNING] Model name is not registerd: {self.model}')
            model_name_for_plot = self.model_name
        all_data = [self.profile_json_data.dict]
        x_data = []
        fwd_time = []
        bwd_time = []
        update_time = []
        copy_time = []
        for data in all_data:
            x_data.append(str(data['num_models']))
            fwd_time.append(data['fwd_time']/data['total_time'])
            bwd_time.append(data['bwd_time']/data['total_time'])
            update_time.append(data['update_time']/data['total_time'])
            copy_time.append(data['copy_time']/data['total_time'])
        breakdown_data = [fwd_time, bwd_time, update_time, copy_time]
        plt.clf()
        stacked_data = [0 for _ in range(len(x_data))]
        labels = ['Forward', 'Backward', 'Update', 'Copy']
        for i, data in enumerate(breakdown_data):
            plt.bar(x_data, data, bottom=stacked_data, label=labels[i], width=0.5)
            stacked_data = [prev + data for prev, data in zip(stacked_data, data)]
        plt.xlabel('Number of VSWs')
        plt.ylabel('Normalized throughput breakdown')
        plt.legend()
        title = f'{model_name_for_plot} ({self.lbs}) on {self.gpu_type}'
        plt.title(title)
        file_path = f'{socket.gethostname()}_{self.model_name}_{self.lbs}_{self.num_models}_{file_path}'
        os.makedirs(self.plot_dir, exist_ok=True)
        file_path = os.path.join(self.plot_dir, file_path)
        plt.savefig(file_path)


class DDPHelper(object):
    def __init__(self):
        """
        The below member variables are required for DDPBucketProfiler
            ```model_name```
        """
        self.gpu = 0
        self.model_name = None
        self.ddp_module = None
        self.lbs = 1
        self.model = None
        self.criterion = None
        self.step = 2
        self.bucket_size_distribution = []

    def _get_ddp_bucket_indices(self):
        raise NotImplementedError

    def get_bucket_size_distribution(self):
        self._get_ddp_bucket_indices()
        if self.ddp_module is None:
            raise TypeError(
                f'[ERROR][{self.__class__.__name__}] Member variable ddp_module is None')
        print(self.ddp_module.bucket_indices)
        bucket_size_distribution = []
        parameter_size_distribution = []
        for _, param in enumerate(self.ddp_module.ddp_register_params):
            if hasattr(param, 'index'):
                param_mem_value = round(param.nelement() * param.element_size() / (1024 ** 2), 2)
                parameter_size_distribution.append(param_mem_value)

        for bucket in self.ddp_module.bucket_indices:
            bucket_size = 0
            for param_index in bucket:
                param_size = parameter_size_distribution[param_index]
                bucket_size += param_size
            bucket_size_distribution.append(round(bucket_size, 2))
        print(f'[Profile info] bucket_size_distribution (backward order): {bucket_size_distribution}')
        self.bucket_size_distribution = bucket_size_distribution

    def run(self):
        raise NotImplementedError


class DDPBucketProfiler(object):
    def __init__(self, profiler_instance, profile_dir=None, plot_dir=None):
        torch.cuda.empty_cache()
        if not dist.is_initialized():
            torch.cuda.set_device(0)
            dist.init_process_group(
                backend='nccl', init_method='tcp://127.0.0.1:22222', world_size=1, rank=0)

        if profiler_instance is None:
            raise ValueError('Argument profiler_instance must be configured.')
        self.profiler_instance = profiler_instance
        self.model_name = self.profiler_instance.model_name

        self.profile_dir = profile_dir
        self.plot_dir = plot_dir

        self.profile_data = {
            'model': self.model_name,
            'bucket_size_distribution': []
        }

    def run(self):
        self.profiler_instance.run()
        self.profile_data['bucket_size_distribution'] = self.profiler_instance.bucket_size_distribution
        if dist.get_rank() == 0:
            if self.profile_dir:
                self.record_profile_data()
            if self.plot_dir:
                self.plot_profile_data()

    def record_profile_data(self):
        os.makedirs(self.profile_dir, exist_ok=True)
        json_file = os.path.join(
            self.profile_dir,
            f'{self.model_name}_bucket_size_profile.json'
        )
        try:
            with open(json_file, 'w') as jf:
                json_str = json.dumps(self.profile_data)
                jf.write(json_str)
        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
            exit(1)

        # Test to confirm write json object to file
        json_data = read_json(json_file)
        print(json_data)

    def plot_profile_data(self, file_path='bucket_size_distribution.png'):
        try:
            model_title = MODEL_TITLE_MAP[self.model_name]
        except:
            print(f'[WARNING] Model name is not registerd: {self.model_name}')
            model_title = self.model_name
        plt.clf()
        x = list(range(len(self.profile_data['bucket_size_distribution'])))
        plt.bar(x, self.profile_data['bucket_size_distribution'])

        plt.xlabel('Bucket order (backward)')
        plt.ylabel('Bucket size (MB)')

        title = f'{model_title}'
        plt.title(title)

        os.makedirs(self.plot_dir, exist_ok=True)
        file_path = f'{self.model_name}_{file_path}'
        file_path = os.path.join(self.plot_dir, file_path)
        plt.savefig(file_path)


class MemoryProfileJSONData(object):
    def __init__(self, gpu_type, total_memory):
        self.dict = {
            'gpu_type': gpu_type,
            'total_memory': total_memory
        }

    def update(self, runtime_profile_data):
        self.dict.update(runtime_profile_data)


class IIDPMemoryProfilerHelper(object):
    def __init__(self, lbs, num_models):
        self.gpu = 0
        self.lbs = lbs
        self.num_models = num_models
        self.accum_step = 1
        self.weight_sync_method = 'sequential'

        self.trainer = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.param_groups_func = None
        self.warmup_step = 10
        self.num_minibatches = 90
        # For debugging
        #self.warmup_step = 5
        #self.num_minibatches = 10

        # Profile Data
        # TODO: Update strong data format (potential to have different format defined by users)
        self.profile_data = {}

    def set_optimizer(self):
        raise NotImplementedError

    def prepare(self):

        torch.manual_seed(31415)
        self.trainer = torch.iidp.IIDPTrainer(
            self.gpu, self.lbs, self.num_models, self.accum_step, self.weight_sync_method)
        self.trainer.prepare_stream_parallel(self.model, self.criterion, no_local_aggregation=True)
        self.set_optimizer()
        self.trainer.prepare_weight_sync_method(self.optimizer, None, self.param_groups_func)

    def run(self):
        raise NotImplementedError


class BaseSingleGPUMemoryProfiler(object):
    def __init__(self, profiler_class, profile_dir=None):
        if profiler_class is None:
            raise ValueError('Argument profiler_class must be configured.')
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        self.profiler_class = profiler_class
        # NOTE: If profile_dir is None, not record profile data to JSON file
        self.profile_dir = profile_dir

        self.model_name = '' # defined in run() method
        self.accum_step = 1
        self.weight_sync_method = 'sequential'

        self.hostname = socket.gethostname()
        self.gpu_type = torch.cuda.get_device_name()
        self.total_gpu_memory = torch.cuda.get_device_properties(0).total_memory

        self.profile_json_data = MemoryProfileJSONData(
            self.gpu_type, self.total_gpu_memory
        )
        self.max_mem_profile_json_data = MemoryProfileJSONData(
            self.gpu_type, self.total_gpu_memory
        )

    def log(self, message, status='info'):
        print_msg = f'[{status.upper()}][{self.__class__.__name__}] {message}'
        print(print_msg)

    def record_max_mem_profile_data(self, lbs):
        lbs = str(lbs)
        profile_dir = os.path.join(
                self.profile_dir, self.model_name, lbs, self.hostname)
        os.makedirs(profile_dir, exist_ok=True)
        json_file = os.path.join(
            profile_dir,
            MAX_MEM_PROFILE_FILE_NAME
        )
        try:
            with open(json_file, 'w') as jf:
                json_str = json.dumps(self.max_mem_profile_json_data.dict)
                jf.write(json_str)
        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
            exit(1)

        # Test to confirm write json object to file
        json_data = read_json(json_file)
        print(json_data)

    def record_profile_data(self, lbs, num_models):
        lbs = str(lbs)
        profile_dir = os.path.join(
                self.profile_dir, self.model_name, lbs, self.hostname)
        os.makedirs(profile_dir, exist_ok=True)
        json_file = os.path.join(
            profile_dir,
            f'{self.hostname}_{self.model_name}_{lbs}_{num_models}_mem_profile.json'
        )
        try:
            with open(json_file, 'w') as jf:
                json_str = json.dumps(self.profile_json_data.dict)
                jf.write(json_str)
        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
            exit(1)

        # Test to confirm write json object to file
        json_data = read_json(json_file)
        print(json_data)

    def run(self):
        raise NotImplementedError


class StaticLocalBatchSizMemoryProfiler(BaseSingleGPUMemoryProfiler):
    def __init__(self, profiler_class, local_batch_size, profile_dir=None):
        super().__init__(profiler_class, profile_dir)
        if local_batch_size is None:
            raise ValueError('Argument local_batch_size must be configured.')
        if not isinstance(local_batch_size, int):
            raise ValueError(
                f'Argument local_batch_size must be integer type, '
                f'but {type(local_batch_size)}')
        self.lbs = local_batch_size

    def run(self):
        max_num_models_on_hardware = min(os.cpu_count() // torch.cuda.device_count(), 10)
        self.log(f'Max number of models that this GPU server can run: '
                 f'{max_num_models_on_hardware} | '
                 f'CPU count: {os.cpu_count()} | GPU count: {torch.cuda.device_count()}')

        is_oom_by_num_models = False
        is_oom_by_lbs = False
        while True:
            for num_models in range(1, max_num_models_on_hardware+1):
                try:
                    gc.collect()
                    with torch.no_grad():
                        torch.cuda.empty_cache()
                    self.log(f'Profiling with LBS: {self.lbs} | num models: {num_models} .. ')
                    profiler_instance = self.profiler_class(self.lbs, num_models)
                    profiler_instance.run()
                    self.model_name = profiler_instance.model_name
                    self.profile_json_data.update(profiler_instance.profile_data)
                    del profiler_instance
                    if self.profile_dir:
                        self.record_profile_data(self.lbs, num_models)
                except RuntimeError as e:
                    del profiler_instance
                    self.log(f'OOM error by LBS: {self.lbs} | num models: {num_models}')
                    max_num_models = num_models - 1
                    if max_num_models == 0:
                        is_oom_by_lbs = True
                        break
                    is_oom_by_num_models = True
                    self.max_mem_profile_json_data.update({'lbs': self.lbs, 'max_num_models': max_num_models})
                    if self.profile_dir:
                        self.record_max_mem_profile_data(self.lbs)
                    break
            if is_oom_by_lbs is True: # Reach max local batch size
                break
            if is_oom_by_num_models is True:
                break
            if is_oom_by_num_models is False: # With LBS, it can run with max_num_models_on_hardware
                self.max_mem_profile_json_data.update({'lbs': self.lbs, 'max_num_models': max_num_models_on_hardware})
                if self.profile_dir:
                    self.record_max_mem_profile_data(self.lbs)
                break


class DynamicLocalBatchSizeMemoryProfiler(BaseSingleGPUMemoryProfiler):
    def __init__(self, profiler_class, min_lbs, max_lbs=None,
                 profile_dir=None, search_lbs_fn=None):
        super().__init__(profiler_class, profile_dir)
        if min_lbs is None:
            raise ValueError('Argument min_lbs must be configured.')
        if not isinstance(min_lbs, int):
            raise ValueError(f'Argument min_lbs must be integer type, but {type(min_lbs)}')
        if search_lbs_fn is None:
            raise ValueError(
                f'Argumnet search_lbs_fn must be configured')

        self.min_batch_size = min_lbs
        self.max_batch_size = max_lbs
        self.search_lbs_fn = search_lbs_fn

    def run(self):
        max_num_models_on_hardware = min(os.cpu_count() // torch.cuda.device_count(), 10)
        self.log(f'Max number of models that this GPU server can run: '
                 f'{max_num_models_on_hardware} | '
                 f'CPU count: {os.cpu_count()} | GPU count: {torch.cuda.device_count()}')

        local_batch_size = self.min_batch_size
        is_oom_by_num_models = False
        is_oom_by_lbs = False
        while True:
            if self.max_batch_size is not None and local_batch_size > self.max_batch_size:
                break

            for num_models in range(1, max_num_models_on_hardware+1):
                profiler_instance = None
                try:
                    gc.collect()
                    with torch.no_grad():
                        torch.cuda.empty_cache()
                    self.log(f'Profiling with LBS: {local_batch_size} | num models: {num_models} .. ')
                    self.log(f'cuda memory allocated: {torch.cuda.memory_allocated()}')
                    assert torch.cuda.memory_allocated() == 0, \
                        f"Before runnig profiler, CUDA allocated memory must be 0, " \
                        f"but {round(torch.cuda.memory_allocated() / (1024*1024))} MB"
                    profiler_instance = self.profiler_class(local_batch_size, num_models)
                    profiler_instance.run()
                    self.model_name = profiler_instance.model_name
                    self.profile_json_data.update(profiler_instance.profile_data)
                    del profiler_instance
                    if self.profile_dir:
                        self.record_profile_data(local_batch_size, num_models)
                except RuntimeError as e:
                    self.log(f'OOM error by LBS: {local_batch_size} | num models: {num_models}')
                    torch.cuda.synchronize()
                    # TODO: Find the way to guarantee all theads in trainer.profile_parallel_compute()
                    # finish join() here.
                    time.sleep(10)
                    del profiler_instance
                    gc.collect()
                    with torch.no_grad():
                        torch.cuda.empty_cache()
                    max_num_models = num_models - 1
                    if max_num_models == 0:
                        is_oom_by_lbs = True
                        break
                    is_oom_by_num_models = True
                    self.max_mem_profile_json_data.update({'lbs': local_batch_size, 'max_num_models': max_num_models})
                    if self.profile_dir:
                        self.record_max_mem_profile_data(local_batch_size)
                    break
            if is_oom_by_lbs is True: # Reach max local batch size
                break
            if is_oom_by_num_models is False: # With LBS, it can run with max_num_models_on_hardware
                self.max_mem_profile_json_data.update({'lbs': local_batch_size, 'max_num_models': max_num_models_on_hardware})
                if self.profile_dir:
                    self.record_max_mem_profile_data(local_batch_size)

            local_batch_size = self.search_lbs_fn(local_batch_size)


class BaseMultiGPUMemoryProfiler(object):
    def __init__(self, profile_dir, model_name, training_scheme, shell_script,
                 timeout=60, mem_util_threshold=100, user_defined_cmd_fn=None):
        if training_scheme not in ['static', 'adaptive']:
            raise ValueError(
                f'Argument training_schemes must be configured among [static, adaptive]')
        if not os.path.exists(shell_script):
            raise ValueError(f'Argument shell_script must exist')

        self.profile_dir = profile_dir
        self.model_name = model_name

        self.hostname = socket.gethostname()
        self.gpu_type = torch.cuda.get_device_name()
        self.total_gpu_memory = torch.cuda.get_device_properties(0).total_memory

        self.max_mem_profile_json_data = MemoryProfileJSONData(
            self.gpu_type, self.total_gpu_memory
        )

        self.training_scheme = training_scheme
        self.shell_script = shell_script
        self.timeout = timeout
        self.user_defined_cmd_fn = user_defined_cmd_fn
        self.mem_util_threshold = mem_util_threshold

        self.max_num_models_on_hardware = min(os.cpu_count() // torch.cuda.device_count(), 10)
        self.log(f'Max number of models that this GPU server can run: ' \
                 f'{self.max_num_models_on_hardware} | ' \
                 f'CPU count: {os.cpu_count()} | GPU count: {torch.cuda.device_count()}')
        self.max_num_models = 1

        self.loop = None

    def log(self, message, status='info'):
        print_msg = f'[{status.upper()}][{self.__class__.__name__}] {message}'
        print(print_msg)

    def record_max_mem_profile_data(self, lbs):
        lbs = str(lbs)
        profile_dir = os.path.join(
                self.profile_dir, self.model_name, lbs, self.hostname)
        os.makedirs(profile_dir, exist_ok=True)
        json_file = os.path.join(
            profile_dir,
            MAX_MEM_PROFILE_FILE_NAME
        )
        try:
            with open(json_file, 'w') as jf:
                json_str = json.dumps(self.max_mem_profile_json_data.dict)
                jf.write(json_str)
        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
            exit(1)

        # Test to confirm write json object to file
        json_data = read_json(json_file)
        self.log(f'Record data: {json_data} to {json_file}')

    def _prepare_for_training_scheme(self):
        if self.training_scheme == 'static':
            pass
        elif self.training_scheme == 'adaptive':
            # generate mock config json file - enable_adjust = False (defined in torch/iidp/trainer.py)
            mock_adaptive_config_data = {
                "metric": "similarity",
                "enable_adjust": "False",
                "batch_size_adjust_interval": 10
            }
            mock_config_dir = 'adaptive_config'
            os.makedirs(mock_config_dir, exist_ok=True)
            self.mock_adaptive_config_file = os.path.join(
                mock_config_dir, 'adaptive_config_for_mem_profile_validation.json')
            write_json(self.mock_adaptive_config_file, mock_adaptive_config_data)
        else:
            raise ValueError(f'Not support such trainin scheme: {self.training_scheme}')

    @contextmanager
    def execute_handler(self):
        try:
            yield
        finally:
            if self.training_scheme == 'adaptive':
                self.log(f'Remove mock config file for adaptive training: {self.mock_adaptive_config_file}')
                os.system(f'rm -rf {self.mock_adaptive_config_file}')

    def _execute(self):
        lbs_str = str(self.local_batch_size)
        while True:
            try:
                rank = 0
                world_size = 1
                weight_sync_method = 'recommend'
                if self.training_scheme == 'static':
                    accum_step = 1
                    if self.user_defined_cmd_fn is not None:
                        command = self.user_defined_cmd_fn(
                            self.shell_script, rank, world_size, lbs_str,
                            self.max_num_models, accum_step, weight_sync_method
                        )
                    else:
                        command = [
                            self.shell_script, rank, world_size, lbs_str,
                            self.max_num_models, accum_step, weight_sync_method
                        ]
                elif self.training_scheme == 'adaptive':
                    accum_step = 2
                    if self.user_defined_cmd_fn is not None:
                        command = self.user_defined_cmd_fn(
                            self.shell_script, rank, world_size, lbs_str,
                            self.max_num_models, accum_step, weight_sync_method,
                            self.mock_adaptive_config_file
                        )
                    else:
                        command = [
                            self.shell_script, rank, world_size, lbs_str,
                            self.max_num_models, accum_step, weight_sync_method,
                            self.mock_adaptive_config_file
                        ]
                log_str = f'Start to profile number of VSWs ({lbs_str}): {self.max_num_models}'
                col_str = '=' * (len(log_str) + 1)
                self.log(col_str)
                self.log(log_str)
                self.log(col_str)
                with nvidia_smi_memory_monitoring(self.max_num_models, self.mem_util_threshold):
                    self.loop = asyncio.get_event_loop()
                    self.loop.run_until_complete(async_run_command(command, self.timeout))
                self.log('Success to execute command')
                self.log('Sleep 30 sec ..')
                time.sleep(30)
                self.max_num_models += 1
            except RuntimeError as e: # OOM happen
                self.log(e)
                kill_python_cmd = "kill -9 `ps | grep python | grep -v {0} | grep -v defunct | awk -F' ' '{{print $1}}'`".format(os.getpid())
                os.system(kill_python_cmd)
                kill_nvidiasmi_query_cmd = \
                    f"kill -9 `ps -ef | grep -v grep | grep \"nvidia-smi --query\" | awk '{{print $2}}' `"
                os.system(kill_nvidiasmi_query_cmd)
                self.log('Sleep 30 sec ..')
                time.sleep(30)
                self.max_num_models -= 1
                break

            if self.max_num_models >= self.max_num_models_on_hardware:
                self.log('Terminate to reach max number of VSWs to max constaint on hardware')
                break

    def run(self):
        raise NotImplementedError


class StaticLocalBatchSizeMultiGPUMemoryProfiler(BaseMultiGPUMemoryProfiler):
    def __init__(self, profile_dir, model_name, training_scheme, shell_script, local_batch_size,
                 timeout=60, mem_util_threshold=100, user_defined_cmd_fn=None):
        super().__init__(profile_dir, model_name, training_scheme, shell_script,
                         timeout, mem_util_threshold, user_defined_cmd_fn)
        if local_batch_size is None:
            raise ValueError('Argument local_batch_size must be configured.')
        if not isinstance(local_batch_size, int):
            raise ValueError(
                f'Argument local_batch_size must be integer type, '
                f'but {type(local_batch_size)}')

        self.local_batch_size = str(local_batch_size)

        self._prepare_for_training_scheme()

    def run(self):
        with self.execute_handler():
            self._execute()
            # Get max num models
            self.log('========================================================')
            self.log(f'Profiled max number of VSWs ({self.local_batch_size}): {self.max_num_models}')
            self.log('========================================================')
            if self.max_num_models >= 1:
                self.max_mem_profile_json_data.update(
                    {'lbs': self.local_batch_size, 'max_num_models': self.max_num_models}
                )
                self.record_max_mem_profile_data(self.local_batch_size)
            self.log('Sleep 30 sec ..')
            time.sleep(30)

        self.loop.close()


class DynamicLocalBatchSizeMultiGPUMemoryProfiler(BaseMultiGPUMemoryProfiler):
    def __init__(self, profile_dir, model_name, training_scheme, shell_script,
                 min_lbs, search_lbs_fn, max_lbs=None,
                 timeout=60, mem_util_threshold=100, user_defined_cmd_fn=None):
        super().__init__(profile_dir, model_name, training_scheme, shell_script,
                         timeout, mem_util_threshold, user_defined_cmd_fn)
        if min_lbs is None:
            raise ValueError('Argument min_lbs must be configured.')
        if not isinstance(min_lbs, int):
            raise ValueError(f'Argument min_lbs must be integer type, but {type(min_lbs)}')
        if search_lbs_fn is None:
            raise ValueError(
                f'Argumnet search_lbs_fn must be configured')

        self.min_batch_size = min_lbs
        self.max_batch_size = max_lbs
        self.search_lbs_fn = search_lbs_fn

        self.local_batch_size = self.min_batch_size

        self._prepare_for_training_scheme()

    def run(self):
        with self.execute_handler():
            while True:
                if self.max_batch_size is not None and self.local_batch_size > self.max_batch_size:
                    break
                self._execute()
                # Get max num models
                self.log('========================================================')
                self.log(f'Profiled max number of VSWs ({self.local_batch_size}): {self.max_num_models}')
                self.log('========================================================')
                if self.max_num_models >= 1:
                    self.max_mem_profile_json_data.update(
                        {'lbs': self.local_batch_size, 'max_num_models': self.max_num_models}
                    )
                    self.record_max_mem_profile_data(self.local_batch_size)
                else:
                    break
                self.log('Sleep 30 sec ..')
                time.sleep(30)
                self.local_batch_size = self.search_lbs_fn(self.local_batch_size)
                self.max_num_models = 1