import os
import socket
import time
import subprocess
import sys
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

import asyncio
from asyncio.subprocess import PIPE, STDOUT

from contextlib import contextmanager

from torch.iidp.utils.json_utils import read_json, write_json
from torch.iidp.profiler.profiler import MAX_MEM_PROFILE_FILE_NAME


def get_mem_profile_data_summary(profile_dir):
    summary_str = ''
    col_str = '---------------------------------------------------'
    for lbs in sorted(os.listdir(profile_dir), key=lambda x: int(x)):
        static_lbs_profile_dir = os.path.join(profile_dir, lbs)
        for server_name in os.listdir(static_lbs_profile_dir):
            max_memory_profile_file = os.path.join(
                static_lbs_profile_dir, server_name, MAX_MEM_PROFILE_FILE_NAME)
            memory_profile_json_data = read_json(max_memory_profile_file)
            max_num_models = memory_profile_json_data['max_num_models']
            gpu_type = memory_profile_json_data['gpu_type']
            #summary_str += f'LBS: {lbs} | GPU: {gpu_type} | Max number of VSWs: {max_num_models} \n'
            summary_str += f'   {lbs}\t|\t{gpu_type}\t|    {max_num_models} \n'
        summary_str += col_str+'\n'
    return summary_str


def get_max_profile_json_data(profile_dir, lbs):
    memory_profile_json_data, max_memory_profile_file = None, None
    static_lbs_profile_dir = os.path.join(profile_dir, lbs)
    for server_name in os.listdir(static_lbs_profile_dir):
        if socket.gethostname() != server_name:
            continue
        max_memory_profile_file = os.path.join(
            static_lbs_profile_dir, server_name, MAX_MEM_PROFILE_FILE_NAME)
        memory_profile_json_data = read_json(max_memory_profile_file)
    if max_memory_profile_file is None:
        current_server_max_memory_profile_file = os.path.join(
            static_lbs_profile_dir, socket.gethostname(), MAX_MEM_PROFILE_FILE_NAME)
        raise ValueError(f'No such memory profile dir: {current_server_max_memory_profile_file}')
    if memory_profile_json_data is None:
        raise ValueError('return value memory_profile_json_data is None')
    return memory_profile_json_data


def set_max_profile_json_data(profile_dir, lbs, data):
    static_lbs_profile_dir = os.path.join(profile_dir, lbs)
    max_memory_profile_file = None
    for server_name in os.listdir(static_lbs_profile_dir):
        if socket.gethostname() != server_name:
            continue
        max_memory_profile_file = os.path.join(
            static_lbs_profile_dir, server_name, MAX_MEM_PROFILE_FILE_NAME)
    if max_memory_profile_file is None:
        raise ValueError('max_memory_profile_file is None')
    write_json(max_memory_profile_file, data)


def get_max_num_models_for_static_lbs(profile_dir, lbs):
    memory_profile_json_data = get_max_profile_json_data(profile_dir, lbs)
    profiled_max_num_models = memory_profile_json_data['max_num_models']
    return profiled_max_num_models


# reference: Second solution in https://stackoverflow.com/questions/10756383/timeout-on-subprocess-readline-in-python
async def async_run_command(command, timeout=60):
    command = [str(arg) for arg in command]
    program = command[0]
    args = command[1:]
    print(f'[INFO] command: {" ".join(command)}')
    print(f'[INFO] It might take time .. please wait ..')
    proc = await asyncio.create_subprocess_exec(program, *args, stdout=PIPE, stderr=STDOUT)
    while True:
        try:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout)
        except asyncio.TimeoutError:
            pass
        else:
            if not line: # EOF
                break
            else:
                print(line.decode('utf-8').replace('\n',''))
                log_str = line.decode('utf-8')
                if 'out of memory' in log_str or 'RuntimeError' in log_str:
                    proc.kill()
                    raise RuntimeError('CUDA out of memory error')
                continue
        proc.kill() # Timeout or some criterion is not satisfied
        raise RuntimeError('TimeoutExpired - CUDA out of memory error')
    return await proc.wait() # Wait for the child process to exit


class nvidia_smi_memory_monitoring(object):
    def __init__(self, num_models, mem_util_threshold):
        self.num_models = num_models
        self.mem_util_threshold = mem_util_threshold
        self.proc = None
        if self.num_models > 1:
            nvidiasmi_query_cmd = "nvidia-smi --query-gpu=memory.total,memory.used --format=csv -lms 100 &"
            # NOTE: shell=True for background command
            self.proc = subprocess.Popen(nvidiasmi_query_cmd,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT,
                                         shell=True)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        #print(f'[DEBUG][{self.__class__.__name__}] __exit__ => exc_type: {exc_type} | exc_val: {exc_val} | exc_tb: {exc_tb}')
        is_timeout_error, is_mem_util_over = False, False
        if self.num_models > 1:
            kill_nvidiasmi_query_cmd = \
                f"kill -9 `ps -ef | grep -v grep | grep \"nvidia-smi --query\" | awk '{{print $2}}' `"
            os.system(kill_nvidiasmi_query_cmd)
            # NOTE: Safe to check max mem usage by TimeoutExpired error
            if exc_type is RuntimeError:
                if 'TimeoutExpired' in str(exc_val):
                    is_timeout_error = True
                    print(f'[INFO][{self.__class__.__name__}] '
                          f'Safe to check max mem usage by TimeoutExpired error: {exc_val}')
                else:
                    return
            memory_value_parsing_count = 0
            max_mem_util, max_mem_used = 0, 0
            for stdout in self.proc.stdout.readlines():
                log_str = stdout.decode('utf-8').replace('\n','')
                #print(f'[DEBUG] nvidia-smi memory query: {log_str}')
                if 'memory' not in log_str and 'MiB' in log_str:
                    if len(log_str.split()) != 4: # NOTE: To avoid some stdout that has only total memory size
                        continue
                    try:
                        mem_total, mem_used = float(log_str.split()[0]), float(log_str.split()[2])
                    except:
                        print(f'[ERROR][{self.__class__.__name__}] '
                              f'log_str: {log_str}\n '
                              f'log_str.split(): {log_str.split()} | '
                              f'log_str.split(): {log_str.split()}')
                        exit(1)
                    if memory_value_parsing_count == 0 and mem_used != 0:
                        #assert mem_used == 0, f"The first parsed memory used must be zero, but {mem_used}"
                        print(f'[WARNING][{self.__class__.__name__}] '
                              f'The first parsed memory used must be zero, but {mem_used} | '
                              f'Log: {log_str}')
                    memory_value_parsing_count+=1
                    mem_util = (mem_used / mem_total) * 100
                    if mem_util >= self.mem_util_threshold:
                        is_mem_util_over = True
                        max_mem_util = mem_util
                        max_mem_used = mem_used
                        self.proc.kill()
                        self.proc = None
                        raise RuntimeError(
                            f'[{self.__class__.__name__}] CUDA out of memory error - '
                            f'Memory util: {mem_util:.2f}% > threshold: 99% | '
                            f'{mem_used}MiB / {mem_total}MiB')
                    else:
                        if mem_util > max_mem_util:
                            max_mem_util = mem_util
                            max_mem_used = mem_used

            if is_timeout_error is True and is_mem_util_over is False:
                print(f'[WARNING][{self.__class__.__name__}] '
                      f'TimeoutExpired error might not caused by Out of Memory error | '
                      f'Max memory usage: {max_mem_used}MiB / {mem_total}MiB')

            print(f'[INFO][{self.__class__.__name__}] '
                  f'Max memory usage util: {max_mem_util:.2f}% | '
                  f'{max_mem_used}MiB / {mem_total}MiB')


class BaseMultiGPUMemoryProfileValidator(object):
    def __init__(self, profile_dir, dest_profile_dir, training_scheme, shell_script,
                 timeout=60, user_defined_cmd_fn=None, mem_util_threshold=100):
        if not os.path.exists(profile_dir):
            raise ValueError(f'Argument profile_dir must exist')
        if training_scheme not in ['static', 'adaptive']:
            raise ValueError(
                f'Argument training_schemes must be configured among [static, adaptive]')
        if not os.path.exists(shell_script):
            raise ValueError(f'Argument shell_script must exist')

        self.profile_dir = profile_dir
        self.dest_profile_dir = dest_profile_dir
        self.training_scheme = training_scheme
        self.shell_script = shell_script
        self.timeout = timeout
        self.user_defined_cmd_fn = user_defined_cmd_fn
        self.mem_util_threshold = mem_util_threshold

    def log(self, message, status='info'):
        print_msg = f'[{status.upper()}][{self.__class__.__name__}] {message}'
        print(print_msg)

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
            self.mock_adaptive_config_file = 'adaptive_config/adaptive_config_for_mem_profile_validation.json'
            write_json(self.mock_adaptive_config_file, mock_adaptive_config_data)
        else:
            raise ValueError(f'Not support such trainin scheme: {self.training_scheme}')

    @contextmanager
    def evaluate_handler(self):
        try:
            yield
        finally:
            if self.training_scheme == 'adaptive':
                self.log(f'Remove mock config file for adaptive training: {self.mock_adaptive_config_file}')
                os.system(f'rm -rf {self.mock_adaptive_config_file}')

    def evaluate(self):
        raise NotImplementedError


class MemoryProfileValidator(BaseMultiGPUMemoryProfileValidator):
    def __init__(self, profile_dir, dest_profile_dir, training_scheme, shell_script,
                 timeout=60, user_defined_cmd_fn=None, mem_util_threshold=100):
        super().__init__(profile_dir, dest_profile_dir, training_scheme, shell_script,
                         timeout, user_defined_cmd_fn, mem_util_threshold)
        try:
            if len(os.listdir(self.profile_dir)) >= 1:
                int(os.listdir(self.profile_dir)[0])
        except ValueError:
            raise ValueError(f"profile dir must contain local batch size sub-directory, "
                             f"but: {os.listdir(self.profile_dir)}")

        os.makedirs(dest_profile_dir)
        os.system(f'cp -r {self.profile_dir} {dest_profile_dir}')
        self.log('========================================================')
        self.log(f'[Initialize] Copy {self.profile_dir} to {dest_profile_dir}')
        self.log('========================================================')
        # NOTE: profile dir must contain one sub-directory of model name
        assert len(os.listdir(dest_profile_dir)) == 1
        model_name = os.listdir(dest_profile_dir)[0]
        self.dest_profile_dir = os.path.join(dest_profile_dir, model_name)

        self._prepare_for_training_scheme()

    def evaluate(self):
        with self.evaluate_handler():
            max_num_models_with_prev_lbs = -1
            for lbs in sorted(os.listdir(self.profile_dir), key=lambda x: int(x)):
                profiled_max_num_models = get_max_num_models_for_static_lbs(self.profile_dir, lbs)
                if max_num_models_with_prev_lbs < 0:
                    max_num_models = profiled_max_num_models
                else:
                    max_num_models = max_num_models_with_prev_lbs
                while True:
                    if max_num_models < 1:
                        break
                    try:
                        rank = 0
                        world_size = 1
                        local_batch_size = lbs
                        weight_sync_method = 'recommend'
                        if self.training_scheme == 'static':
                            accum_step = 1
                            if self.user_defined_cmd_fn is not None:
                                command = self.user_defined_cmd_fn(
                                    self.shell_script, rank, world_size, local_batch_size,
                                    max_num_models, accum_step, weight_sync_method
                                )
                            else:
                                command = [
                                    self.shell_script, rank, world_size, local_batch_size,
                                    max_num_models, accum_step, weight_sync_method
                                ]
                        elif self.training_scheme == 'adaptive':
                            accum_step = 2
                            if self.user_defined_cmd_fn is not None:
                                command = self.user_defined_cmd_fn(
                                    self.shell_script, rank, world_size, local_batch_size,
                                    max_num_models, accum_step, weight_sync_method,
                                    self.mock_adaptive_config_file
                                )
                            else:
                                command = [
                                    self.shell_script, rank, world_size, local_batch_size,
                                    max_num_models, accum_step, weight_sync_method,
                                    self.mock_adaptive_config_file
                                ]
                        with nvidia_smi_memory_monitoring(max_num_models, self.mem_util_threshold):
                            loop = asyncio.get_event_loop()
                            loop.run_until_complete(async_run_command(command, self.timeout))
                        self.log('Success to execute command')
                        self.log('Sleep 30 sec ..')
                        time.sleep(30)
                        break
                    except RuntimeError as e: # OOM happen
                        self.log(e)
                        kill_python_cmd = "kill -9 `ps | grep python | grep -v {0} | grep -v defunct | awk -F' ' '{{print $1}}'`".format(os.getpid())
                        os.system(kill_python_cmd)
                        kill_nvidiasmi_query_cmd = \
                            f"kill -9 `ps -ef | grep -v grep | grep \"nvidia-smi --query\" | awk '{{print $2}}' `"
                        os.system(kill_nvidiasmi_query_cmd)
                        self.log('Sleep 30 sec ..')
                        time.sleep(30)
                        max_num_models -= 1
                # Get real max num models
                self.log('========================================================')
                self.log(f'Profiled max number of VSWs ({lbs}): {profiled_max_num_models}')
                self.log(f'Real max number of VSWs ({lbs}): {max_num_models}')
                self.log('========================================================')
                # Change max num models if it is different
                if profiled_max_num_models != max_num_models:
                    if max_num_models == 0:
                        self.log(f'Remove profile data in {os.path.join(self.dest_profile_dir, lbs)}')
                        os.system(f'rm -rf {os.path.join(self.dest_profile_dir, lbs)}')
                    else:
                        self.log(f'Update profile data in {os.path.join(self.dest_profile_dir, lbs)}')
                        memory_profile_json_data = get_max_profile_json_data(self.dest_profile_dir, lbs)
                        memory_profile_json_data['max_num_models'] = max_num_models
                        set_max_profile_json_data(self.dest_profile_dir, lbs, memory_profile_json_data)
                # Record current profiled max number of VSWs for the next local batch size
                max_num_models_with_prev_lbs = max_num_models
                self.log('Sleep 30 sec ..')
                time.sleep(30)

        loop.close()


class StaticLocalBatchSizeMemoryProfileValidator(BaseMultiGPUMemoryProfileValidator):
    def __init__(self, profile_dir, dest_profile_dir, training_scheme, shell_script,
                 local_batch_size, timeout=60, user_defined_cmd_fn=None, mem_util_threshold=100):
        super().__init__(profile_dir, dest_profile_dir, training_scheme, shell_script,
                         timeout, user_defined_cmd_fn, mem_util_threshold)
        # Handle both case of '{profile dir}/{model}' and '{profile dir}/{model}/'
        if self.profile_dir[-1] == '/':
            self.profile_dir = self.profile_dir[:-1]
        # Check structure of profile dir is {profile dir}/{model}
        assert len(self.profile_dir.split('/')) == 2, \
            "[ERROR] Argument profile_dir must have {profile dir}/{model}, " \
            f"but profile_dir: {self.profile_dir} | " \
            f"len(profile_dir.split('/')): {len(self.profile_dir.split('/'))}"
        model_name = self.profile_dir.split('/')[-1]

        self.local_batch_size = str(local_batch_size)
        self.static_lbs_profile_dir = os.path.join(self.profile_dir, self.local_batch_size)
        # Check structure of self.static_lbs_profile_dir is {profile dir}/{model}/{lbs}
        if not (len(os.listdir(self.static_lbs_profile_dir)) == 1 and \
                os.listdir(self.static_lbs_profile_dir)[0] == socket.gethostname()):
            raise ValueError(f"profile dir must contain local batch size sub-directory, "
                             f"but: {os.listdir(self.static_lbs_profile_dir)}")

        # NOTE: profile dir must contain one sub-directory of model name: {dest_profile_dir}/{model}
        self.dest_profile_dir = os.path.join(dest_profile_dir, model_name)
        os.makedirs(self.dest_profile_dir, exist_ok=True)
        if os.path.exists(os.path.join(self.dest_profile_dir, self.local_batch_size)):
            raise ValueError(f'{os.path.join(self.dest_profile_dir, self.local_batch_size)} already exist')
        self.log('========================================================')
        self.log(f'[Initialize] Copy {self.static_lbs_profile_dir} to {self.dest_profile_dir}')
        self.log('========================================================')
        os.system(f'cp -r {self.static_lbs_profile_dir} {self.dest_profile_dir}')

        self._prepare_for_training_scheme()

    def evaluate(self):
        with self.evaluate_handler():
            profiled_max_num_models = get_max_num_models_for_static_lbs(self.profile_dir, self.local_batch_size)
            max_num_models = profiled_max_num_models
            while True:
                if max_num_models < 1:
                    break
                try:
                    rank = 0
                    world_size = 1
                    weight_sync_method = 'recommend'
                    if self.training_scheme == 'static':
                        accum_step = 1
                        if self.user_defined_cmd_fn is not None:
                            command = self.user_defined_cmd_fn(
                                self.shell_script, rank, world_size, self.local_batch_size,
                                max_num_models, accum_step, weight_sync_method
                            )
                        else:
                            command = [
                                self.shell_script, rank, world_size, self.local_batch_size,
                                max_num_models, accum_step, weight_sync_method
                            ]
                    elif self.training_scheme == 'adaptive':
                        accum_step = 2
                        if self.user_defined_cmd_fn is not None:
                            command = self.user_defined_cmd_fn(
                                self.shell_script, rank, world_size, self.local_batch_size,
                                max_num_models, accum_step, weight_sync_method,
                                self.mock_adaptive_config_file
                            )
                        else:
                            command = [
                                self.shell_script, rank, world_size, self.local_batch_size,
                                max_num_models, accum_step, weight_sync_method,
                                self.mock_adaptive_config_file
                            ]
                    with nvidia_smi_memory_monitoring(max_num_models, self.mem_util_threshold):
                        loop = asyncio.get_event_loop()
                        loop.run_until_complete(async_run_command(command, self.timeout))
                    self.log('Success to execute command')
                    self.log('Sleep 30 sec ..')
                    time.sleep(30)
                    break
                except RuntimeError as e: # OOM happen
                    self.log(e)
                    kill_python_cmd = "kill -9 `ps | grep python | grep -v {0} | grep -v defunct | awk -F' ' '{{print $1}}'`".format(os.getpid())
                    os.system(kill_python_cmd)
                    kill_nvidiasmi_query_cmd = \
                        f"kill -9 `ps -ef | grep -v grep | grep \"nvidia-smi --query\" | awk '{{print $2}}' `"
                    os.system(kill_nvidiasmi_query_cmd)
                    self.log('Sleep 30 sec ..')
                    time.sleep(30)
                    max_num_models -= 1
            # Get real max num models
            self.log('========================================================')
            self.log(f'Profiled max number of VSWs ({self.local_batch_size}): {profiled_max_num_models}')
            self.log(f'Real max number of VSWs ({self.local_batch_size}): {max_num_models}')
            self.log('========================================================')
            # change max num models if it is different
            if profiled_max_num_models != max_num_models:
                if max_num_models == 0:
                    self.log(f'Remove profile data in {os.path.join(self.dest_profile_dir, self.local_batch_size)}')
                    os.system(f'rm -rf {os.path.join(self.dest_profile_dir, self.local_batch_size)}')
                else:
                    self.log(f'Update profile data in {os.path.join(self.dest_profile_dir, self.local_batch_size)}')
                    memory_profile_json_data = get_max_profile_json_data(self.dest_profile_dir, self.local_batch_size)
                    memory_profile_json_data['max_num_models'] = max_num_models
                    set_max_profile_json_data(self.dest_profile_dir, self.local_batch_size, memory_profile_json_data)
            self.log('Sleep 30 sec ..')
            time.sleep(30)

        loop.close()