import os
import math
import warnings

import torch

from torch.iidp.utils.distributed import get_allgather_value, print_one_rank
from torch.iidp.utils.json_utils import read_json
from torch.iidp.config.model.throughput.throughput_model import ThroughputModel
from torch.iidp.cluster.resource import ResourceInfo, GlobalResourceInfo
from torch.iidp.profiler.profiler import MAX_MEM_PROFILE_FILE_NAME
from torch.iidp.config.config_utils import print_table, sorted_listdir, sorted_config_map_by_rank


class IIDPConfig(object):
    def __init__(self, lbs, num_models, accum_step, weight_sync_method):
        self.lbs = lbs # static
        self.num_models = num_models
        self.accum_step = accum_step
        self.weight_sync_method = weight_sync_method # static


class IIDPConfigurator(object):
    def __init__(self, comp_profile_dir, comm_profile_dir, bucket_profile_dir,
                 memory_profile_dir, local_config, global_server_info,
                 max_global_batch_size=-1, is_dynamic_local_batch_size=False, gpu=None):
        if not isinstance(max_global_batch_size, int) or max_global_batch_size < 0:
            raise TypeError(
                f'Argument ```max_global_batch_size``` must be positive integer, '
                f'but {max_global_batch_size} and type: {type(max_global_batch_size)}')
        self.max_global_batch_size = max_global_batch_size
        self.global_server_info = global_server_info
        self.is_dynamic_local_batch_size = is_dynamic_local_batch_size
        self.gpu = gpu
        self.total_num_gpus = self.global_server_info.total_num_gpus

        self.static_lbs = local_config.lbs if self.is_dynamic_local_batch_size is False else -1

        self.iidp_config_map_in_cluster, self.updated_iidp_config_map = {}, {}
        self._init_config_map_in_cluster(local_config)

        all_server_names = []
        for server_info in self.global_server_info:
            all_server_names.append(server_info.name)
        self.configurators = {}

        if len(sorted_listdir(comp_profile_dir)) != len(sorted_listdir(memory_profile_dir)):
            raise ValueError(
                f'[ERROR][{self.__class__.__name__}] '
                f'Computation and memory profile data for range of local batch size '
                f'must be equal, but comp: {sorted_listdir(comp_profile_dir)} | mem: {sorted_listdir(memory_profile_dir)}')
        # Create memory profile info (type: dict)
        memory_profile_info = {}
        for lbs in sorted_listdir(memory_profile_dir):
            if not lbs in memory_profile_info.keys():
                memory_profile_info[lbs] = {}
            static_lbs_mem_profile_dir = os.path.join(memory_profile_dir, lbs)
            static_lbs_comp_profile_dir = os.path.join(comp_profile_dir, lbs)
            # Check the same server profile data in computation and memory profile dir
            if os.listdir(static_lbs_comp_profile_dir) != os.listdir(static_lbs_mem_profile_dir):
                raise ValueError(
                    f'[ERROR] For static LBS of {lbs}, server profile data is not consistent among comp and memory profile dir!\n'
                    f'comp profile dir - {static_lbs_comp_profile_dir} : {os.listdir(static_lbs_comp_profile_dir)}\n'
                    f'memory profile dir - {static_lbs_mem_profile_dir} : {os.listdir(static_lbs_mem_profile_dir)}')
            for server_name in os.listdir(static_lbs_mem_profile_dir):
                max_memory_profile_file = os.path.join(
                    static_lbs_mem_profile_dir, server_name, MAX_MEM_PROFILE_FILE_NAME)
                memory_profile_json_data = read_json(max_memory_profile_file)
                memory_profile_info[lbs][memory_profile_json_data['gpu_type']] = memory_profile_json_data['max_num_models']

        # Instantiate configurator for static local batch size
        for lbs in sorted_listdir(comp_profile_dir):
            local_batch_size = int(lbs)
            if self.is_dynamic_local_batch_size is False and local_batch_size != local_config.lbs:
                continue
            static_lbs_comp_profile_dir = os.path.join(comp_profile_dir, lbs)
            # Check current local batch size can be supported by current global servers
            print_one_rank(f'[{self.__class__.__name__}] all_server_names: {all_server_names} | '
                    f'static_lbs_comp_profile_dir: {static_lbs_comp_profile_dir} | '
                    f'os.listdir(static_lbs_comp_profile_dir): {os.listdir(static_lbs_comp_profile_dir)}', 'debug')
            if not set(all_server_names).issubset(set(os.listdir(static_lbs_comp_profile_dir))):
                print_one_rank(
                    f'[{self.__class__.__name__}] local_batch_size: {local_batch_size} '
                    f'is not supported by current cluster: {self.global_server_info} '
                    f'==> skip it for IIDP configuration'
                )
                continue
            if max_global_batch_size//local_batch_size < self.total_num_gpus:
                print_one_rank(
                    f'[{self.__class__.__name__}] local_batch_size: {local_batch_size} '
                    f'is not satisfied with current total number of GPUs: {self.total_num_gpus} '
                    f'==> skip it for IIDP configuration'
                )
                continue
            static_lbs_memory_profile_info = memory_profile_info[lbs]
            max_num_workers = max_global_batch_size//local_batch_size+1
            self.configurators[local_batch_size] = IIDPStaticLocalBatchSizeConfigurator(
                static_lbs_comp_profile_dir, comm_profile_dir, bucket_profile_dir,
                static_lbs_memory_profile_info, local_batch_size,
                local_config.weight_sync_method, global_server_info, max_num_workers
            )
        if local_config.lbs not in self.configurators.keys():
            raise ValueError(
                f'No such profile computation data for local batch size: '
                f'{local_config.lbs}, but existing data: {self.configurators.keys()}'
            )

    def _init_config_map_in_cluster(self, local_config):
        all_num_models_in_process_group = {
            rank: value for rank, value in enumerate(get_allgather_value(local_config.num_models, self.gpu))
        }
        all_accum_step_in_process_group = {
            rank: value for rank, value in enumerate(get_allgather_value(local_config.accum_step, self.gpu))
        }
        for rank, (num_models, accum_step) in enumerate(
                zip(all_num_models_in_process_group.values(), all_accum_step_in_process_group.values())):
            self.iidp_config_map_in_cluster[rank] = (num_models, accum_step)
        #print(f'[DEBUG] self.iidp_config_map_in_cluster: {self.iidp_config_map_in_cluster}')

    def update(self):
        """
        Lazy update - After IIDP trainer actually updates all of states,
        configuration map will be updated by being called update()
        """
        if len(self.updated_iidp_config_map) == 0:
            raise ValueError(f'self.updated_iidp_config_map must not be empty, '
                             f'but {self.updated_iidp_config_map} => '
                             f'solve_placement() may return empty config map')
        self.iidp_config_map_in_cluster = self.updated_iidp_config_map

    def solve_placement(self, global_batch_size, current_global_batch_size):
        if self.is_dynamic_local_batch_size is False: # [EXPERIMENTAL] TODO: remove
            warnings.warn(
                "If dynamic local batch size is False, argument global_batch_size "
                "indicates total number of workers = (global batch size / local batch size)"
            )
            # NOTE: If dynamic local batch size is False, argument ```global_batch_size```
            # indicates total number of workers = (global batch size / local batch size)
            total_num_workers = global_batch_size
            new_iidp_config_map = {}
            _, _, solved_iidp_config_map = \
                self.configurators[self.static_lbs].solve_dynamic_programming(total_num_workers)
            if solved_iidp_config_map == {}:
                return new_iidp_config_map
            for rank in range(self.total_num_gpus):
                new_num_models = solved_iidp_config_map[rank][0] - self.iidp_config_map_in_cluster[rank][0]
                new_accum_step = solved_iidp_config_map[rank][1] - self.iidp_config_map_in_cluster[rank][1]
                if new_num_models == 0 and new_accum_step == 0:
                    continue
                new_iidp_config_map[rank] = (new_num_models, new_accum_step)
            if new_iidp_config_map:
                self.updated_iidp_config_map = solved_iidp_config_map
            return new_iidp_config_map

        new_iidp_config_map = {} # return value
        best_throughput = -1
        best_solved_iidp_config_map = {}
        best_local_batch_size = 0
        best_total_num_workers = 0
        new_global_batch_size = 0
        for local_batch_size, configurator in self.configurators.items():
            print_one_rank(
                f'[{self.__class__.__name__}] solve_placement() '
                f'global_batch_size: {global_batch_size} | local_batch_size: {local_batch_size} | '
                f'round(global_batch_size/local_batch_size): {round(global_batch_size/local_batch_size)}'
            )
            if global_batch_size > current_global_batch_size: # Increase global batch size
                total_num_workers = round(global_batch_size/local_batch_size)
                total_num_workers += (total_num_workers % 2) # SimiGrad constraint
                if local_batch_size*total_num_workers <= current_global_batch_size:
                    continue
            else:
                total_num_workers = round(global_batch_size/local_batch_size)
                total_num_workers -= (total_num_workers % 2) # SimiGrad constraint
                if local_batch_size*total_num_workers >= current_global_batch_size:
                    continue
            """ Ceil / Floor
            print_one_rank(
                f'[{self.__class__.__name__}] solve_placement() '
                f'global_batch_size: {global_batch_size} | local_batch_size: {local_batch_size} | '
                f'math.ceil(global_batch_size/local_batch_size): {math.ceil(global_batch_size/local_batch_size)}'
            )
            if is_increase_batch_size:
                total_num_workers = math.ceil(global_batch_size/local_batch_size)
                total_num_workers += (total_num_workers % 2) # SimiGrad constraint
            else:
                total_num_workers = math.floor(global_batch_size/local_batch_size)
                total_num_workers -= (total_num_workers % 2) # SimiGrad constraint
            """
            #print_one_rank(f'[{self.__class__.__name__}] solve_placement() '
            #               f'local_batch_size: {local_batch_size} | '
            #               f'total_num_workers: {total_num_workers}', 'debug')
            if total_num_workers < self.total_num_gpus:
                print_one_rank(f'[{self.__class__.__name__}] solve_placement() '
                               f'local_batch_size: {local_batch_size} | '
                               f'total_num_workers: {total_num_workers} | '
                               f'self.total_num_gpus: {self.total_num_gpus} ==> continue! ', 'debug')
                continue
            throughput, _, solved_iidp_config_map = configurator.solve_dynamic_programming(total_num_workers)
            print_one_rank(f'[{self.__class__.__name__}] solve_placement() '
                           f'local_batch_size: {local_batch_size} | '
                           f'total_num_workers: {total_num_workers} | '
                           f'throughput: {throughput} | '
                           f'solved_iidp_config_map: {solved_iidp_config_map}', 'debug')
            if solved_iidp_config_map == {}:
                continue
            if throughput > best_throughput:
                best_local_batch_size = local_batch_size
                best_total_num_workers = total_num_workers
                best_throughput = throughput
                best_solved_iidp_config_map = solved_iidp_config_map
                new_global_batch_size = best_local_batch_size * best_total_num_workers

        if best_solved_iidp_config_map == {}:
            return new_iidp_config_map, best_local_batch_size, new_global_batch_size
        print_one_rank('==================================================================')
        print_one_rank(
            f'[{self.__class__.__name__}] solve_placement()  ** best config ** | '
            f'local_batch_size: {best_local_batch_size} | '
            f'total_num_workers: {best_total_num_workers} | '
            f'throughput: {best_throughput:.2f} | '
            f'solved_iidp_config_map: {best_solved_iidp_config_map}', 'info')
        print_one_rank('==================================================================')
        try:
            for rank in range(self.total_num_gpus):
                new_num_models = best_solved_iidp_config_map[rank][0] - self.iidp_config_map_in_cluster[rank][0]
                new_accum_step = best_solved_iidp_config_map[rank][1] - self.iidp_config_map_in_cluster[rank][1]
                if new_num_models == 0 and new_accum_step == 0:
                    continue
                new_iidp_config_map[rank] = (new_num_models, new_accum_step)
        except Exception as e:
            print_one_rank(f'[{self.__class__.__name__}] solve_placement() | rank: {rank} | '
                            f'best_solved_iidp_config_map: {best_solved_iidp_config_map} | '
                            f'self.iidp_config_map_in_cluster: {self.iidp_config_map_in_cluster}', 'debug')
            raise e
        if new_iidp_config_map:
            self.updated_iidp_config_map = best_solved_iidp_config_map
        return new_iidp_config_map, best_local_batch_size, new_global_batch_size


class IIDPStaticLocalBatchSizeConfigurator(object):
    def __init__(self, comp_profile_dir, comm_profile_dir, bucket_profile_dir,
                 memory_profile_info, local_batch_size, weight_sync_method, global_server_info, max_num_workers=-1):
        self.comp_profile_dir = comp_profile_dir
        self.comm_profile_dir = comm_profile_dir
        self.bucket_profile_dir = bucket_profile_dir
        self.global_server_info = global_server_info
        self.total_num_gpus = self.global_server_info.total_num_gpus
        self.weight_sync_method = weight_sync_method
        self.local_batch_size = local_batch_size
        self.throughput_models = {}
        self.all_max_num_local_models_in_process_group = memory_profile_info
        self.max_num_workers = max_num_workers

        self._build_throughput_model()

        self._init_dp_solver()

    def _build_throughput_model(self):
        """
        Assumption: Profile data of all servers must be placed on 'comp_profile_dir/{server name}'
        """
        for server_info in self.global_server_info:
            local_comp_profile_dir = os.path.join(self.comp_profile_dir, server_info.name)
            if server_info.name not in self.throughput_models.keys():
                self.throughput_models[server_info.name] = \
                    ThroughputModel(local_comp_profile_dir, self.comm_profile_dir, self.bucket_profile_dir)

    def _init_dp_solver(self):
        self.dp_solver = DynamicProgrammingSolver(
            self.local_batch_size,
            self.weight_sync_method,
            self.throughput_models,
            self.all_max_num_local_models_in_process_group,
            self.global_server_info,
            self.max_num_workers
        )

    def estimate_time(self, server_name, num_models, accum_step,
                      resource_info: ResourceInfo, global_resource_info: GlobalResourceInfo):
        """Estimate local server iteration time"""
        iter_time, _ = self.throughput_models[server_name].evaluate(
            num_models, accum_step, self.weight_sync_method, resource_info, global_resource_info)
        return iter_time, _

    def estimate_throughput(self, total_num_models: int, iter_time: float):
        global_batch_size = self.local_batch_size * total_num_models
        thp = global_batch_size / iter_time
        return thp

    def update(self):
        """
        Lazy update - After IIDP trainer actually updates all of states,
        configuration map will be updated by being called update()
        """
        self.iidp_config_map_in_cluster = self.updated_iidp_config_map

    def solve_dynamic_programming(self, total_num_workers):
        """
        Args:
            total_num_workers (int): Total number of virtual workers to configure GBS

        Returns:
            List: [throughput: float, iteration time: float, configuration map: dict - {rank: (num_models, accum_step)}]
        """
        throughput, iter_time, new_config_map = -1, -1, {}
        try:
            throughput, iter_time, _, new_config_set = self.dp_solver.solve(total_num_workers)
        except: # No solution
            return [throughput, iter_time, new_config_map]
        #print_one_rank(f'[DEBUG] configurator - new_config_set by DP solver: {new_config_set}')
        new_config_map = sorted_config_map_by_rank(
                            self.dp_solver.generate_config_map(new_config_set))
        return [throughput, iter_time, new_config_map]


class DynamicProgrammingSolver(object):
    def __init__(self, local_batch_size, weight_sync_method, throughput_models,
                 all_max_num_models_info, global_server_info, max_num_workers=-1):
        self.local_batch_size = local_batch_size
        self.weight_sync_method = weight_sync_method
        self.throughput_models = throughput_models
        # NOTE: all_max_num_local_models_in_process_group = {'device name (str)': max number of VSWs (int)}
        self.all_max_num_local_models_in_process_group = all_max_num_models_info
        self.global_server_info = global_server_info
        self.total_num_gpus = self.global_server_info.global_resource_info.total_num_gpus
        self.max_num_workers = max_num_workers
        self.A = self.create_table(max_num_workers)

    def _split_rank_for_simigrad(self, arr, size):
        arrays = []
        while len(arr) > size:
            pice = arr[:size]
            arrays.append(pice)
            arr   = arr[size:]
        arrays.append(arr)
        return arrays

    def _generate_config_name(self, server_name, ranks, num_models, accum_step):
        """e.g, server1:0,VSW:3,GA:1 -> server1: ranks: [0, 1] (VSW, GA) = (3,1)"""
        return server_name+':'+str(ranks[0])+','+'VSW:'+str(num_models)+','+'GA:'+str(accum_step)

    def generate_config_map(self, config_set: list):
        config_map = {} # {rank: (num_models, accum_step)}
        for config_name in config_set:
            region_str, num_models_str, accum_step_str = config_name.split(',')
            head_rank = self.convert_config_str_to_int(region_str)
            num_models = self.convert_config_str_to_int(num_models_str)
            accum_step = self.convert_config_str_to_int(accum_step_str)
            # For SimiGrad
            config_map[head_rank] = (num_models, accum_step)
            config_map[head_rank+1] = (num_models, accum_step)
        return config_map

    def estimate_throughput(self, total_num_models: int, iter_time: float):
        global_batch_size = self.local_batch_size * total_num_models
        thp = global_batch_size / iter_time
        return thp

    def get_pruned_table_by_hash(self, A):
        """
            Table element: [iter time, number of workers(= number of VSWs * GA), config_name]
            NOTE: Important assumption - sort by iteration time in increasing order
            Prune by Hashing => The first unique config has the fastest iter time
        """
        """
        print('========= Before pruning ==============')
        print_table(A, len(A))
        print('=======================================')
        """
        pruned_A = []
        pruned_A_hashmap = {}
        for A_elem in A:
            _, num_worker, config_name = A_elem
            if num_worker == 1:
                pruned_A.append(A_elem)
            else:
                config_region = config_name.split(',')[0]
                hash_key = str(num_worker) + config_region
                if hash_key not in pruned_A_hashmap:
                    pruned_A_hashmap[hash_key] = A_elem
        pruned_A.extend(pruned_A_hashmap.values())
        """
        print('========= After pruning ==============')
        print(f'Number of search space '
            f'before pruning: {len(A)} | '
            f'after pruning: {len(pruned_A)}')
        print_table(pruned_A, len(pruned_A))
        """
        return pruned_A

    def get_pruned_table(self, A):
        pruned_A = []
        #print('========> Before pruning ==============')
        #print_table(A, len(A))

        for A_elem in A:
            iter_time, num_worker, config_name = A_elem
            if num_worker == 1:
                pruned_A.append(A_elem)
            else:
                remove_idx = -1
                is_append = True
                # Search the candidate of filtering
                for idx, pruned_data in enumerate(pruned_A):
                    config_region = config_name.split(',')[0]
                    # Pruning space: Same number of virtual workers within the same regions (2-GPU unit allocation)
                    if config_region in pruned_data[2] and num_worker == pruned_data[1]:
                        # 1-1) Add a new data with faster time among the same processing configuration
                        if iter_time < pruned_data[0]:
                            remove_idx = idx
                            pruned_A.append(A_elem)
                            break
                        # 2) Not add data with slower time
                        else:
                            is_append = False

                if is_append:
                    if remove_idx == -1: # 3) Add a new data
                        pruned_A.append(A_elem)
                    else: # 1-2) Remove data for previous slower configuration
                        pruned_A.remove(pruned_A[remove_idx])
        """
        print('========> After pruning ==============')
        print(f'Number of search space '
            f'before pruning: {len(A)} | '
            f'after pruning: {len(pruned_A)}')
        print_table(pruned_A, len(pruned_A))
        """
        return pruned_A

    def create_table(self, max_num_workers=-1):
        """Table: [iter time, number of workers(= number of VSWs * GA), config_name]"""
        if max_num_workers < self.total_num_gpus:
            raise ValueError(f"Argument max_num_workers: {max_num_workers} < self.total_num_gpus: {self.total_num_gpus}")
        #print(f'[DEBUG] create_table() - max_num_workers: {max_num_workers}')
        A = []
        MAX_GA_STEPS = 1000
        #print(f'[DEBUG] MAX_GA_STEPS: {MAX_GA_STEPS}')
        for server_info in self.global_server_info:
            server_name = server_info.name
            gpu_type = server_info.resource_info.device_name
            split_ranks = self._split_rank_for_simigrad(server_info.ranks, 2)
            for i, ranks in enumerate(split_ranks):
                if os.getenv("EASYSCALE") == "1" or os.getenv("SIMIGRAD") == "1":
                    max_num_models = 1
                else:
                    try:
                        max_num_models = self.all_max_num_local_models_in_process_group[gpu_type]
                    except Exception as e:
                        print(f'self.all_max_num_local_models_in_process_group: {self.all_max_num_local_models_in_process_group}')
                        print(f'self.global_server_info: {self.global_server_info}')
                        raise e

                if max_num_workers > 0:
                    min_running_workers = self.total_num_gpus-2 # -2 means excluding current 2-GPUs allocation
                    pruned_max_num_models = max(min((max_num_workers-min_running_workers)//2, max_num_models), 1)
                    assert pruned_max_num_models >= 1
                    max_num_models = pruned_max_num_models
                for num_models in range(1, max_num_models+1):
                    if max_num_workers > 0:
                        pruned_max_accum_step = min(MAX_GA_STEPS, ((max_num_workers-min_running_workers)//2//num_models))
                        assert pruned_max_accum_step > 0, f"{pruned_max_accum_step} | {num_models}"
                        max_accum_step = pruned_max_accum_step
                    else:
                        max_accum_step = MAX_GA_STEPS
                    """
                    print_one_rank(f'[DEBUG] =============> create_table() | '
                                   f'server_name: {server_name} | '
                                   f'gpu_type: {gpu_type} | '
                                   f'max_num_models: {max_num_models} | '
                                   f'max_accum_step: {max_accum_step}')
                    """
                    for accum_step in range(1, max_accum_step+1):
                        iter_time, _ = self.throughput_models[server_name].evaluate(
                            num_models, accum_step, self.weight_sync_method,
                            server_info.resource_info, self.global_server_info.global_resource_info
                        )
                        config_name = self._generate_config_name(server_name, ranks, num_models, accum_step)
                        A.append([iter_time, num_models * accum_step, config_name])
        #print_one_rank(f'[DEBUG] create_table() - A: {A}')
        #print_table(A, len(A))
        # NOTE: As at least one worker must exists on every GPUs, one worker must be put ahead of table
        A.sort(key=lambda x: x[1])
        A_with_one_worker = A[:self.total_num_gpus//2]
        A_over_one_worker = A[self.total_num_gpus//2:]
        # NOTE: Important assumption - sort by iteration time in increasing order
        A_with_one_worker.sort(key=lambda x: x[0])
        A_over_one_worker.sort(key=lambda x: x[0])
        #print_table(A_with_one_worker, len(A_with_one_worker))
        #print_table(A_over_one_worker, len(A_over_one_worker))
        A = A_with_one_worker + A_over_one_worker
        #print_table(A, len(A))
        # ==== case that is NOT required to allocate all of GPUs in global server info =======
        # NOTE: Important assumption - sort by iteration time in increasing order
        #A.sort(key=lambda x: x[0])
        # ====================================================================================
        return self.get_pruned_table_by_hash(A)

    def convert_config_str_to_int(self, config_str):
        return int(config_str.split(':')[-1])

    def _combine_same_region_config(self, prev_candidate_config_set: list, curr_config_name: str):
        if not isinstance(prev_candidate_config_set, list):
            raise TypeError(f"prev_candidate_config_set must be list type, but {type(prev_candidate_config_set)}")
        #print(f'[DEBUG] _combine_same_region_config() - prev_candidate_config_set: {prev_candidate_config_set} | curr_config_name: {curr_config_name}')
        curr_region, curr_num_models_str, curr_accum_step_str = curr_config_name.split(',')
        curr_num_models = self.convert_config_str_to_int(curr_num_models_str)
        curr_accum_step = self.convert_config_str_to_int(curr_accum_step_str)
        new_config_set = []
        for config_name in prev_candidate_config_set:
            prev_region = config_name.split(',')[0]
            if prev_region != curr_region:
                new_config_set.append(config_name)
        new_curr_config_name = ','.join([curr_region, 'VSW:' +str(curr_num_models), 'GA:'+str(curr_accum_step)])
        new_config_set.append(new_curr_config_name)
        new_config_set.sort()
        return new_config_set

    def _get_curr_config_set(self, prev_candidate_config_set: list, curr_config_name: str) -> list:
        return self._combine_same_region_config(prev_candidate_config_set, curr_config_name)

    def _get_curr_num_workers(self, config_set: list):
        total_num_models = 0
        for config_name in config_set:
            _, curr_num_models_str, curr_accum_step_str = config_name.split(',')
            curr_num_models = self.convert_config_str_to_int(curr_num_models_str)
            curr_accum_step = self.convert_config_str_to_int(curr_accum_step_str)
            total_num_models += (curr_num_models*curr_accum_step)
        return total_num_models

    def solve(self, total_num_workers):
        PRINT_DEBUG = False
        #print(f'[INFO] solve() - total_num_workers: {total_num_workers}')
        #print_table(self.A, len(self.A))
        """
        NOTE: [Important] With SimiGrad, number of workers must be double because one configuration assumes allocation of 2 GPUs
        A[i][0] = iter time, A[i][1] = number of (virtual) workers, A[i][2] = config name on one allocation
        [Dynamic Programming] Table for DP
        each element has [iter_time, config_name]
        col: Candidate ranks to be assigned new virtual workers
        row: Candidate number of virtual workers to be assigned
        -----------------------------------------------
                            |                real idx (number of virtual workers)                |
        ____________________|                               0                                    | 1(2) | 2(4) | 3(6) ..
        server1:0,VSW:1,GA:1| [throughput, iter time, number of workers, config set: List]   ..  |
        server1:2,VSW:1,GA:1|                                                                    |
        server2:4,VSW:1,GA:1|
        server2:6,VSW:1,GA:1|
        -----------------------------------------------
        'config_name' is a unit of 2 GPUs assignment
            e.g, config_name = server1:0,VSW:3,GA:1 ==> server1:[0,1] -> (VSW, GA) = (3,1)
        DP element: [throughput, iter time, number of workers, config set]
        DP[i][j][0] = throughput, DP[i][j][1] = iter_time, DP[i][j][2] = number of (virtual) workers, DP[i][j][3] = [config_name, ..]
        """
        if len(self.A) == 0:
            print('[INFO] No table for Dynamic Programming')
            return
        dp_row = total_num_workers//2+1 # Purpose of +1 is that the row index of DP table indicates the number of current workers
        dp_col = len(self.A)
        if PRINT_DEBUG is True:
            print_one_rank('============ DP table ============')
            print_one_rank(f'dp_col: {dp_col} | dp_row: {dp_row}')
            print_table(self.A, dp_col)
            print_one_rank('==================================')

        # DP element: [throughput, iter time, number of workers, config set]
        dp_elem = [0, 0, 0, []]
        DP = [[dp_elem for _ in range(dp_row)] for _ in range(dp_col)]
        # Initialize table for DP
        for j in range(1, dp_row):
            if self.A[0][1] <= j:
                iter_time = self.A[0][0]
                num_workers = self.A[0][1]
                config_name = [self.A[0][2]]
                thp = self.estimate_throughput(num_workers*2, iter_time)
                DP[0][j] = [thp, iter_time, num_workers, config_name]

        # [ Main algorithm ]
        # NOTE: Assumption: A - sorted by iteration time in increasing order (reference: create_table())
        # previous configuration with prev_max_workers has optimal sub-structure => DP[i-1][prev_max_workers]
        prev_max_workers = 1
        for i in range(1, dp_col): # i: Candidate configuration of GPU allocation
            if self.A[i][1] > dp_row: # Number of workers in a new configuration (self.A[i][1]) is over than the required total number of workers (dp_row)
                #break
                continue
            # Update current DP table to previous optimal configuration (<=prev_max_workers)
            if i == 1:
                DP[i][prev_max_workers] = DP[i-1][prev_max_workers]
                prev_max_workers+=1
            for j in range(1, prev_max_workers):
                DP[i][j] = DP[i-1][j]
            # Traverse right direction (toward increasing number of workers)
            for j in range(prev_max_workers, dp_row): # j: Candidate number of (virtual) workers
                curr_config_thp, prev_config_thp = 0, 0
                # [ Main logic for DP ] - combine previous set with a new configuration and compute objective value (throughput)
                curr_config_set = self._get_curr_config_set(DP[i-1][j][3], self.A[i][2])
                curr_max_iter_time = max(self.A[i][0], DP[i-1][j][1])
                curr_num_workers = self._get_curr_num_workers(curr_config_set)
                curr_config_thp = self.estimate_throughput(curr_num_workers*2, curr_max_iter_time)
                """
                print(f'[DEBUG] ===========> i: {i} | j: {j} | new config name: {self.A[i][2]} | '
                        f'curr config set: {curr_config_set} | '
                        f'curr_max_iter_time: {curr_max_iter_time} | '
                        f'curr num workers: {curr_num_workers} | '
                        f'curr_config_thp: {curr_config_thp}')
                """
                prev_config_thp = DP[i-1][j][0]
                #print(f'[DEBUG] curr thp: {curr_config_thp} | prev config set: {DP[i-1][j][3]} = prev thp: {prev_config_thp}')
                # [ Main logic for DP ] - compare objective value (throughput) in previous optimal sub-problem
                if (curr_config_thp > prev_config_thp and curr_num_workers < j) or curr_num_workers == j:
                    DP[i][j] = [curr_config_thp, curr_max_iter_time, curr_num_workers, curr_config_set]
                    #print(f'[DEBUG] i: {i} | j: {j} | current new config update DP[i][j] = {DP[i][j]}')
                    if curr_num_workers == j:
                        prev_max_workers = DP[i][j][2] + 1
                        for k in range(j+1, dp_row):
                            DP[i][k] = DP[i][j]
                        break
                else:
                    DP[i][j] = DP[i-1][j]
            if PRINT_DEBUG is True:
                print(f'************************** i: {i} *******************************')
                print(f'************************** config: {self.A[i][-1]} *******************************')
                for k in range(dp_row):
                    print(f'j:{k} - {DP[i][k][-1]} | {DP[i][k][2]}') # print configuration
                print('******************************************************************')

        if PRINT_DEBUG is True:
            print('[DEBUG] solve() ============================= Final DP table =============================')
            print(f'[DEBUG] i: {i} | dp_col: {dp_col} | dp_row: {dp_row}')
            print_one_rank('============ DP table ============')
            print_one_rank(f'dp_col: {dp_col} | dp_row: {dp_row}')
            print_table(self.A, dp_col)
            print_one_rank('==================================')

        solution = None
        for s in range(dp_col, 0, -1):
            is_total_num_workers_required = (DP[s-1][dp_row-1][2] == dp_row-1)
            is_num_gpu_allocation_required = (len(DP[s-1][dp_row-1][-1]) == self.total_num_gpus//2)
            if is_total_num_workers_required and is_num_gpu_allocation_required:
                solution = DP[s-1][dp_row-1]
                break
        if solution is None:
            raise AssertionError(f'[ERROR] DP Solution for total_num_workers: {total_num_workers} does not exist')
        return solution


class IIDPFutureConfigurator(object):
    def __init__(self, comp_profile_dir, comm_profile_dir, bucket_profile_dir,
                 memory_profile_dir, local_config, candidate_global_server_infos,
                 max_global_batch_size=-1, is_dynamic_local_batch_size=False,
                 gpu=None, utility_type='memory'):
        self.comp_profile_dir = comp_profile_dir
        self.comm_profile_dir = comm_profile_dir
        self.bucket_profile_dir = bucket_profile_dir

        if len(sorted_listdir(comp_profile_dir)) != len(sorted_listdir(memory_profile_dir)):
            raise ValueError(
                f'[ERROR][{self.__class__.__name__}] '
                f'Computation and memory profile data for range of local batch size '
                f'must be equal, but comp: {sorted_listdir(comp_profile_dir)} | mem: {sorted_listdir(memory_profile_dir)}')
        # Create memory profile info (type: dict)
        self.memory_profile_info = {}
        for lbs in sorted_listdir(memory_profile_dir):
            if not lbs in self.memory_profile_info.keys():
                self.memory_profile_info[lbs] = {}
            static_lbs_mem_profile_dir = os.path.join(memory_profile_dir, lbs)
            # Check the same server profile data in computation and memory profile dir
            if os.listdir(os.path.join(self.comp_profile_dir, lbs)) != os.listdir(static_lbs_mem_profile_dir):
                raise ValueError(
                    f'[ERROR] For static LBS of {lbs}, server profile data is not consistent among comp and memory profile dir!\n'
                    f'comp profile dir - {os.path.join(self.comp_profile_dir, lbs)} : {os.listdir(os.path.join(self.comp_profile_dir, lbs))}\n'
                    f'memory profile dir - {static_lbs_mem_profile_dir} : {os.listdir(static_lbs_mem_profile_dir)}')
            for server_name in os.listdir(static_lbs_mem_profile_dir):
                max_memory_profile_file = os.path.join(
                    static_lbs_mem_profile_dir, server_name, MAX_MEM_PROFILE_FILE_NAME)
                memory_profile_json_data = read_json(max_memory_profile_file)
                self.memory_profile_info[lbs][memory_profile_json_data['gpu_type']] = memory_profile_json_data['max_num_models']

        self.local_config = local_config
        if not isinstance(max_global_batch_size, int) or max_global_batch_size < 0:
            raise TypeError(
                f'Argument ```max_global_batch_size``` must be positive integer, '
                f'but {max_global_batch_size} and type: {type(max_global_batch_size)}')
        self.max_global_batch_size = max_global_batch_size
        # NOTE: candidate global server info is defined by list type in [torch/iidp/cluster/cluster_manager.py]
        if not isinstance(candidate_global_server_infos, list):
            candidate_global_server_infos = [candidate_global_server_infos]
        self.candidate_global_server_infos = candidate_global_server_infos
        self.is_dynamic_local_batch_size = is_dynamic_local_batch_size
        self.gpu = gpu

        self.static_lbs = local_config.lbs if self.is_dynamic_local_batch_size is False else -1
        # [Set of IIDPConfigurator for all global server info]
        # built once by prepare() at the initial phase of elastic training
        self.all_candidate_server_configurators = {}
        # [IIDPConfigurator for each global server info]
        # IIDPConfigurator => set of IIDPStaticLocalBatchSizeConfigurator
        # updated by update()
        # used in estimate_time_and_utility()
        self.configurators = {}
        self.utility_type = utility_type

        self._prepared = False
        self.current_global_batch_size = 0
        self.current_local_batch_size = 0
        self._update_lock = False

    def state_dict(self):
        return self.all_candidate_server_configurators

    def prepare(self, verbose=True):
        if len(self.all_candidate_server_configurators) == 0:
            self._init_configurators(verbose)
        self._prepared = True

    def _init_configurators(self, verbose=True):
        print_one_rank(
            '==============================================================\n'
            f'[{self.__class__.__name__}] Start to initialize configurators for candidate servers\n'
            f'[{self.__class__.__name__}] Total number of candidate severs to build: {len(self.candidate_global_server_infos)}\n'
            f'[{self.__class__.__name__}] verbose: {verbose}\n'
            f'[{self.__class__.__name__}] It might take time ..\n'
            '=============================================================='
        )
        for server_id, global_server_info in enumerate(self.candidate_global_server_infos):
            total_num_gpus = global_server_info.total_num_gpus
            self.all_candidate_server_configurators[server_id] = {}
            configurators = {}
            all_server_names = []
            for server_info in global_server_info:
                all_server_names.append(server_info.name)
            # Instantiate configurator for static local batch size
            for lbs in sorted_listdir(self.comp_profile_dir):
                try:
                    local_batch_size = int(lbs)
                except:
                    print_one_rank(
                        f'[{self.__class__.__name__}] init_configurators() '
                        f'Computation profile dir structure is not suitable for local batch size: '
                        f'{sorted_listdir(self.comp_profile_dir)}', 'error')
                    exit(1)
                if self.is_dynamic_local_batch_size is False and local_batch_size != self.local_config.lbs:
                    if verbose is True:
                        print_one_rank(
                            f'[{self.__class__.__name__}] init_configurators() '
                            f'self.is_dynamic_local_batch_size: {self.is_dynamic_local_batch_size} | '
                            f'self.local_config.lbs: {self.local_config.lbs} | '
                            f'local_batch_size: {local_batch_size} ===> continue!!', 'debug')
                    continue
                static_lbs_comp_profile_dir = os.path.join(self.comp_profile_dir, lbs)
                # Check current local batch size can be supported by current global servers
                #print_one_rank(f'[{self.__class__.__name__}] all_server_names: {all_server_names} | '
                #        f'static_lbs_comp_profile_dir: {static_lbs_comp_profile_dir} | '
                #        f'os.listdir(static_lbs_comp_profile_dir): {os.listdir(static_lbs_comp_profile_dir)}', 'debug')
                if not set(all_server_names).issubset(set(os.listdir(static_lbs_comp_profile_dir))):
                    if verbose is True:
                        print_one_rank(
                            f'[{self.__class__.__name__}] init_configurators() '
                            f'local_batch_size: {local_batch_size} is not supported '
                            f'by current cluster: {global_server_info} ==> skip it for IIDP configuration'
                        )
                    continue
                if self.max_global_batch_size//local_batch_size < total_num_gpus:
                    if verbose is True:
                        print_one_rank(
                            f'[{self.__class__.__name__}] init_configurators() '
                            f'local_batch_size: {local_batch_size} '
                            f'is not satisfied with current total number of GPUs: {total_num_gpus} '
                            f'==> skip it for IIDP configuration'
                        )
                    continue
                static_lbs_memory_profile_info = self.memory_profile_info[lbs]
                max_num_workers = self.max_global_batch_size//local_batch_size+1
                configurators[local_batch_size] = IIDPStaticLocalBatchSizeConfigurator(
                    static_lbs_comp_profile_dir, self.comm_profile_dir, self.bucket_profile_dir,
                    static_lbs_memory_profile_info, local_batch_size,
                    self.local_config.weight_sync_method, global_server_info, max_num_workers
                )
            self.all_candidate_server_configurators[server_id] = configurators

            if verbose is True:
                final_result_log_str = \
                    f'[{server_id} / {len(self.candidate_global_server_infos)}] ' \
                    f'server id: {server_id} | ' \
                    f'all_server_names: {all_server_names} | ' \
                    f'total number of GPUs: {global_server_info.total_num_gpus}'
                length = len(final_result_log_str) + 1
                print_one_rank('=' * length)
                print_one_rank(final_result_log_str)
                print_one_rank('=' * length)

        # NOTE: Check if at least one of the candidate servers can support the initial local batch size
        is_support_initial_lbs = False
        for server_id, configurators in self.all_candidate_server_configurators.items():
            if self.local_config.lbs in configurators.keys():
                is_support_initial_lbs = True
        if is_support_initial_lbs is False:
            raise ValueError(
                f'No candidate server to support such initial local batch size: {self.local_config.lbs}'
            )
        print_one_rank(
            '==============================================================\n'
            f'[{self.__class__.__name__}] Finish to initialize configurators for candidate servers\n'
            '=============================================================='
        )

    def update(self, server_id, local_batch_size, global_batch_size):
        if self._prepared is False:
            raise RuntimeError(
                f'[ERROR][{self.__class__.__name__}] update() must be called '
                f'after prepare() is called'
            )
        # Update current local & global batch size -> must be preserved for next epoch
        self.current_global_batch_size = global_batch_size
        self.current_local_batch_size = local_batch_size
        # Update configurators for each static local batch size with global resource
        self.configurators = self.all_candidate_server_configurators[server_id]
        self._update_lock = True

    def estimate_time_and_utility(self, global_batch_size, iteration, remaining_num_dataset):
        if self._update_lock is False:
            raise RuntimeError(
                f'[ERROR][{self.__class__.__name__}] estimate_time_and_utility() must be called '
                f'after update() is called'
            )
        min_duration, ret_utility, ret_solved_iidp_config_map, \
            ret_expected_gbs, ret_expected_step = math.inf, -1, {}, global_batch_size, 0
        best_local_batch_size = self.current_local_batch_size # for debugging
        for local_batch_size, configurator in self.configurators.items():
            # NOTE: current LBS and GBS must be preserved for next epoch
            if global_batch_size == self.current_global_batch_size:
                if self.current_local_batch_size != local_batch_size:
                    continue
                total_num_workers = global_batch_size // self.current_local_batch_size
                if total_num_workers >= configurator.total_num_gpus and iteration > 0:
                    _, iter_time, solved_iidp_config_map = configurator.solve_dynamic_programming(total_num_workers)
                    if solved_iidp_config_map == {}: # Candidate global server cannot support current global batch size
                        return min_duration, ret_utility, ret_solved_iidp_config_map, ret_expected_gbs, ret_expected_step
                    #print_one_rank(f'[{self.__class__.__name__}] estimate_time_and_utility - solved_iidp_config_map: {solved_iidp_config_map}', 'debug')
                    if remaining_num_dataset - global_batch_size*iteration < 0:
                        expected_step = (remaining_num_dataset // global_batch_size) + 1
                    else:
                        expected_step = iteration
                    if expected_step <= 0:
                        return min_duration, ret_utility, ret_solved_iidp_config_map, ret_expected_gbs, ret_expected_step
                    duration = iter_time * expected_step
                    utility = self.utility_func(local_batch_size, solved_iidp_config_map)
                    return duration, utility, solved_iidp_config_map, global_batch_size, expected_step
                else:
                    return min_duration, ret_utility, ret_solved_iidp_config_map, ret_expected_gbs, ret_expected_step
            else:
                if global_batch_size > self.current_global_batch_size: # Increasing global batch size
                    total_num_workers = round(global_batch_size/local_batch_size)
                    total_num_workers += (total_num_workers % 2) # SimiGrad constraint
                    if local_batch_size*total_num_workers < self.current_global_batch_size:
                        continue
                else:
                    total_num_workers = round(global_batch_size/local_batch_size)
                    total_num_workers -= (total_num_workers % 2) # SimiGrad constraint
                    if local_batch_size*total_num_workers > self.current_global_batch_size:
                        continue
                if total_num_workers >= configurator.total_num_gpus and iteration > 0:
                    _, iter_time, solved_iidp_config_map = configurator.solve_dynamic_programming(total_num_workers)
                    if solved_iidp_config_map == {}:
                        continue
                    expected_gbs = total_num_workers * local_batch_size
                    if remaining_num_dataset - expected_gbs*iteration < 0:
                        expected_step = (remaining_num_dataset // expected_gbs) + 1
                    else:
                        expected_step = iteration
                    if expected_step <= 0:
                        continue
                    duration = iter_time * expected_step
                    utility = self.utility_func(local_batch_size, solved_iidp_config_map)
                    if duration < min_duration:
                        min_duration = duration
                        ret_utility = utility
                        ret_solved_iidp_config_map = solved_iidp_config_map
                        ret_expected_gbs = expected_gbs
                        ret_expected_step = expected_step
                        best_local_batch_size = local_batch_size
        #print_one_rank(f'[{self.__class__.__name__}] best local batch size: {best_local_batch_size}', 'debug')
        return min_duration, ret_utility, ret_solved_iidp_config_map, \
                ret_expected_gbs, ret_expected_step

    def utility_func(self, local_batch_size, new_config_map):
        if self.utility_type == 'memory': # TODO: If model is GA-effective, memory util is low
            gpu_util_in_cluster = []
            #print_one_rank(f'utility_func() - self.global_server_info: {self.global_server_info}', 'debug')
            for rank, (num_models, _) in new_config_map.items():
                server_info = self.configurators[local_batch_size].global_server_info.rank_to_server_map[rank]
                gpu_type = server_info.resource_info.device_name
                max_num_models = self.configurators[local_batch_size].all_max_num_local_models_in_process_group[gpu_type]
                gpu_util_in_cluster.append(num_models/max_num_models)
            return round(sum(gpu_util_in_cluster) / len(gpu_util_in_cluster), 2)
