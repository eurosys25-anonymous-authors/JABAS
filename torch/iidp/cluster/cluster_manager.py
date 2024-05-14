import socket
import os
import itertools

import torch

from torch.iidp.utils.json_utils import read_json
from torch.iidp.utils.distributed import get_allgather_value, print_one_rank
from torch.iidp.cluster.server import GlobalServerInfo, ServerInfo


class IIDPClusterManager(object):
    def __init__(self, gpu_cluster_info_file, available_server_name_list=[], gpu=None, **kwargs):
        #print_one_rank(f'[torch/iidp/cluster/cluster_manager.py] args - gpu_cluster_info_file: {gpu_cluster_info_file}', 'debug')
        self.homo_server_list = kwargs.get('homo_servers') or []
        #print_one_rank(f'[torch/iidp/cluster/cluster_manager.py] args - homo_servers: {self.homo_server_list}', 'debug')
        self.resource_alloc_unit = kwargs.get('resource_alloc_unit') or '2gpu'
        print_one_rank(f'[{self.__class__.__name__}] Arguments - '
                       f'resource_alloc_unit: {self.resource_alloc_unit} | '
                       f'homo_server_list: {self.homo_server_list}')
        self.gpu = gpu
        self.gpu_cluster_info = read_json(gpu_cluster_info_file)
        if socket.gethostname() not in self.gpu_cluster_info.keys():
            raise ValueError(
                f'Current server: {socket.gethostname()} is not registered '
                f'in gpu cluster info json file: {gpu_cluster_info_file}'
            )
        if torch.cuda.get_device_name() != self.gpu_cluster_info[socket.gethostname()]['type']:
            raise ValueError(
                f'Registerd GPU type in server {socket.gethostname()} in {gpu_cluster_info_file} '
                f'```{self.gpu_cluster_info[socket.gethostname()]["type"]}``` is not equal to '
                f'real GPU type in server: ```{torch.cuda.get_device_name()}```'
            )
        server_name_enum = {}
        for idx, server_name in enumerate(self.gpu_cluster_info.keys()):
            server_name_enum[idx] = server_name
            server_name_enum[server_name] = idx
        #print_one_rank(f'[cluster_manager.py] server_name_enum: {server_name_enum}', 'debug')
        # Initialize global server info (current servers)
        self.global_server_info = GlobalServerInfo()
        all_servername_per_rank_in_process_group = {
            rank: server_name_enum[server_idx] for rank, server_idx in enumerate(get_allgather_value(server_name_enum[socket.gethostname()], self.gpu))
        }
        server_group = {}
        for rank, server_name in all_servername_per_rank_in_process_group.items():
            server_group[server_name] = [rank] if server_name not in server_group.keys() else server_group[server_name] + [rank]
        for name, ranks in server_group.items():
            self.global_server_info.add(ServerInfo(name, ranks, self.gpu_cluster_info[name]))

        # Initialize available server info (future servers)
        self.available_server_info = GlobalServerInfo()
        for server_rank, server_name in enumerate(available_server_name_list):
            max_num_gpus_in_server = self.gpu_cluster_info[server_name]['number']
            ranks = [server_rank * max_num_gpus_in_server + rank for rank in range(int(max_num_gpus_in_server))]
            self.available_server_info.add(ServerInfo(server_name, ranks, self.gpu_cluster_info[server_name]))

        self.candidate_server_infos = []
        max_num_gpus_in_curr_server = self.gpu_cluster_info[socket.gethostname()]['number']
        self.alloc_unit_num = 0
        if self.resource_alloc_unit != 'server':
            try:
                self.alloc_unit_num = int(self.resource_alloc_unit.replace('gpu', ''))
            except:
                raise ValueError(
                    '```resource_alloc_unit``` has the format of ```<number of GPUs>gpu``` '
                    f'e.g, 2gpu, 4gpu, but {self.resource_alloc_unit}')
            if max_num_gpus_in_curr_server % self.alloc_unit_num != 0 or self.alloc_unit_num > max_num_gpus_in_curr_server:
                raise ValueError(
                    f'```resource_alloc_unit```: {self.resource_alloc_unit} -> {self.alloc_unit_num} '
                    f'do not have the right number of GPUs with number of current server '
                    f'{socket.gethostname()}: {max_num_gpus_in_curr_server}')

        if self.resource_alloc_unit == 'server':
            print_one_rank(
                f'[{self.__class__.__name__}] ********************** RESOURCE_ALLOC_UNIT = SERVER **********************', 'experiment'
            )
            if len(self.homo_server_list) > 0:
                raise ValueError('With resource_alloc_unit as ```server```, Not support ```homo_servers```')
            self._generate_candidate_server_pool()
        else:
            self._generate_candidate_resource_pool()

    def _split_rank_for_simigrad(self, arr, size):
        arrays = []
        while len(arr) > size:
            pice = arr[:size]
            arrays.append(pice)
            arr   = arr[size:]
        arrays.append(arr)
        return arrays

    def _generate_candidate_resource_pool(self):
        """Return: List of GlobalServerInfo"""
        if len(self.homo_server_list) > 0:
            return self._generate_candidate_resource_pool_with_homogeneous_server()
        #print_one_rank(f'_generate_candidate_resource_pool() - self.available_server_info: {self.available_server_info}', 'debug')
        num_gpus_unit = self.alloc_unit_num
        total_resource_unit_names = []
        for server_info in self.available_server_info:
            server_name = server_info.name
            split_ranks = self._split_rank_for_simigrad(server_info.ranks, num_gpus_unit)
            for ranks in split_ranks:
                resource_unit_name = server_name+':'+str(ranks[0])
                total_resource_unit_names.append(resource_unit_name)
        #print_one_rank(f'total_resource_unit_names: {total_resource_unit_names}', 'debug')
        candidate_resource_unit_name_config_set = []
        for num_unit in range(1, len(total_resource_unit_names)+1):
            #print_one_rank(f'===================================> num_unit: {num_unit}', 'debug')
            filtered_combinations = []
            combinations = list(itertools.combinations(total_resource_unit_names, num_unit))
            #print_one_rank(f'combinations: {combinations}', 'debug')
            #print_one_rank(f'Number of combinations: {len(combinations)}', 'debug')
            registered_name_combination = []
            for combination in combinations:
                register_name = []
                for unit_name_config in combination:
                    name, rank = unit_name_config.split(':')
                    register_name.append(name)
                if register_name not in registered_name_combination:
                    filtered_combinations.append(combination)
                    registered_name_combination.append(register_name)
            #print_one_rank(f'filtered_combinations: {filtered_combinations}', 'debug')
            #print_one_rank(f'Number of filtered_combinations: {len(filtered_combinations)}', 'debug')
            candidate_resource_unit_name_config_set.extend(filtered_combinations)
        #print_one_rank(f'candidate_resource_unit_name_config_set: {candidate_resource_unit_name_config_set}', 'debug')
        #print_one_rank(f'len(candidate_resource_unit_name_config_set): {len(candidate_resource_unit_name_config_set)}', 'debug')
        # ex) unit_name_config_combination: ('server1:0', 'server1:2', 'server2:4', 'server2:6', 'server3:8', 'server3:10', 'server4:12', 'server4:14')
        for unit_name_config_combination in candidate_resource_unit_name_config_set:
            candidate_server_info = GlobalServerInfo()
            for unit_name_config in unit_name_config_combination:
                name, rank = unit_name_config.split(':')
                ranks = list(range(int(rank), int(rank)+num_gpus_unit))
                candidate_server_info.add(ServerInfo(name, ranks, self.gpu_cluster_info[name]))
            self.candidate_server_infos.append(candidate_server_info)

    def _generate_candidate_resource_pool_with_homogeneous_server(self):
        PRINT_DEBUG_MSG = False
        if PRINT_DEBUG_MSG:
            print_one_rank(
                f'_generate_candidate_resource_pool_with_homogeneous_server() -'
                f'self.available_server_info: {self.available_server_info}', 'debug')
        num_gpus_unit = self.alloc_unit_num
        total_resource_unit_names = []
        for server_info in self.available_server_info:
            server_name = server_info.name
            split_ranks = self._split_rank_for_simigrad(server_info.ranks, num_gpus_unit)
            for ranks in split_ranks:
                resource_unit_name = server_name+':'+str(ranks[0])
                total_resource_unit_names.append(resource_unit_name)
        if PRINT_DEBUG_MSG:
            print_one_rank(f'total_resource_unit_names: {total_resource_unit_names}', 'debug')
        candidate_resource_unit_name_config_set = []
        for num_unit in range(1, len(total_resource_unit_names)+1):
            if PRINT_DEBUG_MSG:
                print_one_rank(f'===================================> num_unit: {num_unit}', 'debug')
            filtered_combinations = []
            combinations = list(itertools.combinations(total_resource_unit_names, num_unit))
            if PRINT_DEBUG_MSG:
                print_one_rank(f'Naive combinations: {combinations}', 'debug')
                print_one_rank(f'Number of naive combinations: {len(combinations)}', 'debug')
            registered_name_combination = []
            registered_device_combination = []
            for combination in combinations:
                register_name = []
                register_device = []
                for unit_name_config in combination:
                    name, rank = unit_name_config.split(':')
                    register_name.append(name)
                    register_device.append(self.gpu_cluster_info[name]['type'])
                if PRINT_DEBUG_MSG:
                    print_one_rank(f'==========> register_device: {register_device}', 'debug')
                if register_name not in registered_name_combination and register_device not in registered_device_combination:
                    filtered_combinations.append(combination)
                    registered_name_combination.append(register_name)
                    registered_device_combination.append(register_device)
            if PRINT_DEBUG_MSG:
                print_one_rank(f'filtered_combinations: {filtered_combinations}', 'debug')
                print_one_rank(f'Number of filtered_combinations: {len(filtered_combinations)}', 'debug')
            candidate_resource_unit_name_config_set.extend(filtered_combinations)
        if PRINT_DEBUG_MSG:
            print_one_rank(f'candidate_resource_unit_name_config_set: {candidate_resource_unit_name_config_set}', 'debug')
            print_one_rank(f'len(candidate_resource_unit_name_config_set): {len(candidate_resource_unit_name_config_set)}', 'debug')
        # ex) unit_name_config_combination: ('server1:0', 'server1:2', 'server2:4', 'server2:6', 'server3:8', 'server3:10', 'server4:12', 'server4:14')
        for unit_name_config_combination in candidate_resource_unit_name_config_set:
            candidate_server_info = GlobalServerInfo()
            for unit_name_config in unit_name_config_combination:
                name, rank = unit_name_config.split(':')
                ranks = list(range(int(rank), int(rank)+num_gpus_unit))
                candidate_server_info.add(ServerInfo(name, ranks, self.gpu_cluster_info[name]))
            self.candidate_server_infos.append(candidate_server_info)

    def _generate_candidate_server_pool(self):
        """Return: List of GlobalServerInfo"""
        num_gpus_unit = 2
        total_resource = dict()
        for server_info in self.available_server_info:
            server_name = server_info.name
            split_ranks = self._split_rank_for_simigrad(server_info.ranks, num_gpus_unit)
            total_resource[server_name] = split_ranks
        candidate_resource_unit_name_config_set = []
        for num_server in range(1, len(total_resource)+1):
            #print_one_rank(f'===================================> num_server: {num_server}', 'debug')
            filtered_combinations = []
            combinations = list(itertools.combinations(total_resource.keys(), num_server))
            #print_one_rank(f'combinations: {combinations}', 'debug')
            #print_one_rank(f'Number of combinations: {len(combinations)}', 'debug')
            registered_name_combination = []
            for combination in combinations:
                register_name = []
                unit_combination = set()
                for server_name in combination:
                    split_ranks = total_resource[server_name]
                    for ranks in split_ranks:
                        unit_name = server_name+':'+str(ranks[0])
                        unit_combination.add(unit_name)
                        register_name.append(server_name)
                if register_name not in registered_name_combination:
                    filtered_combinations.append(unit_combination)
                    registered_name_combination.append(register_name)
            #print_one_rank(f'filtered_combinations: {filtered_combinations}', 'debug')
            #print_one_rank(f'Number of filtered_combinations: {len(filtered_combinations)}', 'debug')
            candidate_resource_unit_name_config_set.extend(filtered_combinations)
        #print_one_rank(f'candidate_resource_unit_name_config_set: {candidate_resource_unit_name_config_set}', 'debug')
        #print_one_rank(f'len(candidate_resource_unit_name_config_set): {len(candidate_resource_unit_name_config_set)}', 'debug')
        # ex) == unit_name_config_combination ==
        # ({'server1:0', 'server1:2'}, {'server2:4', 'server2:6'}, {'server3:8', 'server3:10'}, {'server4:12', 'server4:14'})
        for unit_name_config_combination in candidate_resource_unit_name_config_set:
            candidate_server_info = GlobalServerInfo()
            for unit_name_config in unit_name_config_combination:
                name, rank = unit_name_config.split(':')
                ranks = list(range(int(rank), int(rank)+num_gpus_unit))
                candidate_server_info.add(ServerInfo(name, ranks, self.gpu_cluster_info[name]))
            self.candidate_server_infos.append(candidate_server_info)
        """
        for candidate_server_info in self.candidate_server_infos:
            print_one_rank(f'RESOURCE_ALLOC_UNIT = SERVER --> Candidate server info: {candidate_server_info}', 'debug')
        """