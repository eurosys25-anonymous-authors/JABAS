from torch.iidp.cluster.server import GlobalServerInfo, ServerInfo


def build_global_cluster_by_config_file(available_servers: list, gpu_cluster_info: dict):
    cluster_str_list = []
    for available_server in available_servers:
        num_gpus_in_server = gpu_cluster_info[available_server]['number']
        cluster_str_list.append(f'{available_server}:{num_gpus_in_server}')
    cluster = ','.join(cluster_str_list)
    print('====================================================================')
    print(f'[TEST] Global cluster: {cluster}')
    print('====================================================================')
    return cluster


def build_mock_server_info(cluster: str, gpu_cluster_info: dict):
    if type(cluster) != str:
        raise TypeError(f'[ERROR] Argument cluster must be string type, '
                        f'but type: {type(cluster)} | cluster: {cluster}')
    if type(gpu_cluster_info) != dict:
        raise TypeError(f'[ERROR] Argument gpu_cluster_info must be dictionary type, '
                        f'but type: {type(gpu_cluster_info)} | gpu_cluster_info: {gpu_cluster_info}')
    mock_global_server_group = {}
    server_groups = cluster.split(',')
    last_rank = 0
    total_num_gpus = 0
    for server_group in server_groups:
        hostname, num_gpus_in_server = server_group.split(':')
        ranks = [last_rank + rank for rank in range(int(num_gpus_in_server))]
        last_rank = ranks[-1] + 1
        mock_global_server_group[hostname] = ranks
        total_num_gpus+=int(num_gpus_in_server)
    print(f'[TEST] Mock server group: {mock_global_server_group}')
    mock_global_server_info = GlobalServerInfo()
    for name, ranks in mock_global_server_group.items():
        mock_global_server_info.add(ServerInfo(name, ranks, gpu_cluster_info[name]))
    print(f'[TEST] Mock Global Server Info: {mock_global_server_info}')
    return mock_global_server_info, mock_global_server_group


def build_server_from_resource_info(resource_info: dict, available_servers: GlobalServerInfo, gpu_cluster_info: dict):
    ret_server_info = GlobalServerInfo() # return value
    device_names_in_resource_info = list(resource_info.keys())
    device_names_in_available_servers = []
    for server in available_servers:
        device_names_in_available_servers.append(server.resource_info.device_name)
    if not set(device_names_in_resource_info).issubset(set(device_names_in_available_servers)):
        raise AssertionError(
            f'[ERROR][torch/iidp/test/utils/server.py] '
            f'build_server_from_resource_info() => '
            f'No common device between resource info and available servers!\n'
            f'===================================================\n'
            f'Device names in ```available_servers```: {device_names_in_available_servers} \n'
            f'===================================================\n'
            f'Device names in ```resource_info```: {device_names_in_resource_info}')

    remaining_num_gpus = 0
    for gpu_type, num_gpus in resource_info.items():
        remaining_num_gpus = num_gpus
        for server in available_servers:
            if server.resource_info.device_name == gpu_type and remaining_num_gpus > 0:
                allocated_num_gpus = min(server.resource_info.max_num_gpus_in_server, remaining_num_gpus)
                ranks = list(range(server.ranks[0], server.ranks[0]+allocated_num_gpus))
                ret_server_info.add(
                    ServerInfo(server.name, ranks, gpu_cluster_info[server.name])
                )
                remaining_num_gpus -= allocated_num_gpus
        assert remaining_num_gpus == 0, \
            f"[ERROR][torch/iidp/test/utils/server.py] " \
            f"build_server_from_resource_info() => " \
            f"total number of GPUs from resource info cannot be used by available servers! " \
            f"gpu_type: {gpu_type} | num_gpus: {num_gpus}\n" \
            f'===================================================\n' \
            f'available_servers: {available_servers} \n' \
            f'===================================================\n'

    if ret_server_info.total_num_servers == 0:
        raise AssertionError(
            f'[ERROR][torch/iidp/test/utils/server.py] '
            f'build_server_from_resource_info() => server info is empty!\n'
            f'===================================================\n'
            f'available_servers: {available_servers} \n'
            f'===================================================\n'
            f'resource_info: {resource_info}')
    return ret_server_info