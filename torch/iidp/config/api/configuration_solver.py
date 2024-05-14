import warnings
import time

from torch.iidp.config.configurator import IIDPConfig, IIDPFutureConfigurator
from torch.iidp.cluster.cluster_manager import IIDPClusterManager
from torch.iidp.utils.json_utils import read_json
from torch.iidp.test.utils.server import build_mock_server_info, build_global_cluster_by_config_file
from torch.iidp.test.utils.common_utils import get_possible_batch_size_across_cluster

import argparse


parser = argparse.ArgumentParser(description='Configuration Solver API')
parser.add_argument('--config-file', '-c', type=str, required=True,
                    help='Configuration file path (json)')
parser.add_argument('--global-batch-size', '-gbs', default=None, type=int, required=True,
                    help='Global batch size')
parser.add_argument('--cluster', type=str, default=None, required=True,
                    help='Server group in GPU cluster - ' \
                    f'format: [hostname]:[num_gpus_in_server],[hostname]:[num_gpus_in_server], ..')
# Optional
parser.add_argument('--local-batch-size', '-lbs', default=None, type=int,
                    help='Local batch size to be preserved')
parser.add_argument('--weight-sync-method', type=str, default='overlap',
                    choices=['overlap', 'sequential'],
                    help='Weight synchronization method in IIDP')
parser.add_argument('--fixed-resource','-f', action='store_true',
                    help='Option to configure on --cluster setup only')


def main():
    args = parser.parse_args()

    config_params = read_json(args.config_file)
    # Handle config_params
    if "batch_size_upper_bound" not in config_params.keys():
        config_params["batch_size_upper_bound"] = args.global_batch_size
    if "homo_servers" not in config_params.keys():
        config_params["homo_servers"] = None
    if "resource_alloc_unit" not in config_params.keys():
        config_params["resource_alloc_unit"] = None
    enable_adjust_lbs = False
    if "enable_adjust_lbs" not in config_params.keys():
        if args.local_batch_size is not None:
            enable_adjust_lbs = False
        else:
            enable_adjust_lbs = True
    else:
        enable_adjust_lbs = bool(config_params["enable_adjust_lbs"] == "True")

    if enable_adjust_lbs is True:
        print(f'=====================================================')
        print(f'[INFO] Dynamic local batch size')
        print(f'=====================================================')
        if args.local_batch_size is not None:
            warnings.warn(
                f'With dynamic local batch size mode, not fixed with '
                f'--local-batch-size {args.local_batch_size}'
            )
    else:
        print(f'=====================================================')
        print(f'[INFO] Static local batch size')
        print(f'=====================================================')

    gpu_cluster_info = read_json(config_params["gpu_cluster_info"])

    # Build Mock global and available server info
    if args.cluster == 'all':
        args.cluster = build_global_cluster_by_config_file(
                config_params["available_servers"], gpu_cluster_info)
    mock_available_server_info, mock_global_server_group = build_mock_server_info(args.cluster, gpu_cluster_info)
    # Make arbitrary current global server info
    server_list = args.cluster.split(',')
    if len(server_list) > 1:
        num_total_servers = len(server_list)
        mock_current_cluster = ','.join(args.cluster.split(',')[:num_total_servers//2])
    else:
        mock_current_cluster = args.cluster
    #print(f'[TEST] mock_current_cluster: {mock_current_cluster}')
    mock_global_server_info, _ = build_mock_server_info(mock_current_cluster, gpu_cluster_info)

    #print(f'[TEST] mock_available_server_info: {mock_available_server_info}')
    mock_available_server_name_list = mock_global_server_group.keys()
    mock_cluster_manager = IIDPClusterManager(
            config_params["gpu_cluster_info"], mock_available_server_name_list,
            homo_servers=config_params["homo_servers"],
            resource_alloc_unit=config_params["resource_alloc_unit"])

    #print('\n=============================== Mock Test ==================================\n')
    mock_cluster_manager.global_server_info = mock_global_server_info
    mock_cluster_manager.available_server_info = mock_available_server_info
    mock_cluster_manager.candidate_server_infos = []
    mock_cluster_manager._generate_candidate_resource_pool()

    # Get local batch size
    if args.local_batch_size is None:
        try:
            args.local_batch_size = get_possible_batch_size_across_cluster(
                    config_params["comp_profile_dir"], list(mock_available_server_name_list))
        except Exception as e:
            print(e)
            print('[ERROR] Argument --cluster must be re-configured')
            exit(1)
    #print(f'[DEBUG] args.local_batch_size: {args.local_batch_size}')

    # Build Mock IIDPFutureConfigurator
    local_config = IIDPConfig(args.local_batch_size, 1, 1, args.weight_sync_method)
    mock_future_configurator = IIDPFutureConfigurator(
        config_params["comp_profile_dir"],
        config_params["comm_profile_dir"],
        config_params["bucket_profile_dir"],
        config_params["memory_profile_dir"],
        local_config,
        mock_cluster_manager.candidate_server_infos,
        args.global_batch_size,
        enable_adjust_lbs
    )
    mock_future_configurator.prepare(verbose=False)
    # Search space of GPU configuration
    best_throughput = 0
    best_result = []
    best_lbs = 0
    print('[INFO] ====================== Search start! ======================\n')
    search_start_time = time.time()
    for server_id, candidate_server_info in enumerate(mock_cluster_manager.candidate_server_infos):
        if args.fixed_resource is True and mock_cluster_manager.available_server_info != candidate_server_info:
            continue
        mock_future_configurator.update(server_id, 0, 0)
        for local_batch_size, configurator in mock_future_configurator.configurators.items():
            result = []
            total_num_workers = int(args.global_batch_size / local_batch_size)
            #print(f'[TEST] GBS: {args.global_batch_size} | LBS: {local_batch_size} | M: {total_num_workers}')
            if args.global_batch_size != (local_batch_size * total_num_workers):
                print(f'[WARNING] GBS: {args.global_batch_size} != LBS: {local_batch_size} * M: {total_num_workers} ==> skip!')
                continue
            if total_num_workers >= configurator.total_num_gpus:
                result = configurator.dp_solver.solve(total_num_workers)
                print(f'[TEST] LBS: {local_batch_size} | DP solution - [throughput, iteration time, number of workers/2, configuration set]: {result}')
                if result[2]*2 != total_num_workers:
                    print(f"[WARNING] resulting number of total workers: {result[2]*2}, but required one: {total_num_workers} ==> skip!")
                    continue
            else:
                print(f'[TEST] GBS: {args.global_batch_size} | LBS: {local_batch_size} | '
                      f'total_num_workers: {total_num_workers} < configurator.total_num_gpus: {configurator.total_num_gpus}')
            if len(result)> 0 and result[0] > best_throughput:
                best_throughput = result[0]
                best_result = result
                best_lbs = local_batch_size
                print(f'** [DEBUG] intermediate solution - LBS: {best_lbs} | '
                      f'[throughput, iteration time, number of workers/2, configuration set]: {best_result} **')
    if len(best_result) == 0:
        print(f'\n=====================================================')
        print(f'[INFO] No solution - GBS: {args.global_batch_size}')
        print(f'=====================================================\n')
    else:
        print(f'\n=====================================================')
        print(f'[INFO] Solution - GBS: {args.global_batch_size} | LBS: {best_lbs} | config: {best_result[-1]}')
        print(f'=====================================================')
        print(f'[throughput, iteration time, number of workers/2, configuration set]: {best_result}')
        print(f'=====================================================\n')
    print(f'=====================================================')
    print(f'[INFO] Search time (sec): {time.time()-search_start_time:.3f}')
    print(f'=====================================================')


if __name__ == '__main__':
    main()
