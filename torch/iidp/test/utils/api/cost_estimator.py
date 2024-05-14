import os

from torch.iidp.test.utils.server import build_mock_server_info, build_server_from_resource_info
from torch.iidp.utils.json_utils import read_json

from torch.iidp.utils.cost_utils import estimate_cost

import argparse

parser = argparse.ArgumentParser(description='Cost Estimator')
parser.add_argument('-i', '--input-file', default=None, type=str, required=True,
                    help='path to input file (convergence log)')
parser.add_argument('-c', '--config-file', type=str, required=True,
                    help='Training configuration file path (json)')


def parse_time_per_epoch(args):
    all_time_per_epoch = []
    epoch_identifier = 'Epoch time:'
    with open(args.input_file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if epoch_identifier in line and 'min' not in line and 'total' not in line:
                try:
                    data = int(float(line.split(' ')[3])) # -1 doesn't work due to reallocation time string
                except:
                    print('[ERROR]' + line)
                    print(line.split(' '))
                    exit(1)
                #print(data)
                all_time_per_epoch.append(data)

    return all_time_per_epoch


def parse_as_time_per_epoch(args):
    all_time_per_epoch = []
    epoch_identifier = 'Forecasting overhead takes'
    with open(args.input_file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if epoch_identifier in line:
                try:
                    data = int(float(line.split(' ')[-2])) # -1 is "sec"
                except:
                    print('[ERROR]' + line)
                    print(line.split(' '))
                    exit(1)
                #print(data)
                all_time_per_epoch.append(data)

    return all_time_per_epoch


def parse_resource_per_epoch(args):
    all_data = []
    identifier = 'current resource info:'
    with open(args.input_file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if identifier in line:
                try:
                    data = eval(line.split(identifier)[-1])
                except:
                    print('[ERROR]' + line)
                    print(line.split(' '))
                    exit(1)
                #print(data)
                all_data.append(data)

    num_gpus_data = []
    all_gpus_info = []
    for data in all_data:
        gpus_info = {}
        for key, val in data.items():
            if key == 'total_num_gpus':
                num_gpus_data.append(int(val))
            else:
                gpus_info[key] = int(val)
        all_gpus_info.append(gpus_info)

    return all_gpus_info


def measure_epoch_cost(epoch, epoch_duration, global_server_info, resource_info):
    assert epoch_duration > 0, \
        f"[ERROR][torch/iidp/test/utils/api/cost_estimator.py] measure_epoch_cost() " \
        f"Argument ```epoch_duration``` must be > 0"
    total_cost_per_epoch = 0
    for server_info in global_server_info:
        cost = estimate_cost(
                server_info.resource_info.tfplos,
                server_info.resource_info.num_gpus_in_server,
                epoch_duration / 3600 # convert to sec to hour
            )
        assert cost > 0, \
            f"[ERROR][torch/iidp/test/utils/api/cost_estimator.py] measure_epoch_cost() " \
            f"Return value of ```estimate_cost()``` must be > 0 | " \
            f"server_info: {server_info} | epoch_duration: {epoch_duration}"
        total_cost_per_epoch += cost
    print(f'[epoch {epoch}] Epoch cost: {total_cost_per_epoch:.2f} | '
          f'duration (sec): {epoch_duration} | '
          f'resource info: {resource_info}')
    return total_cost_per_epoch


def main():
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        raise FileExistsError(f'--input-file must be exist: {args.input_file}')
    #args.output_dir = '/'.join(args.input_file.split('/')[:-1])
    #print(f'[INFO] output dir: {args.output_dir}')
    #os.makedirs(args.output_dir, exist_ok=True)

    config_params = read_json(args.config_file)
    gpu_cluster_info = read_json(config_params["gpu_cluster_info"])
    available_servers = config_params["available_servers"]
    server_str_list = []
    for server_name in available_servers:
        server_str_list.append(f'{server_name}:{gpu_cluster_info[server_name]["number"]}')
    cluster_str = ','.join(server_str_list)
    mock_available_server_info, _ = build_mock_server_info(cluster_str, gpu_cluster_info)

    all_time_per_epoch = parse_time_per_epoch(args)
    all_as_time_per_epoch = parse_as_time_per_epoch(args)
    if len(all_as_time_per_epoch) == 0:
        all_as_time_per_epoch = [0] * (len(all_time_per_epoch)-1)
    if len(all_time_per_epoch) != len(all_as_time_per_epoch)+1:
        raise ValueError(f'Number of 1) time per epoch and 2) ```auto-scaling overhead per epoch``` must be same, '
                         f'but 1) {len(all_time_per_epoch)} and 2) {len(all_as_time_per_epoch)}')
    else:
        # NOTE: At last epoch, auto-scaling is not executed
        all_as_time_per_epoch.append(0)

    all_gpus_info_per_epoch = parse_resource_per_epoch(args)
    if len(all_time_per_epoch) != len(all_gpus_info_per_epoch):
        raise ValueError(f'Number of 1) time per epoch and 2) ```resource (GPU) info``` must be same, '
                         f'but 1) {len(all_time_per_epoch)} and 2) {len(all_gpus_info_per_epoch)}')

    total_cost = 0
    for epoch, (epoch_duration, as_time, resource_info) in \
            enumerate(zip(all_time_per_epoch, all_as_time_per_epoch, all_gpus_info_per_epoch)):
        mock_global_server_info = build_server_from_resource_info(resource_info, mock_available_server_info, gpu_cluster_info)
        total_cost += measure_epoch_cost(epoch, epoch_duration+as_time, mock_global_server_info, resource_info)

    print(f'\n=====================================================')
    print(f'[INFO] Total cost (dollar): {total_cost:.3f}')
    print(f'=====================================================')



if __name__ == '__main__':
    main()
