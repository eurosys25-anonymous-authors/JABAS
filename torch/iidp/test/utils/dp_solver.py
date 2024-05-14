import os
from torch.iidp.config.model.throughput.throughput_model import ThroughputModel
from torch.iidp.config.configurator import DynamicProgrammingSolver
from torch.iidp.utils.json_utils import read_json
from torch.iidp.profiler.profiler import MAX_MEM_PROFILE_FILE_NAME


def create_memory_profile_info(memory_profile_dir):
    # Create memory profile info (type: dict)
    memory_profile_info = {}
    for lbs in os.listdir(memory_profile_dir):
        if not lbs in memory_profile_info.keys():
            memory_profile_info[lbs] = {}
        static_lbs_comp_profile_dir = os.path.join(memory_profile_dir, lbs)
        for server_name in os.listdir(static_lbs_comp_profile_dir):
            max_memory_profile_file = os.path.join(
                static_lbs_comp_profile_dir, server_name, MAX_MEM_PROFILE_FILE_NAME)
            memory_profile_json_data = read_json(max_memory_profile_file)
            memory_profile_info[lbs][memory_profile_json_data['gpu_type']] = memory_profile_json_data['max_num_models']
    return memory_profile_info


def instanciate_dp_solver(config_params, global_server_info, local_batch_size, weight_sync_method):
    comp_profile_dir = config_params["comp_profile_dir"]
    comm_profile_dir = config_params["comm_profile_dir"]
    static_lbs_comp_profile_dir = os.path.join(comp_profile_dir, str(local_batch_size))
    bucket_profile_dir = config_params["bucket_profile_dir"]
    memory_profile_dir = config_params["memory_profile_dir"]
    throughput_models = {}
    for server_info in global_server_info:
        local_comp_profile_dir = os.path.join(static_lbs_comp_profile_dir, server_info.name)
        throughput_models[server_info.name] = \
            ThroughputModel(local_comp_profile_dir, comm_profile_dir, bucket_profile_dir)

    if len(os.listdir(comp_profile_dir)) != len(os.listdir(memory_profile_dir)):
        raise ValueError(
            f'[ERROR] Computation and memory profile data for range of local batch size '
            f'must be equal, but comp: {os.listdir(comp_profile_dir)} | mem: {os.listdir(memory_profile_dir)}')
    memory_profile_info = create_memory_profile_info(memory_profile_dir)
    static_lbs_mem_profile_info = memory_profile_info[str(local_batch_size)]

    if "batch_size_upper_bound" not in config_params.keys():
        max_num_workers = -1
    else:
        max_num_workers = config_params["batch_size_upper_bound"]//local_batch_size+1
    dp_solver = DynamicProgrammingSolver(
                    local_batch_size, weight_sync_method, throughput_models, static_lbs_mem_profile_info,
                    global_server_info, max_num_workers)
    return dp_solver