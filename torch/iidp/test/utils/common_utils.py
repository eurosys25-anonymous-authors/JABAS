import os
import math

from torch.iidp.cluster.server import GlobalServerInfo


# Similar logic in __init__ of IIDPConfigurator in torch/iidp/config/configurator.py
def get_possible_batch_size_across_cluster(comp_profile_dir, global_server_info):
    all_server_names = []
    if type(global_server_info) == list:
        all_server_names = global_server_info
    elif type(global_server_info) == GlobalServerInfo:
        for server_info in global_server_info:
            all_server_names.append(server_info.name)
    else:
        raise TypeError(f'Not support type of arugment global_server_info: {type(global_server_info)}')

    min_possible_lbs = math.inf
    for lbs in os.listdir(comp_profile_dir):
        local_batch_size = int(lbs)
        static_lbs_comp_profile_dir = os.path.join(comp_profile_dir, lbs)
        # Check current local batch size can be supported by current global servers
        """
        print(
            f'all_server_names: {all_server_names} | '
            f'static_lbs_comp_profile_dir: {static_lbs_comp_profile_dir} | '
            f'os.listdir(static_lbs_comp_profile_dir): {os.listdir(static_lbs_comp_profile_dir)}'
        )
        """
        if not set(all_server_names).issubset(set(os.listdir(static_lbs_comp_profile_dir))):
            """
            print(
                f'local_batch_size: {local_batch_size} '
                f'is not supported by current cluster: {" ".join(all_server_names)} '
                f'==> skip it for IIDP configuration'
            )
            """
            continue
        if local_batch_size < min_possible_lbs:
            min_possible_lbs = local_batch_size
    if type(min_possible_lbs) != int:
        raise ValueError(
            f'[ERROR][get_possible_batch_size_across_cluster()] '
            f'Not exists any possible min local batch size across cluster: '
            f'{",".join(all_server_names)}.\n'
            f'It might cause since no profile data for local batch size exists on some server.\n'
            f'Please check profile data directory: ```{comp_profile_dir}```')
    return min_possible_lbs
