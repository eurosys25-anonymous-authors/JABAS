

def get_iidp_config_map(config_str):
    iidp_config_map = {}
    unique_ranks = []
    rank_configs = config_str.split('|')
    for rank_config in rank_configs:
        rank, config_tuple = rank_config.split(':')
        iidp_config_map[int(rank)] = eval(config_tuple)
        # Check uniqueness of rank
        if rank not in unique_ranks:
            unique_ranks.append(rank)
        else:
            raise ValueError(f'Rank in argument string must be unique - {config_str}')
    return iidp_config_map