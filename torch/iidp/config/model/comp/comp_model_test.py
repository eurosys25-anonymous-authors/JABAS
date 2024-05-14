import argparse
import os
import json
import socket

from torch.iidp.config.model.comp.comp_model import StreamParallelThroughputModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Computation Model Test code')
    parser.add_argument('--comp-profile-dir', type=str, default=None, required=True,
                        help='computation profile data directory')

    args = parser.parse_args()

    stream_parallel_model = StreamParallelThroughputModel()

    x_data, y_data = [], []
    init_thp = 0
    # NOTE: file must be sorted by number of models in increasing order
    for comp_profile_file in sorted(os.listdir(args.comp_profile_dir)):
        comp_profile_file_path = os.path.join(args.comp_profile_dir, comp_profile_file)
        print(f'[INFO] profile data file path: {comp_profile_file_path}')
        try:
            with open(comp_profile_file_path, 'r') as jf:
                json_data = json.load(jf)
        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
            exit(1)
        lbs = json_data['lbs']
        num_models = json_data['num_models']
        fwd_time = json_data['fwd_time']
        bwd_time = json_data['bwd_time']
        fwd_bwd_time = (fwd_time + bwd_time) / 1000 # convert to ms
        if num_models == 1:
            init_thp = round((lbs * num_models) / fwd_bwd_time, 2)
            print(f'[INFO] initial throughput = {init_thp}')
        thp = (lbs * num_models) / fwd_bwd_time
        norm_thp = round(thp / init_thp, 3)
        x_data.append(num_models)
        y_data.append(norm_thp)
    print(x_data, y_data)
    stream_parallel_model.train(x_data, y_data)
    model_name = json_data['model']
    gpu_type = json_data['gpu_type']
    stream_parallel_model.plot(x_data, y_data, f"{socket.gethostname()}_{model_name}_{lbs}.png", f"{model_name} ({lbs}) on {gpu_type}")

    example_num_models = 3
    predicted_norm_thp = stream_parallel_model.evaluate(example_num_models)
    print(f'Predicted normalized throughput = {round(predicted_norm_thp, 2)}')
    print(f'Predicted throughput (inputs/sec) = {round(predicted_norm_thp * init_thp, 2)}')
    predicted_fwd_bwd_time = (lbs * example_num_models) / (predicted_norm_thp * init_thp)
    print(f'Predicted forward + backward time (sec) on number of VSWs: {example_num_models} = {round(predicted_fwd_bwd_time, 4)}')

