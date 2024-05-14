import argparse
import os

from torch.iidp.config.model.comm.allreduce_model import AllreduceModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Allreduce Regression Model Test code')
    parser.add_argument('--comm-profile-dir', type=str, default=None, required=True,
                        help='(intra, inter) allreduce profile data directory')

    args = parser.parse_args()

    intra_allreduce_model, inter_allreduce_model = AllreduceModel(), AllreduceModel()
    for comm_profile_file in os.listdir(args.comm_profile_dir):
        comm_profile_file_path = os.path.join(args.comm_profile_dir, comm_profile_file)
        print(f'[INFO] profile data file path: {comm_profile_file_path}')
        x_data, y_data = [], []
        with open(comm_profile_file_path, 'r') as f:
            for line in f.readlines():
                x_data.append(float(line.split(',')[0]))
                y_data.append(float(line.split(',')[1]))
        if 'intra' in comm_profile_file:
            intra_allreduce_model.train(x_data, y_data)
            intra_allreduce_model.plot(x_data, y_data, 'intra_allreduce_model.png', 'Intra-node All-reduce Model')
        elif 'inter' in comm_profile_file:
            inter_allreduce_model.train(x_data, y_data)
            inter_allreduce_model.plot(x_data, y_data, 'inter_allreduce_model.png', 'Inter-node All-reduce Model')
        else:
            raise ValueError(
                f'allreduce profile filename must inclue inter or intra term: {comm_profile_file}')

    bucket_cap = 126 * 1024 * 1024
    total_num_gpus = 4
    bandwidth = 15750000000
    ideal_allreduce_time = (bucket_cap * 4 * (total_num_gpus-1)/total_num_gpus) / bandwidth * 1000
    predicted_allreduce_time = inter_allreduce_model.evaluate(ideal_allreduce_time)
    print('Theoretical intra-node all-reduce time (ms):', ideal_allreduce_time)
    print('Predicted intra-node all-reduce time (ms):', predicted_allreduce_time)

    bucket_cap = 126 * 1024 * 1024
    total_num_gpus = 8
    bandwidth = 7000000000
    ideal_allreduce_time = (bucket_cap * 4 * (total_num_gpus-1)/total_num_gpus) / bandwidth * 1000
    predicted_allreduce_time = inter_allreduce_model.evaluate(ideal_allreduce_time)
    print('Theoretical inter-node all-reduce time (ms):', ideal_allreduce_time)
    print('Predicted inter-node all-reduce time (ms):', predicted_allreduce_time)

    print('... test save() ...')
    intra_allreduce_model.save('train_result/intra_model_params.txt')
    inter_allreduce_model.save('train_result/inter_model_params.txt')
    print('... test save() done! ...')

    print('... test load() ...')
    intra_allreduce_model.a = 0
    intra_allreduce_model.b = 0
    print('Initializing model:', intra_allreduce_model)
    intra_allreduce_model.load('train_result/intra_model_params.txt')
    print('Loaded model:', intra_allreduce_model)
    print('... test load() done! ...')
