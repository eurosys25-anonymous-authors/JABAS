import os
import json
import socket

import matplotlib.pyplot as plt

from torch.iidp.utils.json_utils import read_json
from torch.iidp.utils.distributed import print_one_rank
from torch.iidp.config.model.comm.allreduce_model import AllreduceModel
from torch.iidp.config.model.comp.comp_model import StreamParallelThroughputModel, TrueThroughputModel
from torch.iidp.cluster.resource import ResourceInfo, GlobalResourceInfo
from torch.iidp.config.examples.config_utils import MODEL_TITLE_MAP


class ThroughputModel(object):
    def __init__(self, comp_profile_dir, comm_profile_dir, bucket_profile_dir,
                 plot_dir=None, is_real_comp=False, verbose=False):
        self.comp_profile_dir = comp_profile_dir
        self.comm_profile_dir = comm_profile_dir
        self.bucket_profile_dir = bucket_profile_dir
        self.plot_dir = plot_dir
        self.verbose = verbose
        if self.plot_dir:
            print(f'[INFO][{self.__class__.__name__}] Make plot directory: {self.plot_dir}')
            os.makedirs(self.plot_dir, exist_ok=False)

        self.all_comp_data = []

        self._get_all_comp_profile_data()
        self.get_constant()
        if self.plot_dir:
            self.plot_comp_profile_data()
            self.plot_bucket_size_distribution()
        self._build_allreduce_model()
        if is_real_comp:
            self._build_true_comp_model()
        else:
            self._build_comp_model()

    def _get_all_comp_profile_data(self):
        for comp_profile_file in os.listdir(self.comp_profile_dir):
            comp_profile_file_path = os.path.join(self.comp_profile_dir, comp_profile_file)
            #print_one_rank(f'Computation profile data file path: {comp_profile_file_path}', 'debug')
            json_data = read_json(comp_profile_file_path)
            self.all_comp_data.append(json_data)
        # Sort data by number of models in increasing order
        self.all_comp_data.sort(key= lambda x:x['num_models'])

    def get_constant(self):
        json_data = self.all_comp_data[-1] # Data with max number of models
        self.max_num_models = json_data['num_models']
        self.lbs = json_data['lbs']
        self.gpu_type = json_data['gpu_type']
        self.model = json_data['model']
        try:
            self.model_name = MODEL_TITLE_MAP[self.model]
        except:
            print_one_rank(f'[WARNING][{self.__class__.get_constant.__qualname__}] Model name is not registerd: {self.model}')
            self.model_name = self.model
        self.update_time = json_data['update_time'] / 1000 # ms -> s
        if self.max_num_models > 1:
            self.copy_time_per_model = (json_data['copy_time'] / (self.max_num_models-1)) / 1000 # ms -> s
        else:
            self.copy_time_per_model = 0
        self.fwd_ratio = json_data['fwd_time'] / (json_data['fwd_time'] + json_data['bwd_time'])
        self.bwd_ratio = 1 - self.fwd_ratio

        bucket_profile_file_name = sorted(os.listdir(self.bucket_profile_dir))[-1]
        comp_profile_file_path = os.path.join(self.bucket_profile_dir, bucket_profile_file_name)
        json_data = read_json(comp_profile_file_path)
        self.bucket_size_distribution = json_data['bucket_size_distribution']

    def plot_comp_profile_data(self, file_path='comp_profile_data_breakdown.png'):
        x_data = []
        fwd_time = []
        bwd_time = []
        update_time = []
        copy_time = []
        for data in self.all_comp_data:
            x_data.append(str(data['num_models']))
            fwd_time.append(data['fwd_time']/data['total_time'])
            bwd_time.append(data['bwd_time']/data['total_time'])
            update_time.append(data['update_time']/data['total_time'])
            copy_time.append(data['copy_time']/data['total_time'])
        breakdown_data = [fwd_time, bwd_time, update_time, copy_time]
        plt.clf()
        stacked_data = [0 for _ in range(len(x_data))]
        labels = ['Forward', 'Backward', 'Update', 'Copy']
        for i, data in enumerate(breakdown_data):
            plt.bar(x_data, data, bottom=stacked_data, label=labels[i])
            stacked_data = [prev + data for prev, data in zip(stacked_data, data)]
        plt.xlabel('Number of VSWs')
        plt.ylabel('Normalized throughput breakdown')
        plt.legend()
        title = f'{self.model_name} ({self.lbs}) on {self.gpu_type}'
        plt.title(title)
        file_path = f'{socket.gethostname()}_{self.model}_{self.lbs}_{file_path}'
        file_path = os.path.join(self.plot_dir, file_path)
        plt.savefig(file_path)

    def plot_bucket_size_distribution(self, file_path='bucket_size_distribution.png'):
        plt.clf()
        x = list(range(len(self.bucket_size_distribution)))
        plt.bar(x, self.bucket_size_distribution)

        plt.xlabel('Bucket order (backward)')
        plt.ylabel('Bucket size (MB)')

        title = f'{self.model_name}'
        plt.title(title)

        file_path = f'{self.model}_{file_path}'
        file_path = os.path.join(self.plot_dir, file_path)
        plt.savefig(file_path)

    def _build_allreduce_model(self):
        self.intra_allreduce_model, self.inter_allreduce_model = AllreduceModel(), AllreduceModel()
        for comm_profile_file in os.listdir(self.comm_profile_dir):
            comm_profile_file_path = os.path.join(self.comm_profile_dir, comm_profile_file)
            #print_one_rank(f'All-reduce profile data file path: {comm_profile_file_path}')
            x_data, y_data = [], []
            with open(comm_profile_file_path, 'r') as f:
                for line in f.readlines():
                    x_data.append(float(line.split(',')[0]))
                    y_data.append(float(line.split(',')[1]))
            if 'intra' in comm_profile_file:
                self.intra_allreduce_model.train(x_data, y_data)
                if self.plot_dir:
                    plot_file_path = os.path.join(self.plot_dir, 'intra_allreduce_model.png')
                    self.intra_allreduce_model.plot(
                            x_data, y_data, plot_file_path, 'Intra-node All-reduce Model')
            elif 'inter' in comm_profile_file:
                self.inter_allreduce_model.train(x_data, y_data)
                if self.plot_dir:
                    plot_file_path = os.path.join(self.plot_dir, 'inter_allreduce_model.png')
                    self.inter_allreduce_model.plot(
                            x_data, y_data, plot_file_path, 'Inter-node All-reduce Model')
            else:
                raise ValueError(
                    f'allreduce profile filename must inclue inter or intra term: {comm_profile_file}')

    def _build_comp_model(self):
        if self.max_num_models < 3:
            return self._build_true_comp_model()
        self.stream_parallel_thp_model = StreamParallelThroughputModel()
        x_data, y_data = [], []
        self.init_thp = 0
        for json_data in self.all_comp_data:
            num_models = json_data['num_models']
            fwd_time = json_data['fwd_time']
            bwd_time = json_data['bwd_time']
            fwd_bwd_time = (fwd_time + bwd_time) / 1000 # convert to ms
            #print_one_rank(f'comp time on # of VSWs: {num_models} = {fwd_bwd_time}')
            if num_models == 1:
                self.init_thp = round((self.lbs * num_models) / fwd_bwd_time, 2)
                #print_one_rank(f'initial throughput = {self.init_thp}')
            else:
                assert self.init_thp != 0, f"[ERROR] self.init_thp must be > 0 if num_models > 1 - file order maybe not sorted"
            thp = (self.lbs * num_models) / fwd_bwd_time
            #print_one_rank(f'throughput on # of VSWs: {num_models} = {thp}')
            norm_thp = round(thp / self.init_thp, 3)
            x_data.append(num_models)
            y_data.append(norm_thp)
        self.stream_parallel_thp_model.train(x_data, y_data)
        if self.plot_dir:
            self.stream_parallel_thp_model.plot(
                    x_data, y_data,
                    f"{self.plot_dir}/{socket.gethostname()}_{self.model}_{self.lbs}_comp_model.png",
                    f"{self.model_name} ({self.lbs}) on {self.gpu_type}"
            )

    def _build_true_comp_model(self):
        self.stream_parallel_thp_model = TrueThroughputModel()
        x_data, y_data = [], []
        self.init_thp = 0
        for json_data in self.all_comp_data:
            num_models = json_data['num_models']
            fwd_time = json_data['fwd_time']
            bwd_time = json_data['bwd_time']
            fwd_bwd_time = (fwd_time + bwd_time) / 1000 # convert to ms
            if num_models == 1:
                self.init_thp = round((self.lbs * num_models) / fwd_bwd_time, 2)
                #print_one_rank(f'initial throughput = {self.init_thp}')
            thp = (self.lbs * num_models) / fwd_bwd_time
            norm_thp = round(thp / self.init_thp, 3)
            x_data.append(num_models)
            y_data.append(norm_thp)
        self.stream_parallel_thp_model.train(x_data, y_data)

    def plot_evaluate_breakdown(self, comp_data, sync_data, file_path, title):
        if self.plot_dir is None:
            return
        plt.clf()
        plt.rcParams["figure.figsize"] = [8, 3]
        plt.rcParams["figure.autolayout"] = True
        plt.figure()
        data_type = ['comp', 'sync']
        comp_labels = {'fwd': 'Forward', 'bwd': 'Backward', 'GA': 'GA'}
        sync_labels = {
            'overlap': 'overlap all-reduce',
            'last_bucket': 'Non-overlap all-reduce',
            'update': 'Update',
            'copy': 'Weight copy'
        }

        stacked_data = [0 for _ in range(len(data_type))] # [0] - comp, [1] - sync
        for key, val in comp_data.items():
            plt.barh(data_type, [val, 0], left=stacked_data, label=comp_labels[key], height=0.4)
            stacked_data[0] += val

        for key, val in sync_data.items():
            if key == 'overlap':
                if comp_data['bwd'] > val:
                    stacked_data[1] = stacked_data[0] - val
                else:
                    stacked_data[1] = stacked_data[0] - comp_data['bwd']
            plt.barh(data_type, [0, val], left=stacked_data, label=sync_labels[key], height=0.4)
            stacked_data[1] += val

        plt.xlabel('Iteration time (sec)')
        plt.legend()
        plt.title(title)
        file_path = os.path.join(self.plot_dir, file_path)
        plt.savefig(file_path)

    def evaluate_fwd_bwd_time(self, num_models):
        predicted_norm_thp = self.stream_parallel_thp_model.evaluate(num_models)
        predicted_fwd_bwd_time = round((self.lbs * num_models) / (predicted_norm_thp * self.init_thp), 4)
        return predicted_fwd_bwd_time

    def calculate_ideal_allreduce_time(self, bucket_size_byte, total_num_gpus, bandwidth):
        allreduce_step = 4*(total_num_gpus-1)
        network_volume = bucket_size_byte/total_num_gpus * allreduce_step
        return network_volume / bandwidth * 1000 # sec to ms

    def evaluate(self, num_models, accum_step, weight_sync_method,
                 resource_info: ResourceInfo, global_resource_info: GlobalResourceInfo):
        """
        Args:
            num_models (int): Number of VSWs
            accum_step (int): GA steps
            weight_sync_method (str): 'overlap', 'sequential'
            resource_info (ResourceInfo): Resource Information of intra-server aspect

        Returns:
            iter_time (float): predicted iteration time within server level
            predicted_thp (float): predicted throughput within server level
        """
        predicted_thp = 0
        fwd_time = self.evaluate_fwd_bwd_time(num_models) * self.fwd_ratio
        bwd_time = self.evaluate_fwd_bwd_time(num_models) * self.bwd_ratio
        if self.verbose:
            print_one_rank(f'====== [{self.__class__.evaluate.__qualname__}] ======')
            print_one_rank(f'Predicted fwd time: {fwd_time:.3f} | bwd time: {bwd_time:.3f}')
        all_bucket_allreduce_time = []
        if global_resource_info.total_num_servers > 1:
            bandwidth = resource_info.inter_network_bandwidth
            allreduce_model = self.inter_allreduce_model
        else:
            bandwidth = resource_info.intra_network_bandwidth
            allreduce_model = self.intra_allreduce_model
        for bucket_size in self.bucket_size_distribution:
            bucket_size_byte = bucket_size * 1024 * 1024
            ideal_allreduce_time = self.calculate_ideal_allreduce_time(
                bucket_size_byte, global_resource_info.total_num_gpus, bandwidth)
            predicted_allreduce_time = allreduce_model.evaluate(ideal_allreduce_time) / 1000 # ms to sec
            all_bucket_allreduce_time.append(predicted_allreduce_time)
        if weight_sync_method == 'sequential':
            iter_time = (fwd_time + bwd_time) * (accum_step-1) + fwd_time + \
                max(bwd_time, sum(all_bucket_allreduce_time[:-1])) + \
                all_bucket_allreduce_time[-1] + \
                self.update_time + (self.copy_time_per_model * (num_models-1))

            gbs = self.lbs * num_models * accum_step * resource_info.num_gpus_in_server
            predicted_thp = round(gbs / iter_time, 2)
            if self.verbose:
                print_one_rank(f'iter time = {iter_time:.4f}')
                print_one_rank(f'GBS = {gbs}')
                print_one_rank(f'Predicted throughput of GBS: {gbs} = {predicted_thp:.4f}')

                # In-depth analysis
                print_one_rank(f'========================== Breakdown ========================== \n' \
                    f'1) GA + fwd: {(fwd_time + bwd_time) * (accum_step-1) + fwd_time:.4f} \n' \
                    f'2) overlap bwd + allreduce: {max(bwd_time, sum(all_bucket_allreduce_time[:-1])):.4f} \n' \
                    f'\t2-1) bwd: {bwd_time} 2-2) allreduce: {sum(all_bucket_allreduce_time[:-1]):.4f} \n' \
                    f'3) allreduce of last bucket: {all_bucket_allreduce_time[-1]:.4f} \n' \
                    f'4) update: {self.update_time:.4f} \n'
                    f'5) copy: {self.copy_time_per_model * (num_models-1):.4f} \n' \
                    f'===============================================================')
            comp_data = {
                'GA': (fwd_time + bwd_time) * (accum_step-1),
                'fwd': fwd_time,
                'bwd': bwd_time,
            }
            sync_data = {
                'overlap': sum(all_bucket_allreduce_time[:-1]),
                'last_bucket': all_bucket_allreduce_time[-1],
                'update': self.update_time,
                'copy': self.copy_time_per_model * (num_models-1)
            }
            if self.plot_dir:
                self.plot_evaluate_breakdown(
                    comp_data, sync_data,
                    f'{socket.gethostname()}_{self.model}_{self.lbs}_{num_models}_{accum_step}_{weight_sync_method}.png',
                    f'{self.model_name} ({self.lbs}) VSW: {num_models} GA: {accum_step-1} weight sync: {weight_sync_method} on {self.gpu_type}'
                )

        elif weight_sync_method == 'overlap':
            last_bucket_size_ratio = self.bucket_size_distribution[-1] / sum(self.bucket_size_distribution)
            if self.verbose:
                print_one_rank(f'[INFO] last_bucket_size_ratio: {last_bucket_size_ratio}')
            iter_time = (fwd_time + bwd_time) * (accum_step-1) + fwd_time + \
                max(bwd_time, sum(all_bucket_allreduce_time[:-1])) + \
                all_bucket_allreduce_time[-1] + \
                (self.update_time + (self.copy_time_per_model * (num_models-1))) * last_bucket_size_ratio

            gbs = self.lbs * num_models * accum_step * resource_info.num_gpus_in_server
            predicted_thp = round(gbs / iter_time, 2)
            if self.verbose:
                print_one_rank(f'iter time = {iter_time:.4f}')
                print_one_rank(f'GBS = {gbs}')
                print_one_rank(f'Predicted throughput of GBS: {gbs} = {predicted_thp:.4f}')

                # In-depth analysis
                print_one_rank(f'========================== Breakdown ========================== \n' \
                    f'1) GA + fwd: {(fwd_time + bwd_time) * (accum_step-1) + fwd_time:.4f} \n' \
                    f'2) overlap bwd + allreduce: {max(bwd_time, sum(all_bucket_allreduce_time[:-1])):.4f} \n' \
                    f'\t2-1) bwd: {bwd_time} 2-2) allreduce: {sum(all_bucket_allreduce_time[:-1]):.4f} \n' \
                    f'3) allreduce of last bucket: {all_bucket_allreduce_time[-1]:.4f} \n' \
                    f'4) update: {self.update_time*last_bucket_size_ratio:.4f} \n'
                    f'5) copy: {(self.copy_time_per_model * (num_models-1))*last_bucket_size_ratio:.4f} \n' \
                    f'===============================================================')
            comp_data = {
                'GA': (fwd_time + bwd_time) * (accum_step-1),
                'fwd': fwd_time,
                'bwd': bwd_time,
            }
            sync_data = {
                'overlap': sum(all_bucket_allreduce_time[:-1]),
                'last_bucket': all_bucket_allreduce_time[-1],
                'update': self.update_time*last_bucket_size_ratio,
                'copy': (self.copy_time_per_model * (num_models-1))*last_bucket_size_ratio
            }
            if self.plot_dir:
                self.plot_evaluate_breakdown(
                    comp_data, sync_data,
                    f'{socket.gethostname()}_{self.model}_{self.lbs}_{num_models}_{accum_step}_{weight_sync_method}.png',
                    f'{self.model_name} ({self.lbs}) VSW: {num_models} GA: {accum_step-1} weight sync: {weight_sync_method} on {self.gpu_type}'
                )
        else:
            raise ValueError(f'Not support such weight sync method: {weight_sync_method}')

        return iter_time, predicted_thp
