import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

from torch.iidp.data.sampler import ImbalancedSampler
from torch.iidp.trainer import GLOBAL_TRAINER_STATE, LOCAL_TRAINER_STATE


class DataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, batch_fn=None, loading_once=None, shuffle=False, **kwargs):
        if kwargs.get("batch_sampler") is not None:
            sampler = getattr(kwargs.get("batch_sampler"), 'sampler', None)
        elif kwargs.get("sampler") is not None:
            sampler = kwargs.get("sampler")
        else:
            sampler = None
        if sampler is None or type(sampler) is torch.utils.data.distributed.DistributedSampler:
            if dist.is_initialized():
                imblanced_sampler = ImbalancedSampler(
                    dataset, partition_size=GLOBAL_TRAINER_STATE.partition_size)
                if kwargs.get("batch_sampler") is not None:
                    kwargs.get("batch_sampler").sampler = imblanced_sampler
                if kwargs.get("sampler") is not None:
                    kwargs["sampler"] = imblanced_sampler
        if (kwargs.get("num_workers") is not None and kwargs.get("num_workers") > 0) or \
            (kwargs.get("persistent_workers") is not None and kwargs.get("persistent_workers") is True):
            persistent_workers = True
        else:
            persistent_workers = False
        super().__init__(dataset, batch_size, shuffle=shuffle, persistent_workers=persistent_workers, **kwargs)
        self.initial_dataloader_length = super().__len__() # Equal dataloader length among all ranks
        if batch_fn is None:
            raise ValueError(f'Argument "batch_fn" must be configured by user, but: {batch_fn}')
        if loading_once is None:
            raise ValueError(f'Argument "loading_once" must be configured by user, but: {batch_fn}')
        self.batch_fn = batch_fn
        self.loading_once = loading_once
        self.global_batch_size = GLOBAL_TRAINER_STATE.global_batch_size
        self.total_local_num_models = LOCAL_TRAINER_STATE.num_models
        self.accum_step = LOCAL_TRAINER_STATE.accum_step
        self.data_index = -1
        self.done = False

    def __iter__(self):
        self.data_index = 0
        self.done = False
        print(f'[INFO][torch/iidp/data/dataloader.py] Initial loading.. it might take time..')
        if self.loading_once is True:
            for idx, batch in enumerate(super().__iter__()):
                chunked_batch = self.batch_fn(batch, self.total_local_num_models, self.loading_once)
                yield chunked_batch
            self.done = True
        else:
            # NOTE: Since self._index_sampler.batch_size is changed to local batch size,
            # len(super().__iter__()) is also modified.
            self._index_sampler.batch_size = LOCAL_TRAINER_STATE.local_batch_size
            local_batch_data = []
            padding_samples = []
            iter_len = len(self.batch_sampler)
            for idx, batch in enumerate(super().__iter__()):
                last_batch = (idx == iter_len-1)
                if idx < self.total_local_num_models:
                    padding_samples.append(batch)
                if len(local_batch_data) < self.total_local_num_models:
                    local_batch_data.append(batch)
                    # Handle the case that the last batch is reached, but not satisfied with total_local_num_models
                    if last_batch and not (len(local_batch_data) == self.total_local_num_models):
                        padding_num_local_models = self.total_local_num_models - len(local_batch_data)
                        for i in range(padding_num_local_models):
                            local_batch_data.append(padding_samples[i])
                        padding_samples = []
                if len(local_batch_data) == self.total_local_num_models:
                    #assert len(local_batch_data) == total_local_num_models
                    chunked_batch = self.batch_fn(local_batch_data, self.total_local_num_models, self.loading_once)
                    yield chunked_batch
                    local_batch_data = []
                    if idx % self.accum_step == 0:
                        self.data_index += self.global_batch_size
                        #print(f'[DEBUG][torch/iidp/data/dataloader.py] data_index/len(dataset): {self.data_index}/{len(self.dataset)}')
                if self.data_index >= len(self.dataset):
                    self.done = True
                    break
            self.done = True

        if self.done is False:
            raise RuntimeError(f'[ERROR][torch/iidp/data/dataloader.py] Flag done is not True even iterator is finished')

    def __len__(self):
        return self.initial_dataloader_length


class AdaptiveDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, batch_fn=None, size_fn=None, loading_once=None, shuffle=False, **kwargs):
        if kwargs.get("batch_sampler") is not None:
            sampler = getattr(kwargs.get("batch_sampler"), 'sampler', None)
        elif kwargs.get("sampler") is not None:
            sampler = kwargs.get("sampler")
        else:
            sampler = None
        if sampler is None or type(sampler) is torch.utils.data.distributed.DistributedSampler:
            if dist.is_initialized():
                imblanced_sampler = ImbalancedSampler(
                    dataset, partition_size=GLOBAL_TRAINER_STATE.partition_size)
                if kwargs.get("batch_sampler") is not None:
                    kwargs.get("batch_sampler").sampler = imblanced_sampler
                if kwargs.get("sampler") is not None:
                    kwargs["sampler"] = imblanced_sampler
        if (kwargs.get("num_workers") is not None and kwargs.get("num_workers") > 0) or \
            (kwargs.get("persistent_workers") is not None and kwargs.get("persistent_workers") is True):
            persistent_workers = True
        else:
            persistent_workers = False
        super().__init__(dataset, batch_size, shuffle=shuffle, persistent_workers=persistent_workers, **kwargs)
        self.initial_dataloader_length = super().__len__() # Equal dataloader length among all ranks
        if batch_fn is None:
            raise ValueError(f'Argument "batch_fn" must be configured by user, but: {batch_fn}')
        if size_fn is None:
            raise ValueError(f'Argument "size_fn" must be configured by user, but: {size_fn}')
        if loading_once is None:
            raise ValueError(f'Argument "loading_once" must be configured by user, but: {batch_fn}')
        if loading_once is True: # TODO
            raise ValueError(f'Not support with Argument "loading_once" = True')
        self.batch_fn = batch_fn
        self.size_fn = size_fn
        self.loading_once = loading_once
        self.global_batch_size = GLOBAL_TRAINER_STATE.global_batch_size
        self.total_local_num_models = LOCAL_TRAINER_STATE.num_models
        self.accum_step = LOCAL_TRAINER_STATE.accum_step
        self.data_index = -1
        self.done = False
        self.epoch = 0

    def __iter__(self):
        self.data_index = 0
        self.done = False
        iter_idx = 0
        num_yielded = 0
        if self.loading_once is True:
            while not self.done:
                print(f'[INFO][torch/iidp/data/dataloader.py] rank: {dist.get_rank()} | Initial loading.. it might take time..')
                for idx, batch in enumerate(super().__iter__()):
                    # NOTE: index drawn from super().__iter__() is initialized when it is over
                    iter_idx += 1
                    #print(f'[DEBUG][torch/iidp/data/dataloader.py] rank: {dist.get_rank()} | {batch[0].size()} | {self.total_local_num_models}')
                    #assert batch[0].size()[0] == LOCAL_TRAINER_STATE.local_batch_size * self.total_local_num_models, \
                    #    f'[DEBUG][torch/iidp/data/dataloader.py] __iter__ idx: {idx} | rank: {dist.get_rank()} | {batch[0].size()} | {self.total_local_num_models}'
                    chunked_batch = self.batch_fn(batch, self.total_local_num_models, self.loading_once)
                    if chunked_batch == []:
                        print(f'[DEBUG][torch/iidp/data/dataloader.py] rank: {dist.get_rank()} | continue!')
                        continue
                    # NOTE: after yielding, self.global_batch_size and self.accum_step might be changed
                    global_batch_size_progress = self.global_batch_size
                    accum_progress = self.accum_step
                    yield chunked_batch
                    if iter_idx % accum_progress == 0:
                        self.data_index += global_batch_size_progress
                        #print(f'[DEBUG][torch/iidp/data/dataloader.py] data_index/len(dataset): {self.data_index}/{len(self.dataset)}')
                    if self.data_index >= len(self.dataset):
                        self.done = True
                        break
        else:
            # NOTE: Since self._index_sampler.batch_size is changed to local batch size,
            # len(super().__iter__()) is also modified.
            self._index_sampler.batch_size = LOCAL_TRAINER_STATE.local_batch_size
            local_batch_data = []
            while not self.done:
                print(f'[INFO][torch/iidp/data/dataloader.py] rank: {dist.get_rank()} | Initial loading.. it might take time..')
                for idx, batch in enumerate(super().__iter__()):
                    """
                    print(f'[DEBUG][torch/iidp/data/dataloader.py] rank: {dist.get_rank()} | '
                          f'idx: {idx} | iter_idx: {iter_idx} | num_yielded: {num_yielded} | '
                          f'self.size_fn(batch): {self.size_fn(batch)} | self.batch_sampler.batch_size: {self.batch_sampler.batch_size}')
                    """
                    if self.size_fn(batch) != self.batch_sampler.batch_size:
                        """
                        print(f'[DEBUG][torch/iidp/data/dataloader.py] rank: {dist.get_rank()} | '
                              f'idx: {idx} | num_yielded: {num_yielded} | self.size_fn(batch): {self.size_fn(batch)} | '
                              f'self.batch_sampler.batch_size: {self.batch_sampler.batch_size} | '
                              f'skip!!')
                        """
                        continue
                    # NOTE: index drawn from super().__iter__() is initialized when it is over
                    iter_idx += 1
                    if len(local_batch_data) < self.total_local_num_models:
                        """
                        print(f'[DEBUG][torch/iidp/data/dataloader.py] rank: {dist.get_rank()} | '
                              f'idx: {idx} | iter_idx: {iter_idx} | num_yielded: {num_yielded} | '
                              f'self.total_local_num_models: {self.total_local_num_models} | '
                              f'len(local_batch_data): {len(local_batch_data)} | '
                              f'len(local_batch_data) < self.total_local_num_models condition enters!!!!!')
                        """
                        local_batch_data.append(batch)
                    """
                    print(f'[DEBUG][torch/iidp/data/dataloader.py] rank: {dist.get_rank()} | idx: {idx} | iter_idx: {iter_idx} | num_yielded: {num_yielded} | before if len(local_batch_data) == self.total_local_num_models!')
                    print(f'[DEBUG][torch/iidp/data/dataloader.py] rank: {dist.get_rank()} | idx: {idx} | iter_idx: {iter_idx} | num_yielded: {num_yielded} | self.total_local_num_models: {self.total_local_num_models}')
                    print(f'[DEBUG][torch/iidp/data/dataloader.py] rank: {dist.get_rank()} | idx: {idx} | iter_idx: {iter_idx} | num_yielded: {num_yielded} | local_batch_data[0][0].size()[0]: {local_batch_data[0][0].size()[0]}')
                    """
                    if len(local_batch_data) == self.total_local_num_models:
                        # NOTE: after yielding, self.global_batch_size and self.accum_step might be changed
                        global_batch_size_progress = self.global_batch_size
                        accum_progress = self.accum_step
                        chunked_batch = self.batch_fn(local_batch_data, self.total_local_num_models, self.loading_once)
                        yield chunked_batch
                        num_yielded += 1
                        #print(f'[DEBUG][torch/iidp/data/dataloader.py] rank: {dist.get_rank()} | idx: {idx} | iter_idx: {iter_idx} | num_yielded: {num_yielded} | after yielding chunked_batch!')
                        local_batch_data = []
                        #print(f'[DEBUG][torch/iidp/data/dataloader.py] rank: {dist.get_rank()} | idx: {idx} | iter_idx: {iter_idx} | num_yielded: {num_yielded} | accum_progress: {accum_progress}')
                        if num_yielded % accum_progress == 0:
                            self.data_index += global_batch_size_progress
                            num_yielded = 0
                            #print(f'[DEBUG][torch/iidp/data/dataloader.py] ==================> data_index/len(dataset): {self.data_index}/{len(self.dataset)}')
                    if self.data_index >= len(self.dataset):
                        self.done = True
                        break

        if self.done is False:
            raise RuntimeError(f'[ERROR][torch/iidp/data/dataloader.py] Flag done is not True even iterator is finished')
        self.epoch += 1

    def __len__(self):
        return self.initial_dataloader_length

    def get_progress(self):
        return len(self.dataset) * self.epoch + self.data_index

    def state_dict(self):
        return {
            'epoch': self.epoch,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        if hasattr(self._index_sampler.sampler, 'epoch'):
            self._index_sampler.sampler.epoch = self.epoch
        #print(f'[DEBUG][AdaptiveDataLoader] load state dict: {state_dict}')

    def update_local_state(self, batch_size, total_local_num_models, accum_step):
        print(f'[DEBUG][torch/iidp/data/dataloader.py] update_local_state() - '
              f'batch_size: {batch_size} | '
              f'total_local_num_models: {total_local_num_models} | '
              f'accum_step: {accum_step}')
        if self.loading_once is True:
            self.batch_sampler.batch_size = batch_size
        else:
            self.batch_sampler.batch_size = batch_size // total_local_num_models
            self.curr_sampler_iter_len = len(self.batch_sampler)
        self.total_local_num_models = total_local_num_models
        self.accum_step = accum_step

    def update_global_state(self, global_batch_size, partition_size):
        self.global_batch_size = global_batch_size