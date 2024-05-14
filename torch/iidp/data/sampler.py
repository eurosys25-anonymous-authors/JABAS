import math
from typing import TypeVar, Optional, Iterator, List

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist


T_co = TypeVar('T_co', covariant=True)


class ImbalancedSampler(Sampler[T_co]):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, partition_size: List[float] = None) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        if drop_last:
            raise ValueError("Argument drop_last is not supported currently")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.partition_size = partition_size
        self.all_num_samples_in_process_group = [
            math.ceil(len(self.dataset) * partition_size) \
            for partition_size in self.partition_size
        ]
        print('[INFO][torch/iidp/data/sampler.py] == ImbalancedSampler() == ')
        print(f'[INFO][torch/iidp/data/sampler.py] rank: {rank}')
        print(f'[INFO][torch/iidp/data/sampler.py] num_replicas: {num_replicas}')
        print(f'[INFO][torch/iidp/data/sampler.py] dataset length: {len(self.dataset)}')
        print(f'[INFO][torch/iidp/data/sampler.py] partition_size: {self.partition_size}')
        #self.num_samples = math.ceil(len(self.dataset) * self.partition_size)
        self.num_samples = self.all_num_samples_in_process_group[rank]
        self.total_size = len(self.dataset)
        print(f'[INFO][torch/iidp/data/sampler.py] self.num_samples: {self.num_samples}')
        print(f'[INFO][torch/iidp/data/sampler.py] self.total_size: {self.total_size}')
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
        else:
            indices = list(range(len(self.dataset)))  # type: ignore
        #print('[DEBUG][torch/iidp/data/sampler.py] == ImbalancedSampler().__iter__ == ')
        #print(f'[DEBUG][torch/iidp/data/sampler.py] len(indices): {len(indices)}')
        # subsample
        start_index = sum(self.all_num_samples_in_process_group[:self.rank])
        end_index = start_index + self.num_samples
        #print(f'[DEBUG][torch/iidp/data/sampler.py] subsampling index at rank: {self.rank} ==> {start_index} : {end_index}')
        #indices = indices[self.rank*self.num_samples:(self.rank + 1)*self.num_samples]
        indices = indices[start_index:end_index]
        #print('[DEBUG][torch/iidp/data/sampler.py] == after sub-sampling ==')
        #print(f'[DEBUG][torch/iidp/data/sampler.py] len(indices): {len(indices)}')
        #print(f'[DEBUG][torch/iidp/data/sampler.py] self.num_samples: {self.num_samples}')
        if len(indices) != self.num_samples:
            # add extra samples to make it evenly divisible
            padding_size = self.num_samples - len(indices)
            #print(f'[DEBUG][torch/iidp/data/sampler.py] padding_size: {padding_size}')
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]

        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        # print('[DEBUG][torch/iidp/data/sampler.py] == ImbalancedSampler() == ')
        # print(f'[DEBUG][torch/iidp/data/sampler.py] self.num_samples: {self.num_samples}')
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch