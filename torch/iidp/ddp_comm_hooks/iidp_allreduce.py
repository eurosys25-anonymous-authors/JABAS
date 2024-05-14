import torch
import torch.distributed as dist


class IIDPState:
    def __init__(self, process_group, total_num_models):
        self.process_group = process_group
        self.total_num_models = total_num_models


def iidp_allreduce_hook(
    state: IIDPState, bucket: dist._GradBucket
) -> torch.futures.Future:
    group_to_use = state.process_group if state.process_group is not None else dist.group.WORLD
    tensor = bucket.get_tensors()[0]
    fut = dist.all_reduce(tensor, group=group_to_use, async_op=True).get_future()

    def get_average_value(fut):
        #print(f'[DEBUG] get_average_value - {state.total_num_models}')
        return [fut.value()[0].div_(state.total_num_models)]

    return fut.then(get_average_value)


def dummy_hook(state, bucket):
    fut = torch.futures.Future()
    #if dist.get_rank():
    #    print(f'[DEBUG] [torch/iidp/ddp_comm_hooks/__init__.py] dummy_hook!')
    fut.set_result(bucket.get_tensors())
    return fut