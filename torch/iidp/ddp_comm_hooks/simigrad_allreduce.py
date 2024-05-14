import os

import torch
import torch.distributed as dist
from torch.iidp.ddp_comm_hooks.default_hooks import _allreduce_sum_fut
from torch.iidp.ddp_comm_hooks.iidp_allreduce import iidp_allreduce_hook, IIDPState


class SimiGradState(IIDPState):
    def __init__(self, process_group, total_num_models, sub_group, grad_placeholder, interval):
        super().__init__(process_group, total_num_models)
        self.sub_group = sub_group
        self.grad_placeholder = grad_placeholder
        assert self.total_num_models % 2 == 0, \
            f"self.total_num_models must be power of 2, but {self.total_num_models}"
        self.subgroup_total_num_models = self.total_num_models / 2
        self.interval = interval
        self.step = 1 # To avoid first step because of rebuilding DDP bucket
        if os.getenv('GBS_INTERVAL_AS_EPOCH') == "1":
            self.interval = 0 # determined by remaining_epochs() in torch/iidp/trainer.py
            self.done_epoch = False


def subgroup_allreduce_hook(
    state: SimiGradState, bucket: dist._GradBucket
) -> torch.futures.Future:
    if os.getenv('GBS_INTERVAL_AS_EPOCH') == "1":
        if state.done_epoch is False:
            fut = torch.futures.Future()
            fut.set_result(bucket.get_tensors())
            return fut
    elif state.step % state.interval != 0:
        fut = torch.futures.Future()
        fut.set_result(bucket.get_tensors())
        return fut
    # Detach bucket's tensor not to be affected by all-reduce in all process group (allgroup_allreduce())
    tensor = bucket.get_tensors()[0].detach().clone()
    group_to_use = state.sub_group
    future_work = _allreduce_sum_fut(group_to_use, tensor)
    def append_to_grad_placeholder(fut):
        state.grad_placeholder.append(fut.value()[0].div_(state.subgroup_total_num_models))
        return [fut.value()[0]]

    return future_work.then(append_to_grad_placeholder)


def simigrad_allreduce_hook(hook):
    def hook_with_allreduce(state, bucket):
        future_work = hook(state, bucket)
        def allgroup_allreduce(fut):
            iidp_allreduce_hook(state, bucket).wait()
            return bucket.get_tensors()
        return future_work.then(allgroup_allreduce)
    return hook_with_allreduce
