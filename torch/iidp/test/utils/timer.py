import time
from contextlib import ContextDecorator

import torch.distributed as dist


class Timer(ContextDecorator):
    def __init__(self, msg, verbose=True):
        self.msg = msg
        self.verbose = verbose
        self.elapsed = 0

    def __enter__(self):
        self.time = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = time.time() - self.time
        if self.verbose:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    print(f'{self.msg} takes {self.elapsed:.3f} sec')
            else:
                print(f'{self.msg} takes {self.elapsed:.3f} sec')