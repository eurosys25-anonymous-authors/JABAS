from typing import List, Set
from enum import Enum

# Defined in tools/autograd/init.cpp

class ProfilerState(Enum):
    Disable = ...
    CPU = ...
    CUDA = ...
    NVTX = ...
    KINETO = ...

class ProfilerActivity(Enum):
    CPU = ...
    CUDA = ...

class DeviceType(Enum):
    CPU = ...
    CUDA = ...
    ...

class ProfilerConfig:
    def __init__(
        self,
        state: ProfilerState,
        report_input_shapes: bool,
        profile_memory: bool,
        with_stack: bool,
        with_flops: bool
    ) -> None: ...
    ...

class ProfilerEvent:
    def cpu_elapsed_us(self, other: ProfilerEvent) -> float: ...
    def cpu_memory_usage(self) -> int: ...
    def cuda_elapsed_us(self, other: ProfilerEvent) -> float: ...
    def cuda_memory_usage(self) -> int: ...
    def device(self) -> int: ...
    def handle(self) -> int: ...
    def has_cuda(self) -> bool: ...
    def is_remote(self) -> bool: ...
    def kind(self) -> int: ...
    def name(self) -> str: ...
    def node_id(self) -> int: ...
    def sequence_nr(self) -> int: ...
    def shapes(self) -> List[List[int]]: ...
    def thread_id(self) -> int: ...
    def flops(self) -> float: ...
    ...

class KinetoEvent:
    def name(self) -> str: ...
    def device_index(self) -> int: ...
    def start_us(self) -> int: ...
    def duration_us(self) -> int: ...
    ...

class ProfilerResult:
    def events(self) -> List[KinetoEvent]: ...
    def legacy_events(self) -> List[List[ProfilerEvent]]: ...
    def save(self, str) -> None: ...

def _enable_profiler(config: ProfilerConfig, activities: Set[ProfilerActivity]) -> None: ...
def _prepare_profiler(config: ProfilerConfig, activities: Set[ProfilerActivity]) -> None: ...
def _disable_profiler() -> ProfilerResult: ...
def _profiler_enabled() -> bool: ...
def kineto_available() -> bool: ...
def _enable_record_function(enable: bool) -> None: ...
def _set_empty_test_observer(is_global: bool, sampling_prob: float) -> None: ...

def _enable_profiler_legacy(config: ProfilerConfig) -> None: ...
def _disable_profiler_legacy() -> List[List[ProfilerEvent]]: ...

