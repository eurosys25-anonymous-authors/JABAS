# Defined in torch/csrc/onnx/init.cpp

from enum import Enum

PYTORCH_ONNX_CAFFE2_BUNDLE: bool
IR_VERSION: int
PRODUCER_VERSION: str

class TensorProtoDataType(Enum):
    UNDEFINED = ...
    FLOAT = ...
    UINT8 = ...
    INT8 = ...
    UINT16 = ...
    INT16 = ...
    INT32 = ...
    INT64 = ...
    STRING = ...
    BOOL = ...
    FLOAT16 = ...
    DOUBLE = ...
    UINT32 = ...
    UINT64 = ...
    COMPLEX64 = ...
    COMPLEX128 = ...

class OperatorExportTypes(Enum):
    ONNX = ...
    ONNX_ATEN = ...
    ONNX_ATEN_FALLBACK = ...
    RAW = ...
    ONNX_FALLTHROUGH = ...

class TrainingMode(Enum):
    EVAL = ...
    PRESERVE = ...
    TRAINING = ...