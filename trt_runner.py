"""TensorRT inference using PyTorch-owned CUDA buffers, without host copies."""

from pathlib import Path

import tensorrt as trt
import torch


LOGGER = trt.Logger(trt.Logger.WARNING)


class TensorRTRunner:
    """One execution context and one stream; use from a single host thread.

    Call with a contiguous CUDA NCHW tensor on the runner's device. Input work
    must be ordered on the caller's current stream. The runner waits for that
    stream, enqueues inference, then makes the caller's stream wait for logits.
    The return is asynchronous on the CPU, and safe to consume on that stream.
    For another producer/consumer stream, the caller must insert wait_stream().
    Keep this runner alive until synchronize() completes; do not mutate input
    or reuse an output buffer while its inference/consumers are still pending.
    """

    def __init__(self, engine_path, device="cuda:0", stream=None):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("A CUDA device is required")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        with torch.cuda.device(self.device):
            self.stream = stream if stream is not None else torch.cuda.Stream(device=self.device)
            if self.stream.device != self.device:
                raise ValueError("Stream and runner must use the same CUDA device")
            self.logger = LOGGER
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
            if self.engine is None:
                raise RuntimeError(f"Could not deserialize engine: {engine_path}")
            if self.engine.num_io_tensors != 2 or self.engine.num_optimization_profiles != 1:
                raise ValueError("Expected two I/O tensors and one optimization profile")
            for name, mode, shape in (
                ("input", trt.TensorIOMode.INPUT, (-1, 3, 224, 224)),
                ("logits", trt.TensorIOMode.OUTPUT, (-1, 1000)),
            ):
                if (self.engine.get_tensor_mode(name) != mode
                        or tuple(self.engine.get_tensor_shape(name)) != shape
                        or self.engine.get_tensor_location(name) != trt.TensorLocation.DEVICE
                        or self.engine.get_tensor_format(name) != trt.TensorFormat.LINEAR):
                    raise ValueError(f"Unsupported tensor contract: {name}")
            dtypes = {trt.float32: torch.float32, trt.float16: torch.float16}
            self.input_dtype = dtypes[self.engine.get_tensor_dtype("input")]
            self.output_dtype = dtypes[self.engine.get_tensor_dtype("logits")]
            self.profile = tuple(tuple(s) for s in self.engine.get_tensor_profile_shape("input", 0))
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("Could not create the execution context")
        self._shape = None

    def __call__(self, x, out=None):
        if x.device != self.device or x.dtype != self.input_dtype or not x.is_contiguous():
            raise ValueError(f"Input must be contiguous {self.input_dtype} on {self.device}")
        shape = tuple(x.shape)
        if (len(shape) != 4 or shape[1:] != (3, 224, 224)
                or not self.profile[0][0] <= shape[0] <= self.profile[2][0]):
            raise ValueError(f"Expected N x 3 x 224 x 224 within profile {self.profile}")
        with torch.cuda.device(self.device):
            caller = torch.cuda.current_stream(self.device)
            if shape != self._shape:
                # Shape changes may update context memory used by prior inference.
                self.synchronize()
                if not self.context.set_input_shape("input", shape):
                    raise RuntimeError(f"Could not set input shape: {shape}")
                self._shape = shape
            output_shape = tuple(self.context.get_tensor_shape("logits"))
            if output_shape != (shape[0], 1000):
                raise RuntimeError(f"Unexpected logits shape: {output_shape}")
            if out is None:
                out = torch.empty(output_shape, device=self.device, dtype=self.output_dtype)
            if (out.device != self.device or out.dtype != self.output_dtype
                    or tuple(out.shape) != output_shape or not out.is_contiguous()):
                raise ValueError(f"Output must be contiguous {output_shape}, {self.output_dtype}, {self.device}")
            if out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr():
                raise ValueError("Input and output buffers must not share storage")
            if caller != self.stream:
                self.stream.wait_stream(caller)
            for name, tensor in (("input", x), ("logits", out)):
                if not self.context.set_tensor_address(name, tensor.data_ptr()):
                    raise RuntimeError(f"Could not bind {name}")
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT inference failed")
            x.record_stream(self.stream)
            out.record_stream(self.stream)
            if caller != self.stream:
                caller.wait_stream(self.stream)
            return out

    def synchronize(self):
        self.stream.synchronize()
