import torch
import os
import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.supported_models
import folder_paths
from .trt_common import runtime, trt_datatype_to_torch

class TrTUnet:
    def __init__(self, engine_path):
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}. The engine might be incompatible with the current hardware or TensorRT version.")
        self.context = self.engine.create_execution_context()
        self.dtype = torch.float16

    def set_bindings_shape(self, inputs, split_batch):
        for k in inputs:
            shape = inputs[k].shape
            shape = [shape[0] // split_batch] + list(shape[1:])
            self.context.set_input_shape(k, shape)

    def __call__(self, x, timesteps, context, y=None, control=None, transformer_options=None, **kwargs):
        model_inputs = {"x": x, "timesteps": timesteps, "context": context}
        if y is not None:
            model_inputs["y"] = y
        for i in range(len(model_inputs), self.engine.num_io_tensors - 1):
            name = self.engine.get_tensor_name(i)
            model_inputs[name] = kwargs[name]

        batch_size = x.shape[0]
        dims = self.engine.get_tensor_profile_shape(self.engine.get_tensor_name(0), 0)
        max_batch = dims[2][0]
        min_batch = dims[0][0]

        curr_split_batch = 1
        for i in range(max_batch, min_batch - 1, -1):
            if i > 0 and batch_size % i == 0:
                curr_split_batch = batch_size // i
                break

        self.set_bindings_shape(model_inputs, curr_split_batch)
        model_inputs_converted = {}
        for k in model_inputs:
            data_type = self.engine.get_tensor_dtype(k)
            model_inputs_converted[k] = model_inputs[k].to(dtype=trt_datatype_to_torch(data_type))

        output_binding_name = self.engine.get_tensor_name(len(model_inputs))
        out_shape_dims = self.engine.get_tensor_shape(output_binding_name)
        out_shape = [out_shape_dims[i] for i in range(len(out_shape_dims))]

        for idx in range(len(out_shape)):
            if out_shape[idx] == -1:
                out_shape[idx] = x.shape[idx]
            else:
                if idx == 0:
                    out_shape[idx] *= curr_split_batch

        out = torch.empty(out_shape, device=x.device, dtype=trt_datatype_to_torch(self.engine.get_tensor_dtype(output_binding_name)))
        model_inputs_converted[output_binding_name] = out

        stream = torch.cuda.default_stream(x.device)
        for i in range(curr_split_batch):
            for k in model_inputs_converted:
                tensor = model_inputs_converted[k]
                self.context.set_tensor_address(k, tensor[(tensor.shape[0] // curr_split_batch) * i:].data_ptr())
            self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        return out

    def load_state_dict(self, sd, strict=False):
        pass
    def state_dict(self):
        return {}

class UNET_TENSORRT_LOADER:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"unet_name": (folder_paths.get_filename_list("tensorrt"), ),
                             "model_type": (["sdxl_base"], ),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "TensorRT"

    def load_unet(self, unet_name, model_type):
        unet_path = folder_paths.get_full_path("tensorrt", unet_name)
        if not unet_path or not os.path.isfile(unet_path):
            raise FileNotFoundError(f"File {unet_path} does not exist")
        unet = TrTUnet(unet_path)
        if model_type == "sdxl_base":
            conf = comfy.supported_models.SDXL({"adm_in_channels": 2816})
            conf.unet_config["disable_unet_model_creation"] = True
            model = comfy.model_base.SDXL(conf)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        model.diffusion_model = unet
        model.memory_required = lambda *args, **kwargs: 0
        return (comfy.model_patcher.ModelPatcher(model,
                                                 load_device=comfy.model_management.get_torch_device(),
                                                 offload_device=comfy.model_management.unet_offload_device()),)

NODE_CLASS_MAPPINGS = {
    "UNET_TENSORRT_LOADER": UNET_TENSORRT_LOADER,
}
