import torch
import os
import comfy.model_management
import folder_paths
from .trt_common import runtime, trt_datatype_to_torch

class TrtVaeDecoder:
    def __init__(self, engine_path):
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def set_bindings_shape(self, inputs, split_batch):
        for k in inputs:
            shape = inputs[k].shape
            shape = [shape[0] // split_batch] + list(shape[1:])
            self.context.set_input_shape(k, shape)

    def decode(self, samples):
        model_inputs = {"samples": samples}
        batch_size = samples.shape[0]
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

        output_binding_name = self.engine.get_tensor_name(1)
        out_shape_dims = self.engine.get_tensor_shape(output_binding_name)
        out_shape = [out_shape_dims[i] for i in range(len(out_shape_dims))]

        for idx in range(len(out_shape)):
            if out_shape[idx] == -1:
                if idx == 0:
                    out_shape[idx] = samples.shape[0] // curr_split_batch
                elif idx == 2:
                    out_shape[idx] = samples.shape[2] * 8
                elif idx == 3:
                    out_shape[idx] = samples.shape[3] * 8
            else:
                if idx == 0:
                    out_shape[idx] *= curr_split_batch

        out = torch.empty(out_shape, device=samples.device, dtype=trt_datatype_to_torch(self.engine.get_tensor_dtype(output_binding_name)))
        model_inputs_converted[output_binding_name] = out

        stream = torch.cuda.default_stream(samples.device)
        for i in range(curr_split_batch):
            for k in model_inputs_converted:
                tensor = model_inputs_converted[k]
                self.context.set_tensor_address(k, tensor[(tensor.shape[0] // curr_split_batch) * i:].data_ptr())
            self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        return out

class SDXL_VAE_TENSORRT_LOADER_DECODER:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae_name": (folder_paths.get_filename_list("tensorrt"), ),
                             }}
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "TensorRT"

    def load_vae(self, vae_name):
        vae_path = folder_paths.get_full_path("tensorrt", vae_name)
        if not vae_path or not os.path.isfile(vae_path):
            raise FileNotFoundError(f"File {vae_name} does not exist")
        trt_vae = TrtVaeDecoder(vae_path)
        import comfy.sd
        new_vae = comfy.sd.VAE(sd={})
        class FirstStageModelWrapper:
            def __init__(self, trt_vae):
                self.trt_vae = trt_vae
                self.device = comfy.model_management.get_torch_device()
            def decode(self, z):
                return self.trt_vae.decode(z)
            def encode(self, x):
                return None
        new_vae.first_stage_model = FirstStageModelWrapper(trt_vae)
        return (new_vae,)

NODE_CLASS_MAPPINGS = {
    "SDXL_VAE_TENSORRT_LOADER_DECODER": SDXL_VAE_TENSORRT_LOADER_DECODER,
}
