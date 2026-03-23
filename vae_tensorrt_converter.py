import torch
import os
import time
import comfy.model_management
import tensorrt as trt
import folder_paths
from .trt_common import TRT_MODEL_CONVERSION_BASE, logger, TQDMProgressMonitor

class SDXL_VAE_TENSORRT_CONVERTER(TRT_MODEL_CONVERSION_BASE):
    def __init__(self):
        super(SDXL_VAE_TENSORRT_CONVERTER, self).__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI_SDXL_VAE"}),
                "batch_size_min": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "batch_size_opt": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "batch_size_max": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "height_min": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "height_opt": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "height_max": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "width_min": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "width_opt": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "width_max": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
            },
        }

    def convert(self, vae, filename_prefix, batch_size_min, batch_size_opt, batch_size_max, height_min, height_opt, height_max, width_min, width_opt, width_max):
        output_onnx = os.path.normpath(
            os.path.join(
                os.path.join(self.temp_dir, "{}".format(time.time())), "vae.onnx"
            )
        )
        dtype = torch.float16
        input_names = ["samples"]
        output_names = ["image"]
        dynamic_axes = {"samples": {0: "batch", 2: "height", 3: "width"}}
        
        class VAEDecoder(torch.nn.Module):
            def __init__(self, first_stage_model):
                super().__init__()
                self.first_stage_model = first_stage_model
            def forward(self, samples):
                return self.first_stage_model.decode(samples)

        _vae = VAEDecoder(vae.first_stage_model).to(dtype=dtype, device=comfy.model_management.get_torch_device())
        inputs_shapes_min = ((batch_size_min, 4, height_min // 8, width_min // 8),)
        inputs_shapes_opt = ((batch_size_opt, 4, height_opt // 8, width_opt // 8),)
        inputs_shapes_max = ((batch_size_max, 4, height_max // 8, width_max // 8),)
        inputs = (torch.zeros(inputs_shapes_opt[0], device=comfy.model_management.get_torch_device(), dtype=dtype),)

        os.makedirs(os.path.dirname(output_onnx), exist_ok=True)
        torch.onnx.export(_vae, inputs, output_onnx, verbose=False, input_names=input_names, output_names=output_names, opset_version=17, dynamic_axes=dynamic_axes)
        comfy.model_management.soft_empty_cache()

        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        success = parser.parse_from_file(output_onnx)
        if not success:
            print("ONNX load ERROR")
            return ()

        config = builder.create_builder_config()
        profile = builder.create_optimization_profile()
        self._setup_timing_cache(config)
        config.progress_monitor = TQDMProgressMonitor()

        for k in range(len(input_names)):
            profile.set_shape(input_names[k], inputs_shapes_min[k], inputs_shapes_opt[k], inputs_shapes_max[k])

        config.set_flag(trt.BuilderFlag.FP16)
        config.add_optimization_profile(profile)

        filename_prefix = "{}_${}".format(filename_prefix, "-".join(("vae", "dyn", "b", str(batch_size_min), str(batch_size_max), str(batch_size_opt), "h", str(height_min), str(height_max), str(height_opt), "w", str(width_min), str(width_max), str(width_opt))))
        serialized_engine = builder.build_serialized_network(network, config)
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        output_trt_engine = os.path.join(full_output_folder, f"{filename}_{counter:05}_.engine")

        with open(output_trt_engine, "wb") as f:
            f.write(serialized_engine)
        self._save_timing_cache(config)
        return ()

NODE_CLASS_MAPPINGS = {
    "SDXL_VAE_TENSORRT_CONVERTER": SDXL_VAE_TENSORRT_CONVERTER,
}
