import torch
import os
import time
import tensorrt as trt
import folder_paths
import comfy.model_management
import modelopt.torch.quantization as mtq # <--- ADDED: Import ModelOpt
from .trt_common import logger, TQDMProgressMonitor

class UNET_TENSORRT_CONVERTER_INT8:
    def __init__(self):
        self.output_dir = os.path.join(folder_paths.models_dir, "tensorrt")
        self.temp_dir = folder_paths.get_temp_directory()
        self.timing_cache_path = os.path.normpath(
            os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "timing_cache.trt"))
        )

    RETURN_TYPES = ()
    FUNCTION = "convert"
    OUTPUT_NODE = True
    CATEGORY = "TensorRT"

    def _setup_timing_cache(self, config: trt.IBuilderConfig):
        buffer = b""
        if os.path.exists(self.timing_cache_path):
            with open(self.timing_cache_path, mode="rb") as timing_cache_file:
                buffer = timing_cache_file.read()
            print("Read {} bytes from timing cache.".format(len(buffer)))
        else:
            print("No timing cache found; Initializing a new one.")
        timing_cache: trt.ITimingCache = config.create_timing_cache(buffer)
        config.set_timing_cache(timing_cache, ignore_mismatch=True)

    def _save_timing_cache(self, config: trt.IBuilderConfig):
        timing_cache: trt.ITimingCache = config.get_timing_cache()
        with open(self.timing_cache_path, "wb") as timing_cache_file:
            timing_cache_file.write(memoryview(timing_cache.serialize()))

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "filename_prefix": ("STRING", {"default": "ComfyUI_SDXL_INT8"}),
                "batch_size_min": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "batch_size_opt": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "batch_size_max": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "height_min": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "height_opt": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "height_max": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "width_min": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "width_opt": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "width_max": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "context_min": ("INT", {"default": 1, "min": 1, "max": 128, "step": 1}),
                "context_opt": ("INT", {"default": 1, "min": 1, "max": 128, "step": 1}),
                "context_max": ("INT", {"default": 1, "min": 1, "max": 128, "step": 1}),
            },
        }

    def convert(self, model, filename_prefix, batch_size_min, batch_size_opt, batch_size_max, height_min, height_opt, height_max, width_min, width_opt, width_max, context_min, context_opt, context_max):
        output_onnx = os.path.normpath(
            os.path.join(
                os.path.join(self.temp_dir, "{}".format(time.time())), "model.onnx"
            )
        )

        comfy.model_management.unload_all_models()
        comfy.model_management.load_models_gpu([model], force_patch_weights=True, force_full_load=True)
        unet = model.model.diffusion_model

        context_dim = model.model.model_config.unet_config.get("context_dim", None)
        context_len = 77
        context_len_min = context_len
        y_dim = model.model.adm_channels
        extra_input = {}
        dtype = torch.float16

        if context_dim is not None:
            input_names = ["x", "timesteps", "context"]
            output_names = ["h"]

            dynamic_axes = {
                "x": {0: "batch", 2: "height", 3: "width"},
                "timesteps": {0: "batch"},
                "context": {0: "batch", 1: "num_embeds"},
            }

            input_channels = model.model.model_config.unet_config.get("in_channels", 4)

            inputs_shapes_min = (
                (batch_size_min, input_channels, height_min // 8, width_min // 8),
                (batch_size_min,),
                (batch_size_min, context_len_min * context_min, context_dim),
            )
            inputs_shapes_opt = (
                (batch_size_opt, input_channels, height_opt // 8, width_opt // 8),
                (batch_size_opt,),
                (batch_size_opt, context_len * context_opt, context_dim),
            )
            inputs_shapes_max = (
                (batch_size_max, input_channels, height_max // 8, width_max // 8),
                (batch_size_max,),
                (batch_size_max, context_len * context_max, context_dim),
            )

            if y_dim > 0:
                input_names.append("y")
                dynamic_axes["y"] = {0: "batch"}
                inputs_shapes_min += ((batch_size_min, y_dim),)
                inputs_shapes_opt += ((batch_size_opt, y_dim),)
                inputs_shapes_max += ((batch_size_max, y_dim),)

            for k in extra_input:
                input_names.append(k)
                dynamic_axes[k] = {0: "batch"}
                inputs_shapes_min += ((batch_size_min,) + extra_input[k],)
                inputs_shapes_opt += ((batch_size_opt,) + extra_input[k],)
                inputs_shapes_max += ((batch_size_max,) + extra_input[k],)

            transformer_options = model.model_options['transformer_options'].copy()
            class UNET(torch.nn.Module):
                def __init__(self, unet, transformer_options, extra_input_names):
                    super().__init__()
                    self.unet = unet
                    self.transformer_options = transformer_options
                    self.extra_input_names = extra_input_names
                def forward(self, x, timesteps, context, *args):
                    extra_args = {}
                    for i in range(len(self.extra_input_names)):
                        extra_args[self.extra_input_names[i]] = args[i]
                    return self.unet(x, timesteps, context, transformer_options=self.transformer_options, **extra_args)

            unet_wrapped = UNET(unet, transformer_options, input_names[3:])

            inputs = ()
            for shape in inputs_shapes_opt:
                inputs += (
                    torch.zeros(
                        shape,
                        device=comfy.model_management.get_torch_device(),
                        dtype=dtype,
                    ),
                )

        else:
            print("ERROR: model not supported.")
            return ()

        os.makedirs(os.path.dirname(output_onnx), exist_ok=True)
        
        # --- MODIFIED EXPORT BLOCK ---
        print("Exporting ONNX with ModelOpt INT8 Q/DQ nodes...")
        with mtq.export_onnx(): # <--- ADDED: Context manager for INT8
            torch.onnx.export(
                unet_wrapped,
                inputs,
                output_onnx,
                verbose=False,
                input_names=input_names,
                output_names=output_names,
                opset_version=17,
                dynamic_axes=dynamic_axes,
                do_constant_folding=False, # <--- CHANGED: False prevents the 12GB VRAM OOM
            )
        # -----------------------------

        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        # TRT conversion starts here
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)
        success = parser.parse_from_file(output_onnx)
        for idx in range(parser.num_errors):
            print(parser.get_error(idx))

        if not success:
            print("ONNX load ERROR")
            return ()

        config = builder.create_builder_config()
        profile = builder.create_optimization_profile()
        self._setup_timing_cache(config)
        config.progress_monitor = TQDMProgressMonitor()

        for k in range(len(input_names)):
            profile.set_shape(input_names[k], inputs_shapes_min[k], inputs_shapes_opt[k], inputs_shapes_max[k])

        # Enable both FP16 and INT8 for the engine
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.INT8) # <--- ADDED: Crucial for INT8 speedup

        if dtype == torch.bfloat16:
            config.set_flag(trt.BuilderFlag.BF16)

        config.add_optimization_profile(profile)

        filename_prefix = "{}_${}".format(
            filename_prefix,
            "-".join(
                (
                    "dyn",
                    "b",
                    str(batch_size_min),
                    str(batch_size_max),
                    str(batch_size_opt),
                    "h",
                    str(height_min),
                    str(height_max),
                    str(height_opt),
                    "w",
                    str(width_min),
                    str(width_max),
                    str(width_opt),
                )
            ),
        )

        serialized_engine = builder.build_serialized_network(network, config)

        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        )
        output_trt_engine = os.path.join(
            full_output_folder, f"{filename}_{counter:05}_.engine"
        )

        with open(output_trt_engine, "wb") as f:
            f.write(serialized_engine)

        self._save_timing_cache(config)

        return ()

NODE_CLASS_MAPPINGS = {
    "UNET_TENSORRT_CONVERTER_INT8": UNET_TENSORRT_CONVERTER_INT8,
}
