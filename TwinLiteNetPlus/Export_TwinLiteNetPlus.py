import sys
import cv2
import time
import torch
import numpy as np
import onnxruntime
import matplotlib.pyplot as plt


project_path = '/home/DakeQQ/Downloads/TwinLiteNetPlus-main'                             # The project folder path. https://github.com/chequanghuy/TwinLiteNetPlus
model_path = '/home/DakeQQ/Downloads/TwinLiteNetPlus-main/large.pth'                     # The target TwinLiteNetPlus model. Only support the TwinLiteNetPlus series: "nano", "small", "medium", "large".
onnx_model_A = '/home/DakeQQ/Downloads/TwinLiteNetPlus_ONNX/TwinLiteNetPlus.onnx'        # The exported onnx model path.
test_image_path = './test.jpg'                                                           # The test input after the export process.

INPUT_IMAGE_SIZE = [720, 1280]                                                           # The input image [Height, Width]
IMAGE_RESIZE = [360, 640]                                                                # Resize the image [Height, Width] for executing speed.


if project_path not in sys.path:
    sys.path.append(project_path)
from model.model import TwinLiteNetPlus


class Config:
    def __init__(self):
        self.weight = model_path
        self.source = test_image_path
        self.config = model_path.split('/')[-1].split('.')[-2]


class TWIN_LITE_NET_PLUS(torch.nn.Module):
    def __init__(self, model, input_image_size, image_resize):
        super(TWIN_LITE_NET_PLUS, self).__init__()
        self.model = model
        self.input_image_size = input_image_size
        self.image_resize = image_resize
        self.inv_255 = 1.0 / 255.0
        pad_len = self.image_resize[0] // 8
        if pad_len & 1 == 0:
            self.pad_len = 16
        else:
            self.pad_len = 12
        self.pad_zero = torch.zeros([1, 3, self.pad_len, image_resize[1]], dtype=torch.float32)

    def forward(self, image):
        image = image.float()
        if self.input_image_size != self.image_resize:
            image = torch.nn.functional.interpolate(
                image,
                self.image_resize,
                mode="bilinear",
                align_corners=False,
                antialias=False
            )
        image = image * self.inv_255
        image = torch.cat([self.pad_zero, image, self.pad_zero], dim=2)
        area, lane = self.model(image)
        lane = lane[:, :, self.pad_len:-self.pad_len]
        area = area[:, :, self.pad_len:-self.pad_len]
        if self.input_image_size != self.image_resize:
            area = torch.nn.functional.interpolate(
                area,
                self.input_image_size,
                mode="bilinear",
                align_corners=False,
                antialias=False
            )
            lane = torch.nn.functional.interpolate(
                lane,
                self.input_image_size,
                mode="bilinear",
                align_corners=False,
                antialias=False
            )
        _, area = torch.max(area, 1)
        _, lane = torch.max(lane, 1)
        return area.to(torch.uint8), lane.to(torch.uint8)


print('\nExport Start.')
with torch.inference_mode():
    config = Config()
    model = TwinLiteNetPlus(config)
    model.load_state_dict(torch.load(config.weight, map_location='cpu'), strict=False)
    model.eval().float()
    model = TWIN_LITE_NET_PLUS(model, INPUT_IMAGE_SIZE, IMAGE_RESIZE)
    image = torch.ones([1, 3, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]], dtype=torch.uint8)

    torch.onnx.export(
        model,
        (image,),
        onnx_model_A,
        input_names=['image'],
        output_names=['area', 'lane'],
        do_constant_folding=True,
        opset_version=17,
        dynamo=False
    )
    del model
    del image
    del config

    if project_path in sys.path:
        sys.path.remove(project_path)

    print('\nExport Done.')


# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4              # Error level, it an adjustable value.
session_opts.inter_op_num_threads = 0            # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 0            # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True         # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")
session_opts.add_session_config_entry("session.qdq_matmulnbits_accuracy_level", "4")
session_opts.add_session_config_entry("optimization.enable_gelu_approximation", "1")
session_opts.add_session_config_entry("disable_synchronize_execution_providers", "1")
session_opts.add_session_config_entry("optimization.minimal_build_optimizations", "")
session_opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")


ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=['CPUExecutionProvider'], provider_options=None)
print(f"\nUsable Providers: {ort_session_A.get_providers()}")
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A = [i.name for i in in_name_A]
out_name_A = [i.name for i in out_name_A]


# Load the raw image using OpenCV
raw_img = cv2.imread(test_image_path)
if raw_img is None:
    print(f"Error: Could not read image at {test_image_path}")
    sys.exit()

# The model expects RGB, but OpenCV loads images in BGR format, so we convert.
rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

# Resize the image to the exact size the model expects.
# cv2.resize expects (width, height).
resized_img = cv2.resize(rgb_img, (ort_session_A._inputs_meta[0].shape[-1], ort_session_A._inputs_meta[0].shape[-2]))

# The model's ONNX graph expects a uint8 tensor with the shape (N, C, H, W).
# 1. Add a batch dimension: (H, W, C) -> (1, H, W, C)
# 2. Change layout to (1, C, H, W)
image_tensor = np.expand_dims(resized_img, axis=0).transpose(0, 3, 1, 2)

print(f"\nInput tensor prepared with shape: {image_tensor.shape} and dtype: {image_tensor.dtype}")

# --- Run Inference ---
input_feed_A = {
    in_name_A[0]: onnxruntime.OrtValue.ortvalue_from_numpy(image_tensor, 'cpu', 0)
}


print(f"\nRunning inference on ONNX model...")
start = time.time()
onnx_result = ort_session_A.run_with_ort_values(out_name_A, input_feed_A)
print(f'\nTime cost: {time.time() - start:.3f} seconds')

# The output is a list containing one array. We extract the array.
area = onnx_result[0].numpy()
lane = onnx_result[1].numpy()

# The output depth map has a shape of (1, H, W), so we remove the batch dimension.
area = np.squeeze(area)
lane = np.squeeze(lane)

# =================================================================================
# 3. VISUALIZE THE RESULT (Method 2: Matplotlib)
# =================================================================================

print("\nVisualizing results using Matplotlib...")

# Create overlay with opaque colors
resized_img = resized_img.astype(np.int16)

# Add the masks
resized_img[..., 1] += area * 255  # Green channel
resized_img[..., 0] += lane * 255  # Red channel

# Clip and convert back to uint8
resized_img = np.clip(resized_img, 0, 255).astype(np.uint8)

plt.figure(figsize=(14, 7))
plt.imshow(resized_img)
plt.title('Green: Driable Area, Red: Lane')
plt.axis('off')
plt.show()

