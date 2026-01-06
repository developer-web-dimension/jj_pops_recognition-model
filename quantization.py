import onnx
from onnxconverter_common import float16
import os

FP32_MODEL = "jimjam_classifier_2_classes.onnx"
FP16_MODEL = "jimjam_fp16_2_classes_2.onnx"

# Load the model
model = onnx.load(FP32_MODEL)

# Convert to FP16
fp16_model = float16.convert_float_to_float16(model, keep_io_types=True)

# Save the FP16 ONNX
onnx.save(fp16_model, FP16_MODEL)

print("✔ FP16 model saved as:", FP16_MODEL)

# Show size comparison
fp32_size = os.path.getsize(FP32_MODEL) / 1024 / 1024
fp16_size = os.path.getsize(FP16_MODEL) / 1024 / 1024

print(f"FP32 Size : {fp32_size:.2f} MB")
print(f"FP16 Size : {fp16_size:.2f} MB")
print(f"Compression Ratio: {fp32_size/fp16_size:.2f}x smaller")
