import torch 
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort 
from onnxruntime.quantization import(quantize_static, CalibrationDataReader, CalibrationMethod, QuantType, QuantFormat)
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import os
import time
import argparse
CLASSES = ["green", "red", "stop_sign"]
INPUT_SIZE = (1, 3, 96, 96) #batch size, rcb 3 channels, and resolution
IMAGENET_MEAN = [0.485, 0.456, 0.406] 
IMAGENET_STD = [0.229, 0.224, 0.225]
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights=None)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(576, 128),       
            nn.Hardswish(),
            #nn.Dropout(p=dropout); NO DROPOUT cuz we dont need randomness during inference
            nn.Linear(128, len(CLASSES)),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x
#calibration for int8, onnx runtime needs to see real data
class CalibrationReader(CalibrationDataReader):
    def __init__(self, data_dir: str, num_samples: int = 300): #300 random data as sample
        transform = transforms.Compose([transforms.Resize((INPUT_SIZE[2], INPUT_SIZE[3])), transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
        dataset = ImageFolder(data_dir, transform=transform)
        n = min(num_samples, len(dataset))
        targets = np.array(dataset.targets)
        indices = []
        per_class = 100  # 100 per class = 300 total
        for c in range(len(CLASSES)):
            class_idx = np.where(targets == c)[0]
            chosen = np.random.choice(class_idx, min(per_class, len(class_idx)), replace=False)
            indices.extend(chosen)
        subset = torch.utils.data.Subset(dataset, indices)
        self.loader= iter(DataLoader(subset, batch_size=1, shuffle=False))
        print(f"calibrating with {n} images from original dataset")
    
    def get_next(self):
        try:
            images, _= next(self.loader)
            return {"input": images.numpy()}
        except StopIteration:
            return None
def load_model(checkpoint_path: str) -> Classifier:
    model = Classifier()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True) #load model into cpu, no matter if it were trained on a gpu or cpu.
    state = ckpt["model_state_dict"] #get weights
    cleaned={}
    for k, v in state.items():
        if "classifier.3" in k:
            cleaned[k.replace("classifier.3.", "classifier.2.")] = v
            #we removed dropout, and layers in sequential work on indices, hence lower index
        elif "classifier.2." in k:
            continue
        else:
            cleaned[k] = v 
    model.load_state_dict(cleaned)
    model.eval()
    acc = ckpt.get("val_acc", "?")
    print(f"loaded ckpt w acc val={acc}%.2f")
    return model
def export_onnx(model: nn.Module, path: str) -> str:
    torch.onnx.export(model, torch.randn(*INPUT_SIZE), #onnx needs fake input to understand 
                      path, export_params=True,opset_version=13,do_constant_folding=True,input_names=["input"],output_names=["output"],
                    dynamic_axes={
                        "input": {0: "batch"}, #allow any batch size
                         "output": {0: "batch"},
                        }, dynamo=False) 
    onnx.checker.check_model(onnx.load(path))
    print(f"fp32 onnx: {path}, file size: {file_size_mb(path):.2f} MB")
    return path

def quantize(fp32_path: str, int8_path: str, calibration_dir: str) -> str:
    quantize_static(model_input = fp32_path,
        model_output = int8_path,
        calibration_data_reader=CalibrationReader(calibration_dir),
        quant_format = QuantFormat.QDQ, per_channel = True, #quantise->dequantise method used to inject fake quantisation noise
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        calibrate_method=CalibrationMethod.Percentile,
        extra_options={
            "CalibPercentile": 99.9,  # ignore top 0.1% outliers
        },)
    print(f"  INT8 ONNX: {int8_path} ({file_size_mb(int8_path):.2f} MB)")
    return int8_path
def validate(onnx_path: str, val_dir: str) -> float:
    session = make_session(onnx_path)
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE[2], INPUT_SIZE[3])),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    dataset = ImageFolder(val_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    correct = 0
    total = 0
    per_class = {"correct": [0] * len(CLASSES), "total": [0] * len(CLASSES)}

    for images, labels in loader:
        pred = np.argmax(session.run(None, {"input": images.numpy()})[0])
        label = labels.item()
        total += 1
        per_class["total"][label] += 1
        if pred == label:
            correct += 1
            per_class["correct"][label] += 1

    acc = 100.0 * correct / total
    print(f"\n  Accuracy: {acc:.1f}% ({correct}/{total})")
    for i, cls in enumerate(CLASSES):
        if per_class["total"][i] > 0:
            c_acc = 100.0 * per_class["correct"][i] / per_class["total"][i]
            print(f"    {cls:>10}: {c_acc:.1f}%")
    return acc
def benchmark(onnx_path: str, num_runs: int = 200)->float:
    session = make_session(onnx_path)
    dummy = np.random.randn(*INPUT_SIZE).astype(np.float32)

    for _ in range(20):
        session.run(None, {"input": dummy})

    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        session.run(None, {"input": dummy})
        times.append(time.perf_counter() - t0)

    ms = np.array(times) * 1000
    fps = 1000.0 / ms.mean()
    print(f"\n  Latency: {ms.mean():.1f}ms ± {ms.std():.1f}ms | {fps:.0f} FPS")
    return fps
def make_session(path: str)-> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4 #standard for raspi4
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"]) #force use of cpu
def file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)
##

def main(checkpoint: str, output_dir: str, data_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    fp32_path = os.path.join(output_dir, "classifier_fp32.onnx")
    int8_path = os.path.join(output_dir, "classifier_int8.onnx")
    calib_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    print("\n[1/4] Loading model")
    model = load_model(checkpoint)

    print("\n[2/4] Exporting ONNX")
    export_onnx(model, fp32_path)

    print("\n[3/4] Quantizing to INT8")
    quantize(fp32_path, int8_path, calib_dir)

    print("\n[4/4] Validating")
    if os.path.exists(val_dir):
        print("\nFP32")
        validate(fp32_path, val_dir)
        print("\nINT8")
        validate(int8_path, val_dir)
    else:
        print(f"  Skipping validation ({val_dir} not found)")

    print("\nFP32 benchmark")
    benchmark(fp32_path)
    print("\nINT8 benchmark")
    benchmark(int8_path)

    fp32_size = file_size_mb(fp32_path)
    int8_size = file_size_mb(int8_path)
    print("\n" + "=" * 50)
    print(f"  FP32:  {fp32_size:.2f} MB")
    print(f"  INT8:  {int8_size:.2f} MB  ({int8_size/fp32_size*100:.0f}% of original)")
    print(f"\n  Deploy: {int8_path}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/classifier/best_classifier.pth")
    parser.add_argument("--output_dir", default="models/classifier/deploy")
    parser.add_argument("--data_dir", default="data/traffic_signs")
    args = parser.parse_args()

    main(args.checkpoint, args.output_dir, args.data_dir)
