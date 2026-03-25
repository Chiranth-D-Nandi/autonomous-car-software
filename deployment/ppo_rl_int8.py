import sys
import io

#prevent onnx export printing emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import torch.nn as nn
import numpy as np
import onnx 
import onnxruntime as ort 
from onnxruntime.quantization import quantize_dynamic, QuantType
#we have used dynamic quantisation unlike static for supervised learning, cuz this has variable activations and on the fly unlike fixed calibration once for supervised learning
import os 
import time 
import argparse 
class Policywrapper(nn.Module):
    def __init__(self, sb3_model):
        super().__init__()
        policy = sb3_model.policy
        self.features_extractor = policy.features_extractor
        self.lstm = policy.lstm_actor
        self.action_net = policy.action_net
    def forward(self, observation, h_in, c_in):
        features = self.features_extractor(observation)
        lstm_input = features.unsqueeze(0)                    
        lstm_out, (h_out, c_out) = self.lstm(lstm_input, (h_in, c_in))
        lstm_out = lstm_out.squeeze(0)                     
        action_logits = self.action_net(lstm_out)           
        return action_logits, h_out, c_out
def export_fp32(checkpoint_path, fp32_path):
    from sb3_contrib import RecurrentPPO
    import simulation
    from transformer import TransformerFeatures

    print(f"Loading model: {checkpoint_path}")
    model = RecurrentPPO.load(checkpoint_path, device="cpu")

    wrapper = Policywrapper(model)
    wrapper.eval()

    dummy_obs = torch.randn(1, 15)
    dummy_h = torch.zeros(1, 1, 64)
    dummy_c = torch.zeros(1, 1, 64)

    print(f"Exporting FP32 ONNX: {fp32_path}")
    torch.onnx.export(
        wrapper,
        (dummy_obs, dummy_h, dummy_c),
        fp32_path,
        input_names=["observation", "h_in", "c_in"],
        output_names=["action_logits", "h_out", "c_out"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "h_in": {1: "batch_size"},
            "c_in": {1: "batch_size"},
            "action_logits": {0: "batch_size"},
            "h_out": {1: "batch_size"},
            "c_out": {1: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    onnx.checker.check_model(onnx.load(fp32_path))
    size_kb = os.path.getsize(fp32_path) / 1024
    print(f"  FP32 size: {size_kb:.1f} KB")

    session = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    out = session.run(None, {
        "observation": dummy_obs.numpy(),
        "h_in": dummy_h.numpy(),
        "c_in": dummy_c.numpy(),
    })
    print(f"  Sanity check: logits shape {out[0].shape}, action={np.argmax(out[0])}")

def quantize_int8(fp32_path, int8_path):
    print(f"\nQuantizing: {fp32_path} -> {int8_path}")

    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
    )

    fp32_kb = os.path.getsize(fp32_path) / 1024
    int8_kb = os.path.getsize(int8_path) / 1024
    reduction = (1 - int8_kb / fp32_kb) * 100

    print(f"  FP32: {fp32_kb:.1f} KB")
    print(f"  INT8: {int8_kb:.1f} KB")
    print(f"  Reduction: {reduction:.1f}%")

    if reduction < 10:
        print("  WARNING: Reduction too low. Quantization may not have worked.")
        print("  Check onnxruntime version: pip show onnxruntime")

def verify(fp32_path, int8_path, num_tests=500):
    print(f"\nVerifying ({num_tests} random states)")

    fp32_sess = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    int8_sess = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])

    matches = 0
    for _ in range(num_tests):
        obs = np.random.randn(1, 15).astype(np.float32)
        h = np.zeros((1, 1, 64), dtype=np.float32)
        c = np.zeros((1, 1, 64), dtype=np.float32)
        inputs = {"observation": obs, "h_in": h, "c_in": c}

        fp32_action = np.argmax(fp32_sess.run(None, inputs)[0])
        int8_action = np.argmax(int8_sess.run(None, inputs)[0])
        if fp32_action == int8_action:
            matches += 1

    acc = 100.0 * matches / num_tests
    status = "PASS" if acc > 95 else "FAIL"
    print(f"  Action agreement: {acc:.1f}% [{status}]")
    return acc


def benchmark(onnx_path, label="", num_runs=1000):
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    obs = np.random.randn(1, 15).astype(np.float32)
    h = np.zeros((1, 1, 64), dtype=np.float32)
    c = np.zeros((1, 1, 64), dtype=np.float32)
    inputs = {"observation": obs, "h_in": h, "c_in": c}

    for _ in range(50):
        session.run(None, inputs)

    start = time.perf_counter()
    for _ in range(num_runs):
        out = session.run(None, inputs)
        h, c = out[1], out[2] 
        inputs["h_in"] = h
        inputs["c_in"] = c
    elapsed = time.perf_counter() - start

    ms = (elapsed / num_runs) * 1000
    fps = num_runs / elapsed
    print(f"  {label} Latency: {ms:.2f} ms | {fps:.0f} FPS")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/rl_ppo/final_model.zip")
    parser.add_argument("--output_dir", default="models/onnx")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    fp32_path = os.path.join(args.output_dir, "policy_fp32.onnx")
    int8_path = os.path.join(args.output_dir, "policy_int8.onnx")

    print("  PPO POLICY -> ONNX -> INT8")

    print("\n[1/4] Export FP32")
    export_fp32(args.checkpoint, fp32_path)

    print("\n[2/4] Quantize INT8")
    quantize_int8(fp32_path, int8_path)

    print("\n[3/4] Verify")
    verify(fp32_path, int8_path)

    print("\n[4/4] Benchmark")
    benchmark(fp32_path, label="FP32")
    benchmark(int8_path, label="INT8")

    print(f"  Size: {os.path.getsize(int8_path)/1024:.1f} KB")

if __name__ == "__main__":
    main()