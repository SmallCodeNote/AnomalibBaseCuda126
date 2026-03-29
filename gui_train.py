import os
import glob
import json
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from anomalib.models.image.patchcore.torch_model import PatchcoreModel

CONFIG_FILE = "patchcore_train_gui_config.json"

# ============================================================
# JSON Config Load
# ============================================================
def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            traceback.print_exc()

    return {
        "normal_dir": "",
        "weights_path": "patchcore_weights.pth",
        "onnx_path": "patchcore.onnx",
    }

# ============================================================
# JSON Config Save
# ============================================================
def save_config(cfg):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4, ensure_ascii=False)
    except Exception:
        traceback.print_exc()

# ============================================================
# PyTorch Dataset
# ============================================================
class ImageFolderDataset(Dataset):
    def __init__(self, root, image_size=256):
        self.files = sorted(
            glob.glob(os.path.join(root, "**", "*.*"), recursive=True)
        )
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img)

# ============================================================
# ONNX Wrapper
# ============================================================
class PatchcoreOnnxWrapper(nn.Module):
    def __init__(self, core: PatchcoreModel):
        super().__init__()
        self.core = core

    def forward(self, x):
        out = self.core(x)
        return out.anomaly_map, out.pred_score

# ============================================================
# PatchCore Train（torch_model）
# ============================================================
def train_patchcore(normal_dir, sampling_ratio, image_size, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Device: {device}")

    dataset = ImageFolderDataset(normal_dir, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PatchcoreModel(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        pre_trained=True,
        num_neighbors=1,
    )
    model.to(device)
    model.train()

    print("[Train] Extracting embeddings...")
    for images in loader:
        images = images.to(device)
        _ = model(images)

    print("[Train] Coreset subsampling...")
    model.subsample_embedding(sampling_ratio=sampling_ratio)

    print("[Train] Memory bank size:", model.memory_bank.shape)
    return model

# ============================================================
# ONNX Export（CPU）
# ============================================================

def export_onnx_from_weights(weights_path, onnx_path, image_size):
    print("[ONNX] Loading model from weights...")

    model = PatchcoreModel(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        pre_trained=True,
        num_neighbors=1,
    )
    
    # Load weight
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)

    model.to("cpu")
    model.eval()

    wrapper = PatchcoreOnnxWrapper(model).to("cpu")
    wrapper.eval()

    dummy = torch.randn(1, 3, image_size, image_size)

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    # delete data file
    data_file = onnx_path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)

    print(f"[ONNX] Exporting to {onnx_path}")

    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["anomaly_map", "anomaly_score"],
        dynamic_axes={
            "input": {0: "batch"},
            "anomaly_map": {0: "batch"},
            "anomaly_score": {0: "batch"},
        }
    )

    if os.path.exists(onnx_path):
        size_mb = os.path.getsize(onnx_path) / 1024**2
        print(f"[ONNX] Export completed. File size: {size_mb:.2f} MB")
        if size_mb < 10:
            print("⚠️ Warning: File size is too small. Weights might still be external.")

# ============================================================
# Tkinter GUI
# ============================================================
class PatchcoreGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PatchCore Trainer (torch_model + ONNX Export)")

        # JSON Load
        self.cfg = load_config()

        # textvariable initialize
        self.normal_var = tk.StringVar(value=self.cfg["normal_dir"])
        self.weights_var = tk.StringVar(value=self.cfg["weights_path"])
        self.onnx_var = tk.StringVar(value=self.cfg["onnx_path"])

        self.build_ui()

    def build_ui(self):
        row = 0

        def add_row(label, var, select_dir=False, select_file=False):
            nonlocal row
            tk.Label(self.root, text=label).grid(row=row, column=0, sticky="w")
            tk.Entry(self.root, textvariable=var, width=50).grid(row=row, column=1)
            if select_dir:
                tk.Button(self.root, text="...", command=lambda: self.select_dir(var)).grid(row=row, column=2)
            elif select_file:
                tk.Button(self.root, text="...", command=lambda: self.select_file(var)).grid(row=row, column=2)
            row += 1

        add_row("good directory", self.normal_var, select_dir=True)
        add_row("weights filename", self.weights_var, select_file=True)
        add_row("ONNX filename", self.onnx_var, select_file=True)

        tk.Button(self.root, text="Train + ONNX Export", command=self.run, width=25).grid(row=row, column=1, pady=10)

    def select_dir(self, var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def select_file(self, var):
        path = filedialog.asksaveasfilename()
        if path:
            var.set(path)

    def run(self):
        try:
            normal = self.normal_var.get()
            weights_path = self.weights_var.get()
            onnx_path = self.onnx_var.get()

            if not normal or not weights_path or not onnx_path:
                messagebox.showerror("Error", "file operation error.")
                return

            # ====== JSON Save ======
            cfg = {
                "normal_dir": normal,
                "weights_path": weights_path,
                "onnx_path": onnx_path,
            }
            save_config(cfg)

            # ====== Train ======
            model = train_patchcore(
                normal_dir=normal,
                sampling_ratio=0.1,
                image_size=256,
                batch_size=8,
            )

            # ====== state_dict Save ======
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            torch.save(model.state_dict(), weights_path)
            print(f"[Save] state_dict saved to {weights_path}")

            # ====== ONNX Export ======
            export_onnx_from_weights(weights_path, onnx_path, image_size=256)

            messagebox.showinfo("Complete", "PatchCore Train & ONNX Export Complete")

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", str(e))


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = PatchcoreGUI(root)
    root.mainloop()
