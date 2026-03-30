import os
import json
import threading
import traceback
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import tkinter as tk
from tkinter import filedialog, messagebox

from anomalib.models import Patchcore

# =========================
# Config / constants
# =========================
CONFIG_PATH = "patchcore_image_selector_config.json"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_config(config: dict) -> None:
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to save config: {e}")

def find_image_files(root_dir: str) -> List[Path]:
    root = Path(root_dir)
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(p)
    return sorted(files)

# =========================
# Dataset
# =========================
class ImageDataset(Dataset):
    def __init__(self, image_paths: List[Path], transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, str(path)

# =========================
# Core Logic
# =========================
def build_patchcore_model(device: torch.device):
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=("layer2", "layer3"),
        pre_trained=True,
        coreset_sampling_ratio=0.1,
    )
    model.to(device)
    model.eval()
    return model

def get_preprocessor():
    return transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

@torch.no_grad()
def extract_embeddings(model: Patchcore, dataloader: DataLoader, device: torch.device) -> Tuple[np.ndarray, List[str]]:
    torch_model = model.model
    torch_model.eval()
    torch_model.to(device)

    all_embeddings = []
    all_paths = []

    for batch, paths in dataloader:
        batch = batch.to(device)
        features = torch_model.feature_extractor(batch)
        
        # --- 解像度不一致の解消 (Fix: Expected size 784 but got size 196) ---
        target_size = None
        layer_features = []
        
        # 1. まず最大の解像度（通常は layer2）を確認
        for layer_name in ["layer2", "layer3"]:
            fmap = features[layer_name]
            if target_size is None or fmap.shape[-2:] > target_size:
                target_size = fmap.shape[-2:] # (H, W)
        
        # 2. 全てのレイヤーを最大サイズにリサイズして平坦化
        for layer_name in ["layer2", "layer3"]:
            fmap = features[layer_name]
            if fmap.shape[-2:] != target_size:
                # 補間によりサイズを合わせる
                fmap = F.interpolate(fmap, size=target_size, mode="bilinear", align_corners=False)
            
            b, c, h, w = fmap.shape
            fmap = fmap.reshape(b, c, h * w).permute(0, 2, 1) # [B, H*W, C]
            layer_features.append(fmap)

        # 3. チャンネル方向に結合 (H*Wが揃っているのでエラーにならない)
        embeddings_batch = torch.cat(layer_features, dim=-1)
        embeddings_batch = embeddings_batch.mean(dim=1) # 画像ごとの代表値

        all_embeddings.append(embeddings_batch.cpu().numpy())
        all_paths.extend(paths)

    return np.concatenate(all_embeddings, axis=0), all_paths

def k_center_greedy(embeddings: np.ndarray, k: int) -> List[int]:
    n, d = embeddings.shape
    if k >= n: return list(range(n))
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    feats = embeddings / norms
    selected = [0]
    dist = np.full(n, np.inf, dtype=np.float32)
    for _ in range(1, k):
        last_center = feats[selected[-1]]
        sim = feats @ last_center
        dist = np.minimum(dist, 1.0 - sim)
        next_idx = int(np.argmax(dist))
        selected.append(next_idx)
    return selected

def select_and_copy_images(image_dir: str, save_dir: str, num_images: int, log_callback=None):
    def log(msg: str):
        print(msg)
        if log_callback: log_callback(msg)

    if not os.path.isdir(image_dir): raise ValueError(f"Directory not found: {image_dir}")
    os.makedirs(save_dir, exist_ok=True)

    image_paths = find_image_files(image_dir)
    if not image_paths: raise ValueError("No images found.")
    
    num_images = min(num_images, len(image_paths))
    log(f"Processing {len(image_paths)} images...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pre_processor = get_preprocessor()
    dataset = ImageDataset(image_paths, transform=pre_processor)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    model = build_patchcore_model(device)
    embeddings, paths = extract_embeddings(model, dataloader, device)
    
    log("Selecting diverse images...")
    selected_indices = k_center_greedy(embeddings, num_images)
    
    for i in selected_indices:
        src = Path(paths[i])
        dst = Path(save_dir) / src.name
        # 同名回避
        c = 1
        while dst.exists():
            dst = Path(save_dir) / f"{src.stem}_{c}{src.suffix}"
            c += 1
        shutil.copy2(src, dst)
    log(f"Finished. Saved {num_images} images to {save_dir}")

# =========================
# GUI
# =========================
class PatchcoreSelectorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PatchCore Image Selector")
        self.config = load_config()
        
        self.image_dir_var = tk.StringVar(value=self.config.get("image_dir", ""))
        self.save_dir_var = tk.StringVar(value=self.config.get("save_dir", ""))
        self.num_images_var = tk.StringVar(value=str(self.config.get("num_images", 100)))

        self._build_widgets()

    def _build_widgets(self):
        p = 5
        f1 = tk.Frame(self.root); f1.pack(fill="x", padx=p, pady=p)
        tk.Label(f1, text="Source:").pack(side="left")
        tk.Entry(f1, textvariable=self.image_dir_var, width=50).pack(side="left", padx=p)
        tk.Button(f1, text="...", command=lambda: self.image_dir_var.set(filedialog.askdirectory())).pack(side="left")

        f2 = tk.Frame(self.root); f2.pack(fill="x", padx=p, pady=p)
        tk.Label(f2, text="Output:").pack(side="left")
        tk.Entry(f2, textvariable=self.save_dir_var, width=50).pack(side="left", padx=p)
        tk.Button(f2, text="...", command=lambda: self.save_dir_var.set(filedialog.askdirectory())).pack(side="left")

        f3 = tk.Frame(self.root); f3.pack(fill="x", padx=p, pady=p)
        tk.Label(f3, text="Count:").pack(side="left")
        tk.Entry(f3, textvariable=self.num_images_var, width=10).pack(side="left", padx=p)

        self.txt_log = tk.Text(self.root, height=10, width=70, state="disabled")
        self.txt_log.pack(padx=p, pady=p)

        btn_f = tk.Frame(self.root); btn_f.pack(fill="x", padx=p, pady=p)
        self.btn_run = tk.Button(btn_f, text="Run", command=self.run, width=20)
        self.btn_run.pack(side="left")

    def log(self, msg):
        self.txt_log.config(state="normal")
        self.txt_log.insert("end", msg + "\n")
        self.txt_log.see("end")
        self.txt_log.config(state="disabled")
        self.root.update_idletasks()

    def run(self):
        self.btn_run.config(state="disabled")
        img_d = self.image_dir_var.get()
        sav_d = self.save_dir_var.get()
        try:
            num = int(self.num_images_var.get())
        except:
            print("Invalid number.")
            self.btn_run.config(state="normal")
            return

        def worker():
            try:
                select_and_copy_images(img_d, sav_d, num, self.log)
            except Exception:
                # --- エラーをターミナルへ出力 ---
                print("\n" + "="*30 + " ERROR OCCURRED " + "="*30)
                traceback.print_exc()
                print("="*76 + "\n")
                self.log("Error occurred. Check terminal for details.")
            finally:
                self.btn_run.config(state="normal")
        
        threading.Thread(target=worker, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = PatchcoreSelectorGUI(root)
    root.mainloop()