import os
import json
import glob
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageTk
import matplotlib.cm as cm

CONFIG_FILE = "patchcore_infer_config.json"


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
        "onnx_path": "patchcore.onnx",
        "image_dir": "",
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
# ONNX Predict
# ============================================================
def preprocess_image(path, size=256):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # CHW
    arr = np.expand_dims(arr, axis=0)
    return img, arr


def apply_heatmap(anomaly_map, original_img):
    anomaly_map = anomaly_map.squeeze()
    anomaly_map = anomaly_map / anomaly_map.max()
    anomaly_map = cm.jet(anomaly_map)[:, :, :3]  # RGB
    anomaly_map = (anomaly_map * 255).astype(np.uint8)
    anomaly_map = Image.fromarray(anomaly_map).resize(original_img.size)
    blended = Image.blend(original_img, anomaly_map, alpha=0.5)
    return blended


# ============================================================
# Tkinter GUI
# ============================================================
class PatchcoreInferGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PatchCore ONNX Inference GUI")

        self.cfg = load_config()

        self.onnx_var = tk.StringVar(value=self.cfg["onnx_path"])
        self.image_dir_var = tk.StringVar(value=self.cfg["image_dir"])

        self.build_ui()

        self.session = None

    def build_ui(self):
        row = 0

        def add_row(label, var, select_file=False, select_dir=False):
            nonlocal row
            tk.Label(self.root, text=label).grid(row=row, column=0, sticky="w")
            tk.Entry(self.root, textvariable=var, width=50).grid(row=row, column=1)
            if select_file:
                tk.Button(self.root, text="...", command=lambda: self.select_file(var)).grid(row=row, column=2)
            elif select_dir:
                tk.Button(self.root, text="...", command=lambda: self.select_dir(var)).grid(row=row, column=2)
            row += 1

        add_row("ONNX file", self.onnx_var, select_file=True)
        add_row("Image files directory", self.image_dir_var, select_dir=True)

        tk.Button(self.root, text="Start Predict", command=self.run_inference, width=20).grid(row=row, column=1, pady=10)

        self.canvas = tk.Canvas(self.root, width=600, height=300)
        self.canvas.grid(row=row + 1, column=0, columnspan=3)

    def select_file(self, var):
        path = filedialog.askopenfilename()
        if path:
            var.set(path)

    def select_dir(self, var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def load_onnx(self):
        onnx_path = self.onnx_var.get().strip()
        onnx_path = os.path.abspath(onnx_path)
    
        if not os.path.exists(onnx_path):
            messagebox.showerror("ERROR", f"ONNX file not found.\n{onnx_path}")
            return False
    
        # .data file check
        data_path = onnx_path + ".data"
        if not os.path.exists(data_path):
            messagebox.showwarning("Worning", f" .data file notfound.\n{data_path}")

        print("[ONNX] Loading from path:", onnx_path)
    
        try:
            self.session = ort.InferenceSession(
                onnx_path,
                providers=["CPUExecutionProvider"],
            )
            return True
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("ONNX Load Error", str(e))
        return False


    def run_inference(self):
        try:
            # JSON Config Save
            cfg = {
                "onnx_path": self.onnx_var.get(),
                "image_dir": self.image_dir_var.get(),
            }
            save_config(cfg)

            if not self.load_onnx():
                return

            image_dir = self.image_dir_var.get()
            files = sorted(glob.glob(os.path.join(image_dir, "*.*")))

            if not files:
                messagebox.showerror("Error", "Image file not found.")
                return

            # Prediction
            img_path = files[0]
            print("[Infer] Image:", img_path)

            original_img, arr = preprocess_image(img_path)

            anomaly_map, score = self.session.run(
                ["anomaly_map", "anomaly_score"],
                {"input": arr},
            )

            anomaly_map = anomaly_map[0]
            score = float(score[0])

            print("[Infer] Score:", score)

            heatmap_img = apply_heatmap(anomaly_map, original_img)

            # Tkinter view
            original_tk = ImageTk.PhotoImage(original_img.resize((300, 300)))
            heatmap_tk = ImageTk.PhotoImage(heatmap_img.resize((300, 300)))

            self.canvas.create_image(0, 0, anchor="nw", image=original_tk)
            self.canvas.create_image(300, 0, anchor="nw", image=heatmap_tk)

            self.original_tk = original_tk
            self.heatmap_tk = heatmap_tk

            messagebox.showinfo("Complete", f"Score: {score:.4f}")

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", str(e))


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = PatchcoreInferGUI(root)
    root.mainloop()
