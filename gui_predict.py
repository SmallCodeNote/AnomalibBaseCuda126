import os
import json
import glob
import traceback
import tkinter as tk
from tkinter import filedialog

from tkinterdnd2 import DND_FILES, TkinterDnD

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
        "heatmap_min": 0.0,
        "heatmap_max": 1.0,
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


def apply_heatmap(anomaly_map, original_img, vmin, vmax):
    #anomaly_map = anomaly_map / anomaly_map.max()
    print(anomaly_map.max())
    anomaly_map = anomaly_map.squeeze()

    # --- カラースケール固定 ---
    anomaly_map = np.clip((anomaly_map - vmin) / (vmax - vmin), 0, 1)

    anomaly_map = cm.jet(anomaly_map)[:, :, :3]
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

        self.session = None
        
        self.heatmap_min_var = tk.DoubleVar(value=self.cfg.get("heatmap_min", 0.0))
        self.heatmap_max_var = tk.DoubleVar(value=self.cfg.get("heatmap_max", 1.0))


        self.build_ui()

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------
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

        # --- Score Label ---
        self.score_label = tk.Label(self.root, text="Score: ---", font=("Arial", 12))
        self.score_label.grid(row=row + 1, column=1)

        # --- File Path Label ---
        self.path_label = tk.Label(self.root, text="File: ---", font=("Arial", 10))
        self.path_label.grid(row=row + 2, column=1)

        # --- Canvas ---
        self.canvas = tk.Canvas(self.root, width=600, height=300, bg="gray")
        self.canvas.grid(row=row + 3, column=0, columnspan=3, pady=10)

        # --- Drag & Drop (tkinterDnD2) ---
        self.canvas.drop_target_register(DND_FILES)
        self.canvas.dnd_bind("<<Drop>>", self.on_drop)

        # --- Heatmap Min ---
        tk.Label(self.root, text="Heatmap Min").grid(row=row+4, column=0, sticky="w")
        tk.Scale(self.root, variable=self.heatmap_min_var, from_=0.0, to=100.0,
                resolution=0.01, orient="horizontal", length=200).grid(row=row+4, column=1)

        # --- Heatmap Max ---
        tk.Label(self.root, text="Heatmap Max").grid(row=row+5, column=0, sticky="w")
        tk.Scale(self.root, variable=self.heatmap_max_var, from_=0.0, to=100.0,
                resolution=0.01, orient="horizontal", length=200).grid(row=row+5, column=1)


    # --------------------------------------------------------
    # File Select
    # --------------------------------------------------------
    def select_file(self, var):
        path = filedialog.askopenfilename()
        if path:
            var.set(path)

    def select_dir(self, var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    # --------------------------------------------------------
    # ONNX Load
    # --------------------------------------------------------
    def load_onnx(self):
        onnx_path = self.onnx_var.get().strip()
        onnx_path = os.path.abspath(onnx_path)

        if not os.path.exists(onnx_path):
            self.score_label.config(text="ONNX file not found")
            return False

        try:
            self.session = ort.InferenceSession(
                onnx_path,
                providers=["CPUExecutionProvider"],
            )
            return True
        except Exception as e:
            traceback.print_exc()
            self.score_label.config(text=f"ONNX Load Error: {e}")
            return False

    # --------------------------------------------------------
    # Drag & Drop handler
    # --------------------------------------------------------
    def on_drop(self, event):
        path = event.data.strip("{}")
        if os.path.isfile(path):
            self.predict_single_image(path)

    # --------------------------------------------------------
    # Predict for one image
    # --------------------------------------------------------
    def predict_single_image(self, img_path):
        try:
            if not self.session:
                if not self.load_onnx():
                    return

            original_img, arr = preprocess_image(img_path)

            anomaly_map, score = self.session.run(
                ["anomaly_map", "anomaly_score"],
                {"input": arr},
            )

            anomaly_map = anomaly_map[0]
            score = float(score[0])

            vmin = self.heatmap_min_var.get()
            vmax = self.heatmap_max_var.get()

            heatmap_img = apply_heatmap(anomaly_map, original_img, vmin, vmax)


            # Tkinter view
            original_tk = ImageTk.PhotoImage(original_img.resize((300, 300)))
            heatmap_tk = ImageTk.PhotoImage(heatmap_img.resize((300, 300)))

            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=original_tk)
            self.canvas.create_image(300, 0, anchor="nw", image=heatmap_tk)

            self.original_tk = original_tk
            self.heatmap_tk = heatmap_tk

            # Update labels
            self.score_label.config(text=f"Score: {score:.4f}")
            self.path_label.config(text=f"File: {img_path}")

        except Exception as e:
            traceback.print_exc()
            self.score_label.config(text=f"Error: {e}")

    # --------------------------------------------------------
    # Predict first image in directory
    # --------------------------------------------------------
    def run_inference(self):
        # --- Save config ---
        cfg = {
            "onnx_path": self.onnx_var.get(),
            "image_dir": self.image_dir_var.get(),
            "heatmap_min": self.heatmap_min_var.get(),
            "heatmap_max": self.heatmap_max_var.get(),
        }
        save_config(cfg)

        # --- Load ONNX ---
        if not self.load_onnx():
            return

        image_dir = self.image_dir_var.get().strip()
        if not os.path.isdir(image_dir):
            self.score_label.config(text="Invalid image directory")
            return

        # --- Get Files ---
        files = sorted(glob.glob(os.path.join(image_dir, "*.*")))
        if not files:
            self.score_label.config(text="No image found")
            return

        # --- SaveFileDialog ---
        save_path = filedialog.asksaveasfilename(
            initialdir=image_dir,
            initialfile="result.txt",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt")]
        )

        if not save_path:
            self.score_label.config(text="Save canceled")
            return

        # --- Prediction Loop ---
        vmin = self.heatmap_min_var.get()
        vmax = self.heatmap_max_var.get()

        total = len(files)
        done = 0

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                for img_path in files:
                    try:
                        original_img, arr = preprocess_image(img_path)

                        anomaly_map, score = self.session.run(
                            ["anomaly_map", "anomaly_score"],
                            {"input": arr},
                        )

                        score = float(score[0])

                        # --- Write Result ---
                        fname = os.path.basename(img_path)
                        f.write(f"{fname}\t{score:.6f}\n")

                    except Exception:
                        traceback.print_exc()
                        fname = os.path.basename(img_path)
                        f.write(f"{fname}\tERROR\n")

                    # --- Progress Upadte ---
                    done += 1
                    self.path_label.config(text=f"Progress: {done} / {total} files")
                    self.root.update_idletasks()

            self.score_label.config(text=f"Saved: {save_path}")
            self.path_label.config(text=f"Completed: {total} files")

        except Exception as e:
            traceback.print_exc()
            self.score_label.config(text=f"Error: {e}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    root = TkinterDnD.Tk()  # ★ ここが重要（DnD対応）
    app = PatchcoreInferGUI(root)
    root.mainloop()
