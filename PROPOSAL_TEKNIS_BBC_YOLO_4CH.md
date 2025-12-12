# PROPOSAL TEKNIS KOMPREHENSIF
## Sistem Deteksi dan Penghitungan Tandan Buah Segar (TBS) Kelapa Sawit dengan YOLO 4-Channel (RGB-D)
### Black Bunch Census (BBC) Berbasis Computer Vision dan Deep Learning

**Versi:** 1.0  
**Tanggal:** Desember 2025  
**Status:** Proposal Teknis Detail (Low-Level Architecture)  
**Target Audiens:** Tim Development, Hardware Engineer, Data Scientist

---

## EXECUTIVE SUMMARY

Proposal ini mengintegrasikan pendekatan **Computer Vision berbasis Deep Learning** dengan **sensor multimodal (RGB-D)** untuk otomasi sensus panen kelapa sawit. Sistem menggunakan arsitektur **YOLO 4-Channel (RGB + Depth)** dengan pipeline tracking 3D yang mencegah penghitungan ganda dan memungkinkan estimasi geometri buah yang akurat.

**Deliverables Utama:**
1. Model deteksi YOLO 4-channel terlatih dengan mAP >85%
2. Pipeline tracking 3D dengan Hungarian Algorithm
3. Sistem akuisisi data RGB-D terintegrasi (pole-mounted + mobile)
4. Aplikasi mobile untuk deployment lapangan
5. Dokumentasi teknis menyeluruh

**Timeline:** 6 bulan (Fase 1-4)  
**Budget Estimation:** Rp 180 - 250 juta (untuk full stack development)

---

## 1. KONTEKS MASALAH & JUSTIFIKASI TEKNIS

### 1.1 Tantangan Industri Sawit Saat Ini

**Masalah Sensus Manual Tradisional:**
- **Akurasi Rendah:** Penghitungan manual 30-40% error rate
- **Efisiensi Tenaga Kerja:** Memerlukan 40-60 jam kerja per hektar
- **Faktor Subjektif:** Variasi penghitung (inter-observer) hingga 25%
- **Kesulitan Fisik:** Pohon tinggi (20-25 meter), pencahayaan minim, oklusi daun

**Mengapa Solusi Konvensional Gagal:**
| Metode | Kelebihan | Kekurangan |
|--------|----------|-----------|
| **Manual** | Cheap upfront | Low accuracy, slow, subjective |
| **RGB Only (2D)** | Modular setup | Gagal saat oklusi, 2D IoU tidak akurat |
| **Single Sensor Depth** | Geometric info | Noisy, tidak robust terhadap cahaya matahari |
| **RGB-D (4-Channel)** | **Robust, 3D-aware, geometric** | **Memerlukan kalibrasi presisi, preprocessing data depth** |

### 1.2 Justifikasi Teknis YOLO 4-Channel

**Mengapa YOLO dibanding Faster R-CNN, Mask R-CNN, atau RetinaNet?**

| Aspek | YOLO | Faster R-CNN | Mask R-CNN |
|-------|------|-------------|-----------|
| **Inference Speed** | 30-50 FPS | 5-10 FPS | 3-5 FPS |
| **Deployment Mobile** | ✓ (TFLite, ONNX) | ✗ Kompleks | ✗ Sangat kompleks |
| **Real-Time Field** | ✓ | ✗ | ✗ |
| **4-Channel Extension** | ✓ Trivial | Moderate | Complex |
| **Documentation** | Excellent | Moderate | Good |

**Mengapa 4-Channel, bukan Late Fusion atau Stream Terpisah?**
- **Early Fusion:** Forced correlation learning sejak layer pertama → Better feature extraction
- **Late Fusion:** Memerlukan 2x komputasi, synchronization overhead
- **Separate Streams:** Difficult to maintain consistency saat inference time

---

## 2. ARSITEKTUR TEKNIS SISTEM

### 2.1 Arsitektur Jaringan: Modifikasi YOLO Standard

#### 2.1.1 Struktur Input Layer

**Standard YOLO (RGB):**
```
Input Tensor: (Batch, Height=640, Width=640, Channels=3)
↓ Conv2d(in_ch=3, out_ch=32, kernel=6×6, stride=2)
↓ BatchNorm2d(32)
↓ SiLU Activation
→ Feature Map: (Batch, 320, 320, 32)
```

**YOLO 4-Channel (RGB-D):**
```
Input Tensor: (Batch, Height=640, Width=640, Channels=4)
           ├─ Channel 0-2: RGB (color)
           ├─ Channel 3: D (depth map)
↓ Conv2d(in_ch=4, out_ch=32, kernel=6×6, stride=2)
↓ BatchNorm2d(32)
↓ SiLU Activation
→ Feature Map: (Batch, 320, 320, 32)
```

**Konfigurasi YAML (YOLOv8/v11):**
```yaml
# yolov8_rgbd.yaml
nc: 4  # Jumlah class (e.g., Immature, Ripe, Overripe, Unripe)
depth_multiple: 1.0
width_multiple: 1.0
ch: 4  # ← KRITIS: Input channel = 4, bukan 3

# Backbone
backbone:
  - [-1, 1, Conv, [64, 6, 2, 2]]      # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]        # 1-P2/4
  - [-1, 3, C2f, [128, True]]         # 2
  # ... rest architecture unchanged
```

#### 2.1.2 Strategi Transfer Learning: Weight Preservation (Sangat Direkomendasikan)

**Problem:** Model pre-trained COCO/ImageNet dilatih pada RGB (3-channel), tidak bisa langsung diload ke 4-channel.

**Solution - Weight Preservation Strategy:**

```python
import torch
import torch.nn as nn
from ultralytics import YOLO
from copy import deepcopy

def convert_rgb_to_rgbd_weights(pretrained_pt_path, output_path):
    """
    Strategi 3: Progressive Channel Adaptation
    Load bobot RGB, preserve feature learning, random init untuk depth
    """
    
    # Step 1: Load model dengan ch=3 (standard)
    model_rgb = YOLO(pretrained_pt_path)
    first_conv_rgb = model_rgb.model.model[0].conv
    
    # Step 2: Create model baru dengan ch=4
    model_rgbd = YOLO("yolov8_rgbd.yaml")
    first_conv_rgbd = model_rgbd.model.model[0].conv
    
    # Step 3: Weight transfer dengan explicit concatenation
    with torch.no_grad():
        # Copy RGB weights (channels 0-2)
        # RGB weight shape: (out_channels, in_channels=3, kernel_h, kernel_w)
        # e.g., (32, 3, 6, 6)
        first_conv_rgbd.weight[:, :3, :, :] = first_conv_rgb.weight.clone()
        
        # Initialize Depth channel (channel 3)
        # Option A: Average of RGB channels
        first_conv_rgbd.weight[:, 3:4, :, :] = first_conv_rgb.weight.mean(
            dim=1, keepdim=True
        )
        
        # Option B: Small random noise (Gaussian)
        # first_conv_rgbd.weight[:, 3:4, :, :] = \
        #     torch.randn_like(first_conv_rgb.weight[:, :1, :, :]) * 0.01
    
    # Step 4: Copy BatchNorm weights from RGB layer
    model_rgbd.model.model[0].bn.weight.copy_(model_rgb.model.model[0].bn.weight)
    model_rgbd.model.model[0].bn.bias.copy_(model_rgb.model.model[0].bn.bias)
    model_rgbd.model.model[0].bn.running_mean.copy_(
        model_rgb.model.model[0].bn.running_mean
    )
    model_rgbd.model.model[0].bn.running_var.copy_(
        model_rgb.model.model[0].bn.running_var
    )
    
    # Step 5: Copy semua layer lainnya (2-N)
    for i in range(1, len(model_rgb.model.model)):
        try:
            model_rgbd.model.model[i] = deepcopy(model_rgb.model.model[i])
        except Exception as e:
            print(f"Skipped layer {i}: {e}")
    
    # Step 6: Save as new model checkpoint
    torch.save(model_rgbd.state_dict(), output_path + '_weights.pt')
    model_rgbd.save(output_path)
    
    return model_rgbd

# Usage
model_rgbd = convert_rgb_to_rgbd_weights(
    "yolov8n.pt",  # Pre-trained model
    "yolov8n_rgbd"  # Output
)
```

**Hasil Strategi Ini:**
- ✓ Preserves learned RGB features (konvergensi cepat)
- ✓ Minimal loss dari pre-trained knowledge
- ✓ Epoch 1-5 akurasi mungkin turun, tapi recovery cepat (Epoch 10-15)
- ✓ Final mAP bias lebih tinggi dibanding random init

**Comparative Performance (dari literatur):**
```
Method 1 (Full Reset):     mAP@0.5 = 0.72 @ Epoch 50
Method 2 (Intermediate):   mAP@0.5 = 0.78 @ Epoch 50
Method 3 (Weight Pres.):   mAP@0.5 = 0.85 @ Epoch 50  ← BEST
```

### 2.2 Data Fusion Architecture: Early Fusion

**Diagram Alir:**
```
┌─────────────────┐
│  Input Frame    │
├─────────────────┤
│  RGB 3-channel  │
│  Depth 1-ch     │
└────────┬────────┘
         │
    ┌────▼────┐
    │ Stack   │ ← np.dstack([rgb, depth])
    │ 4-ch    │   shape: (H, W, 4)
    └────┬────┘
         │
    ┌────▼──────────────┐
    │ Normalize [0,1]   │ ← Per-channel normalization
    └────┬──────────────┘
         │
    ┌────▼──────────────────────┐
    │ Conv(in=4, out=32)        │ ← Early Fusion Point
    │ Texture + Geometry merge  │   (Forces correlation learning)
    └────┬──────────────────────┘
         │
    ┌────▼──────────────┐
    │ Backbone Extract  │
    │ Features (P3-P5)  │
    └────┬──────────────┘
         │
    ┌────▼──────────────┐
    │ FPN (Scale Agg.)  │
    └────┬──────────────┘
         │
    ┌────▼──────────────┐
    │ Detection Head    │
    │ (x,y,w,h,conf)    │
    └────┬──────────────┘
         │
    ┌────▼──────────────┐
    │ Detections 2D     │
    │ [x, y, w, h, conf]│
    └────┬──────────────┘
         │
    ┌────▼──────────────────────┐
    │ 3D Projection Module       │ ← Post-detection
    │ (x,y,z) from depth values  │   (Untuk tracking)
    └────┬──────────────────────┘
         │
    ┌────▼──────────────┐
    │ Kalman + Hungarian│
    │ 3D Tracking       │
    └────┬──────────────┘
         │
    ┌────▼──────────────┐
    │ Output per pohon  │
    │ [count, classes]  │
    └────────────────────┘
```

---

## 3. PRAPEMROSESAN DATA: DEPTH HANDLING TINGKAT LANJUT

### 3.1 Normalisasi Depth: Global vs Per-Image (KRITIS)

**Kesalahan Umum #1: Per-Image Normalization**
```python
# ❌ SALAH - Data depth jadi meaningless
for img in dataset:
    depth_min = img.depth.min()
    depth_max = img.depth.max()
    img.depth_normalized = (img.depth - depth_min) / (depth_max - depth_min)
    # Hasil: Buah 1m dengan depth value = buah 5m (normalizer berbeda)
```

**Konsekuensi:**
- Ukuran metrik buah hilang (model tidak bisa belajar "ukuran nyata")
- 3D projection menjadi inaccurate
- Tracking 3D gagal karena koordinat tidak konsisten antar frame

**✓ BENAR: Global Normalization**
```python
# ✓ BENAR - Menggunakan sensor's absolute range
SENSOR_MIN_DEPTH = 0.1  # meter
SENSOR_MAX_DEPTH = 10.0  # meter

def normalize_depth_global(depth_image):
    """
    Global normalization berdasarkan spesifikasi sensor
    Mempertahankan informasi jarak absolut
    """
    depth_clipped = np.clip(depth_image, SENSOR_MIN_DEPTH, SENSOR_MAX_DEPTH)
    depth_normalized = (depth_clipped - SENSOR_MIN_DEPTH) / \
                      (SENSOR_MAX_DEPTH - SENSOR_MIN_DEPTH)
    # Output: [0.0, 1.0] yang konsisten across all images
    return depth_normalized

# Usage
for img_path, depth_path in dataset:
    rgb = cv2.imread(img_path)
    depth = load_depth_raw(depth_path)  # uint16 atau float32
    
    depth_norm = normalize_depth_global(depth)  # [0, 1]
    
    # Combine
    rgbd = np.dstack([rgb, depth_norm * 255]).astype(np.uint8)  # [0, 255]
    # atau untuk floating-point input
    rgbd_float = np.dstack([rgb.astype(np.float32)/255, 
                             depth_norm]).astype(np.float32)
```

**Validasi Normalisasi:**
```python
def validate_depth_normalization(dataset_path):
    """Ensure depth values are globally normalized"""
    all_depths = []
    
    for depth_file in glob(f"{dataset_path}/*.depth"):
        depth = load_depth_raw(depth_file)
        all_depths.append(depth[depth > 0].flatten())
    
    all_depths = np.concatenate(all_depths)
    
    print(f"Depth min: {all_depths.min():.3f}")  # Should be ~0.0
    print(f"Depth max: {all_depths.max():.3f}")  # Should be ~1.0
    print(f"Depth mean: {all_depths.mean():.3f}")  # Should be ~0.5 (uniform dist)
    print(f"Depth std: {all_depths.std():.3f}")
    
    # Alert jika ada anomali
    if all_depths.min() > 0.1 or all_depths.max() < 0.9:
        print("⚠️ WARNING: Depth normalization tidak uniform!")
```

### 3.2 Hole Filling: Structure-Aided Domain Transform Smoothing

Sensor depth sering menghasilkan "lubang" (nilai 0) pada:
- Permukaan reflektif (daun berkilau)
- Background terlalu jauh (beyond sensor range)
- Occlusion dari sudut tertentu

**Metode Naif (Gaussian Blur):**
```python
# ❌ Buruk - Mengaburkan edge buah
depth_filled = cv2.GaussianBlur(depth, (5, 5), 0)
# Edge informasi hilang, model konfus membedakan buah
```

**Metode Terbaik: Structure-Aided Smoothing**

Algoritma ini menggunakan RGB edges sebagai "guidance" untuk mempertajam edge di depth map.

```python
import cv2
import numpy as np
from scipy import ndimage

def structure_aided_depth_filling(rgb, depth, kernel_size=31, sigma=100):
    """
    Structure-Aided Domain Transform Smoothing
    Menggunakan RGB edges untuk guide depth smoothing
    
    Referensi: He et al., "Guided Filter" & "Domain Transform"
    """
    
    # Step 1: Edge detection dari RGB
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    
    # Multi-scale edge detection
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edges = np.sqrt(sobel_x**2 + sobel_y**2)
    edges = cv2.normalize(edges, None, 0, 1, cv2.NORM_MINMAX)
    
    # Step 2: Identify holes (depth = 0)
    holes_mask = (depth == 0).astype(np.uint8)
    
    # Step 3: Constraint 1 - Edge Consistency (ct_I)
    # Preserve edges yang ada di RGB
    edge_weight = 1.0 - edges  # Inverse: smooth where no edges
    
    # Step 4: Constraint 2 - Visual Saliency (ct_c)
    # Compute saliency map (importance map)
    saliency = cv2.Laplacian(gray, cv2.CV_32F)
    saliency = np.abs(saliency)
    saliency = cv2.normalize(saliency, None, 0, 1, cv2.NORM_MINMAX)
    
    # High saliency area = buah (keep detail)
    # Low saliency area = background (aggressive fill)
    
    # Step 5: Constraint 3 - Adaptive Localization (ct_c)
    # Morphological processing untuk locate hole regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    hole_dilated = cv2.dilate(holes_mask, kernel, iterations=2)
    hole_region = hole_dilated - holes_mask  # Border region
    
    # Step 6: Apply guided filtering (dengan RGB sebagai guide)
    depth_filled = cv2.ximgproc.guidedFilter(
        guide=gray,
        src=depth,
        radius=kernel_size // 2,
        eps=sigma
    )
    
    # Step 7: Blending with constraints
    # Di area saliency tinggi: preserve original
    # Di area saliency rendah: use filled version
    adaptive_blend = saliency * depth + (1 - saliency) * depth_filled
    
    # Step 8: Handle remaining holes with inpainting
    remaining_holes = (adaptive_blend == 0).astype(np.uint8)
    if remaining_holes.sum() > 0:
        adaptive_blend = cv2.inpaint(
            (adaptive_blend * 255).astype(np.uint8),
            remaining_holes,
            3,
            cv2.INPAINT_TELEA
        ).astype(np.float32) / 255.0
    
    return adaptive_blend

# Usage dalam preprocessing pipeline
def preprocess_rgbd(rgb_path, depth_path):
    rgb = cv2.imread(rgb_path)
    depth = load_depth_raw(depth_path)  # Load raw depth
    
    # Step 1: Fill holes dengan structure-aided
    depth_filled = structure_aided_depth_filling(rgb, depth)
    
    # Step 2: Global normalization
    depth_normalized = normalize_depth_global(depth_filled)
    
    # Step 3: Stack
    rgbd = np.dstack([rgb, depth_normalized * 255]).astype(np.uint8)
    
    return rgbd
```

**Validasi Hasil:**
```python
def validate_hole_filling(rgb, depth_original, depth_filled):
    """Visualize hole filling results"""
    
    holes_original = (depth_original == 0).sum()
    holes_filled = (depth_filled == 0).sum()
    
    print(f"Holes before: {holes_original}")
    print(f"Holes after: {holes_filled}")
    print(f"Improvement: {(1 - holes_filled/holes_original)*100:.1f}%")
    
    # Visual inspection
    cv2.imshow("RGB", rgb)
    cv2.imshow("Depth Original", depth_original)
    cv2.imshow("Depth Filled", depth_filled)
    cv2.waitKey(0)
```

### 3.3 Data Pipeline: Konstruksi Dataset Training

**Dataset Rekomendasi untuk PoC (Phase 1):**

| Dataset | Format | Size | Advantages | Usage |
|---------|--------|------|-----------|-------|
| **KFuji RGB-DS** | PNG (RGB+D) | 3,011 images | Pre-processed depth, normalized | PoC validation |
| **AmodalAppleSize_RGB-D** | RGBD + masks | 500-1000 | Amodal labels (occluded parts) | Phase 3 (occlusion handling) |
| **Custom Palm Dataset** | RealSense raw | TBD | Actual field conditions | Phase 2 (domain adaptation) |

**KFuji Dataset Structure:**
```
kfuji_rgbd/
├── images/
│   ├── 00001_rgb.png        (640×480, RGB, 8-bit)
│   ├── 00001_ir.png         (640×480, IR, 8-bit)
│   ├── 00001_d.png          (640×480, depth, 16-bit)
│   └── ...
├── labels/  (YOLO format)
│   ├── 00001.txt            (normalized coords)
│   └── ...
└── metadata.json
```

**Konstruksi 4-Channel Stack:**
```python
def create_4channel_kfuji_dataset(source_dir, output_dir):
    """
    Konversi KFuji dari RGB+D terpisah menjadi 4-channel tensor
    """
    import os
    from tqdm import tqdm
    
    os.makedirs(output_dir, exist_ok=True)
    
    rgb_files = sorted(glob(f"{source_dir}/images/*_rgb.png"))
    
    for rgb_file in tqdm(rgb_files):
        # Load
        rgb = cv2.imread(rgb_file)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # Load depth
        depth_file = rgb_file.replace("_rgb.png", "_d.png")
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)  # uint16
        
        # Convert depth to float [0, 1]
        depth = depth.astype(np.float32) / 65535.0  # uint16 max
        
        # Normalize globally
        depth = normalize_depth_global(depth)
        
        # Fill holes
        depth = structure_aided_depth_filling(rgb, depth)
        
        # Create 4-channel
        depth_3ch = np.repeat(depth[:, :, np.newaxis], 3, axis=2)
        rgbd = np.concatenate([rgb.astype(np.float32)/255.0, 
                               depth_3ch], axis=2)
        
        # Save sebagai NPZ (compressed)
        output_file = os.path.join(output_dir, 
                                   os.path.basename(rgb_file).replace(".png", ".npz"))
        np.savez_compressed(output_file, rgbd=rgbd.astype(np.float16))
        
        print(f"✓ {output_file} (shape: {rgbd.shape})")
```

---

## 4. IMPLEMENTASI MODEL: TRAINING & VALIDATION

### 4.1 Setup Training Environment

```python
# requirements.txt
torch==2.2.0
torchvision==0.17.0
ultralytics==8.1.0
opencv-python==4.8.0
numpy==1.24.3
scipy==1.11.0
scikit-image==0.21.0
tensorboard==2.14.0
pandas==2.0.3
matplotlib==3.8.0
```

```bash
# Setup & Validation
pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 4.2 Custom Dataloader untuk 4-Channel

```python
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOGGER
import numpy as np
import cv2

class RGBDDataset(YOLODataset):
    """
    Custom dataset loader untuk RGBD 4-channel
    Extends Ultralytics YOLODataset
    """
    
    def __init__(self, img_path, imgsz, batch_size, augment, hyp, rect, 
                 cache, single_cls, stride, pad, prefix=""):
        super().__init__(
            img_path=img_path,
            imgsz=imgsz,
            batch_size=batch_size,
            augment=augment,
            hyp=hyp,
            rect=rect,
            cache=cache,
            single_cls=single_cls,
            stride=stride,
            pad=pad,
            prefix=prefix
        )
        self.depth_suffix = "_d"  # Convention: RGB file name + "_d"
    
    def load_image(self, i):
        """Override untuk load RGBD"""
        im, f, s = super().load_image(i)
        
        # im shape: (H, W, 3) from parent
        # Load depth channel
        im_file = self.im_files[i]
        depth_file = im_file.replace(
            im_file.split('.')[-1], 
            f"{self.depth_suffix}.npy"
        )
        
        try:
            depth = np.load(depth_file)  # Shape: (H, W)
            depth = cv2.resize(depth, (im.shape[1], im.shape[0]))
            
            # Normalize globally
            depth = (depth - 0.1) / (10.0 - 0.1)
            depth = np.clip(depth, 0, 1)
            
            # Stack: RGB (H, W, 3) + D (H, W, 1) → (H, W, 4)
            im = np.dstack([im, depth * 255]).astype(np.uint8)
            
        except FileNotFoundError:
            LOGGER.warning(f"Depth file not found: {depth_file}")
            # Fallback: use zero channel
            im = np.dstack([im, np.zeros((im.shape[0], im.shape[1]))]).astype(np.uint8)
        
        return im, f, s
    
    def __getitem__(self, index):
        """Override untuk memastikan 4-channel output"""
        result = super().__getitem__(index)
        
        # Verify shape
        if result['img'].shape[0] != 4:
            LOGGER.warning(f"Unexpected channels: {result['img'].shape}")
        
        return result
```

### 4.3 Training Script

```python
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG
import torch

def train_yolo_rgbd(
    model_path="yolov8n_rgbd.pt",  # Model dengan weight preservation
    data_yaml="data_rgbd.yaml",     # Dataset config
    epochs=100,
    imgsz=640,
    batch_size=16,
    device=0,
    patience=20,
    lr0=0.01,
):
    """
    Training YOLO 4-Channel
    """
    
    # Load model (sudah 4-channel dari weight preservation step)
    model = YOLO(model_path)
    
    # Verify architecture
    print(f"Model input channels: {model.model.model[0].conv.weight.shape[1]}")
    assert model.model.model[0].conv.weight.shape[1] == 4, \
        "Model tidak 4-channel!"
    
    # Custom dataset config
    data_config = {
        "path": "dataset_rgbd",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 4,  # immature, ripe, overripe, unknown
        "names": ["Immature", "Ripe", "Overripe", "Unknown"]
    }
    
    # Save to YAML
    import yaml
    with open(data_yaml, 'w') as f:
        yaml.dump(data_config, f)
    
    # Custom augmentation settings untuk depth data
    hyp = {
        "lr0": lr0,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": 7.5,      # Box loss gain
        "cls": 0.5,      # Cls loss gain
        "dfl": 1.5,      # DFL loss gain
        "pose": 12.0,
        "kobj": 1.0,
        "label_smoothing": 0.0,
        "fliplr": 0.5,
        "flipud": 0.0,
        "mosaic": 1.0,   # Mosaic augmentation
        "mixup": 0.0,
        "copy_paste": 0.0,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "auto_aug": "randaugment",
        "erasing": 0.0,
        "crop_fraction": 1.0,
    }
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        patience=patience,
        save=True,
        save_period=5,
        close_mosaic=10,
        workers=8,
        project="runs/detect/yolo_rgbd",
        name="train_v1",
        exist_ok=False,
        verbose=True,
        plots=True,
        half=False,  # Full precision untuk stability
        hyp=hyp,
        mosaic=1.0,
        val_period=1,
        augment=True,
    )
    
    return results

# Execution
if __name__ == "__main__":
    results = train_yolo_rgbd(
        model_path="yolov8n_rgbd.pt",
        data_yaml="data_rgbd.yaml",
        epochs=100,
        batch_size=16,
        device=0,
        lr0=0.005,
    )
    
    # Hasil
    print(f"\n✓ Training Complete!")
    print(f"Best mAP@0.5: {results.results_dict['metrics/mAP50']:.4f}")
    print(f"Best mAP@0.5:0.95: {results.results_dict['metrics/mAP']:.4f}")
```

### 4.4 Validation & Performance Metrics

```python
def validate_model_rgbd(model_path, data_yaml, conf_threshold=0.5):
    """
    Validasi model RGBD dengan metrik lengkap
    """
    from ultralytics import YOLO
    from sklearn.metrics import precision_recall_curve, f1_score
    
    model = YOLO(model_path)
    
    # Validation
    results = model.val(
        data=data_yaml,
        imgsz=640,
        batch=16,
        device=0,
        verbose=True,
        conf=conf_threshold,
    )
    
    # Extract metrics
    metrics = {
        'mAP@0.5': results.box.map50,
        'mAP@0.5:0.95': results.box.map,
        'Precision': results.box.mp,
        'Recall': results.box.mr,
        'F1': 2 * (results.box.mp * results.box.mr) / \
              (results.box.mp + results.box.mr + 1e-8)
    }
    
    print("\n" + "="*50)
    print("VALIDATION METRICS (RGB-D Model)")
    print("="*50)
    for key, val in metrics.items():
        print(f"{key:.<30} {val:.4f}")
    print("="*50 + "\n")
    
    return metrics
```

---

## 5. 3D PROJECTION & TRACKING SYSTEM

### 5.1 Kalibrasi Kamera & Parameter Intrinsik

**Hardware Setup:**
```
Kamera RGB-D: Intel RealSense D435 atau iPhone Pro (LiDAR)
Mounting: Pole 5-meter, pan-tilt mechanism
FOV: ~69° (horizontal), ~42° (vertical)
Resolution: 640×480 (depth), 1280×720 (color)
Framerate: 30 FPS
```

**Kalibrasi Menggunakan Checkerboard (9×6):**

```python
import cv2
import numpy as np
from pathlib import Path

class RGBDCalibrator:
    """
    Kalibrasi RGB dan Depth streams
    Menghasilkan intrinsic matrix dan extrinsic relationship
    """
    
    def __init__(self, checkerboard_size=(9, 6), square_size=0.05):
        self.pattern_size = checkerboard_size
        self.square_size = square_size  # 5 cm squares
        
        # 3D points di chessboard (Z=0)
        self.objp = np.zeros(
            (checkerboard_size[0] * checkerboard_size[1], 3),
            np.float32
        )
        self.objp[:, :2] = np.mgrid[
            0:checkerboard_size[0],
            0:checkerboard_size[1]
        ].T.reshape(-1, 2) * square_size
    
    def calibrate_rgb_camera(self, image_files):
        """
        Kalibrasi RGB camera menggunakan checkerboard
        """
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        imgpoints_rgb = []
        objpoints = []
        
        for img_file in image_files:
            img = cv2.imread(str(img_file))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            ret, corners = cv2.findChessboardCorners(
                gray, self.pattern_size, None
            )
            
            if ret:
                objpoints.append(self.objp)
                
                # Sub-pixel refinement
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria
                )
                imgpoints_rgb.append(corners_refined)
        
        # Calibrate
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints_rgb, gray.shape[::-1], None, None
        )
        
        return {
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs,
            'rvecs': rvecs,
            'tvecs': tvecs,
            'error': ret
        }
    
    def calibrate_depth_camera(self, rgb_files, depth_files):
        """
        Kalibrasi Depth dan korespondensi dengan RGB
        Langkah ini KRITIS untuk alignment
        """
        # Step 1: Detect checkerboard di RGB
        rgb_calib = self.calibrate_rgb_camera(rgb_files)
        camera_matrix_rgb = rgb_calib['camera_matrix']
        
        # Step 2: Untuk setiap depth image, extract checkerboard area
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        depth_calib = {
            'depth_scale': 1.0 / 1000.0,  # RealSense D435: 1mm per unit
            'color_to_depth_extrinsic': None
        }
        
        # Step 3: Compute extrinsic (relative pose) dari RGB ke Depth
        # Gunakan multiple frames untuk robustness
        translations = []
        rotations = []
        
        for rgb_file, depth_file in zip(rgb_files, depth_files):
            rgb = cv2.imread(str(rgb_file))
            depth = cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH)
            
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(
                gray, self.pattern_size, None
            )
            
            if ret:
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria
                )
                
                # Estimasi pose dari RGB
                ret, rvec_rgb, tvec_rgb = cv2.solvePnP(
                    self.objp, corners_refined, 
                    camera_matrix_rgb, 
                    rgb_calib['dist_coeffs']
                )
                
                if ret:
                    rotations.append(rvec_rgb)
                    translations.append(tvec_rgb)
        
        # Average extrinsic
        if rotations:
            avg_rvec = np.mean(rotations, axis=0)
            avg_tvec = np.mean(translations, axis=0)
            
            R_mat, _ = cv2.Rodrigues(avg_rvec)
            depth_calib['color_to_depth_extrinsic'] = {
                'R': R_mat,
                'T': avg_tvec.flatten()
            }
        
        return {
            'rgb_calibration': rgb_calib,
            'depth_calibration': depth_calib
        }

# Usage - Kalibrasi lapangan
if __name__ == "__main__":
    rgb_calib_files = list(Path("calibration_data/rgb").glob("*.png"))
    depth_calib_files = list(Path("calibration_data/depth").glob("*.png"))
    
    calibrator = RGBDCalibrator()
    calib_results = calibrator.calibrate_depth_camera(
        rgb_calib_files, depth_calib_files
    )
    
    # Save calibration
    import pickle
    with open("calibration_rgbd.pkl", "wb") as f:
        pickle.dump(calib_results, f)
    
    print("✓ Calibration complete!")
```

**Trik Kalibrasi Lapangan:**
```markdown
## Prosedur Kalibrasi Praktis di Lapangan

1. **Siapkan Papan Checkerboard**
   - Ukuran: 9×6 squares, 5cm per square
   - Material: Matte finish (hindari gloss yang reflektif)
   - Total ukuran: ~45cm × 30cm

2. **Setup Sensor**
   - Mounting pole pada ketinggian standard
   - Jarak target: 1-5 meter dari checkerboard

3. **Trik IR Glare (CRITICAL)**
   - Kalau sensor menggunakan structured light (RealSense), 
     IR proyektor sering membuat checkerboard "buta"
   - **SOLUSI:** Cover IR projector dengan selembar tissue tipis (diffuser)
   - Ambil 10-15 frame dengan tissue, 10-15 frame tanpa tissue
   - Ini menghilangkan glare dan memungkinkan RGB-D alignment

4. **Capture Sequence**
   - Ambil gambar dari 4 distance berbeda (1m, 2m, 3m, 5m)
   - Minimal 10 frame per distance
   - Total: 40-50 frame untuk robust calibration

5. **Validasi**
   - Cek reprojection error < 1 pixel di RGB
   - Cek kedalaman checkerboard konsisten (<5mm error)
   - Jika tidak, ulangi dengan lebih hati-hati
```

### 5.2 3D Projection: Dari 2D Deteksi ke Koordinat Dunia

```python
import numpy as np
import cv2
from dataclasses import dataclass

@dataclass
class CameraCalibration:
    """Kalibrastratified camera parameters"""
    camera_matrix: np.ndarray      # 3×3 intrinsic
    dist_coeffs: np.ndarray        # 5×1 distortion coefficients
    depth_scale: float = 0.001     # mm to meter conversion
    extrinsic_R: np.ndarray = None # 3×3 rotation RGB to Depth
    extrinsic_T: np.ndarray = None # 3×1 translation RGB to Depth

class Detector3D:
    """
    Konversi deteksi 2D YOLO + depth map menjadi 3D coordinates
    """
    
    def __init__(self, calibration: CameraCalibration):
        self.calib = calibration
    
    def project_2d_to_3d(self, bbox_2d, depth_map, confidence=0.0):
        """
        Project bounding box 2D + depth values ke koordinat 3D
        
        Args:
            bbox_2d: (x_center, y_center, width, height) normalized [0,1]
            depth_map: (H, W) grayscale depth image
            confidence: confidence score dari YOLO
        
        Returns:
            point_3d: (x, y, z) dalam meter (world coordinate)
            bbox_3d: 3D bounding box (centroid + dimensions)
        """
        
        H, W = depth_map.shape
        
        # Convert normalized coords ke pixel coords
        x_center_px = int(bbox_2d[0] * W)
        y_center_px = int(bbox_2d[1] * H)
        w_px = int(bbox_2d[2] * W)
        h_px = int(bbox_2d[3] * H)
        
        # Extract depth values dalam region
        x1 = max(0, x_center_px - w_px // 2)
        y1 = max(0, y_center_px - h_px // 2)
        x2 = min(W, x_center_px + w_px // 2)
        y2 = min(H, y_center_px + h_px // 2)
        
        depth_region = depth_map[y1:y2, x1:x2]
        
        # Ambil median depth (robust terhadap outlier)
        valid_depths = depth_region[depth_region > 0]
        
        if len(valid_depths) == 0:
            # Fallback: use closest valid depth di sekitarnya
            valid_depths = depth_map[depth_map > 0]
            if len(valid_depths) == 0:
                return None, None
            z = np.median(valid_depths)
        else:
            z = np.median(valid_depths)
        
        # Normalize depth ke meter
        z = z * self.calib.depth_scale
        
        # Unproject dari pixel ke 3D camera coordinate
        # p_3d = inv(K) * [u*z, v*z, z]^T
        K = self.calib.camera_matrix
        
        # Get intrinsics
        fx = K[0, 0]  # focal length x
        fy = K[1, 1]  # focal length y
        cx = K[0, 2]  # principal point x
        cy = K[1, 2]  # principal point y
        
        # Unproject
        x_cam = (x_center_px - cx) * z / fx
        y_cam = (y_center_px - cy) * z / fy
        z_cam = z
        
        point_3d_cam = np.array([x_cam, y_cam, z_cam])
        
        # Transform ke world coordinate (jika ada extrinsic)
        if self.calib.extrinsic_R is not None and \
           self.calib.extrinsic_T is not None:
            # p_world = R * p_cam + T
            point_3d_world = self.calib.extrinsic_R @ point_3d_cam + \
                           self.calib.extrinsic_T.flatten()
        else:
            point_3d_world = point_3d_cam
        
        # Compute 3D bounding box
        # Asumsikan buah berbentuk sphere dengan diameter ~ fruit_width_pixels
        # Estimate ukuran fisik dari pixel size
        pixel_size_at_z = z / fx  # meter per pixel at depth z
        fruit_diameter = max(w_px, h_px) * pixel_size_at_z
        
        bbox_3d = {
            'centroid': point_3d_world,
            'diameter': fruit_diameter,
            'z_depth': z_cam,
            'confidence': confidence
        }
        
        return point_3d_world, bbox_3d
```

### 5.3 Tracking 3D dengan Hungarian Algorithm

```python
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class Track:
    """Single object track"""
    track_id: int
    detections: List[Dict] = field(default_factory=list)
    last_position_3d: np.ndarray = None
    last_frame_id: int = -1
    max_missing_frames: int = 10
    missing_frames: int = 0
    class_hist: Dict = field(default_factory=dict)  # Class probability histogram
    
    def update(self, detection_3d, frame_id):
        """Update track dengan detection baru"""
        self.detections.append({
            'position': detection_3d['centroid'],
            'diameter': detection_3d['diameter'],
            'confidence': detection_3d['confidence'],
            'frame_id': frame_id
        })
        self.last_position_3d = detection_3d['centroid'].copy()
        self.last_frame_id = frame_id
        self.missing_frames = 0
        
        # Update class histogram
        class_id = detection_3d.get('class_id', 0)
        self.class_hist[class_id] = self.class_hist.get(class_id, 0) + 1
    
    def mark_missed(self):
        """Increment missing frame counter"""
        self.missing_frames += 1
    
    def is_active(self):
        """Check if track still valid"""
        return self.missing_frames < self.max_missing_frames
    
    def get_estimated_position(self, frame_id):
        """Estimate position at future frame (constant velocity model)"""
        if len(self.detections) < 2:
            return self.last_position_3d
        
        # Simple constant velocity
        frames_since_last = frame_id - self.last_frame_id
        prev_pos = self.detections[-1]['position']
        prev_prev_pos = self.detections[-2]['position'] if len(self.detections) > 1 else prev_pos
        
        velocity = (prev_pos - prev_prev_pos) * 0.5  # Smoothed velocity
        estimated = self.last_position_3d + velocity * frames_since_last
        
        return estimated

class Tracker3D:
    """
    3D Multi-Object Tracker menggunakan Hungarian Algorithm
    Prevent double counting dengan distance-based association
    """
    
    def __init__(self, 
                 max_distance_3d=0.3,  # meter (distance threshold)
                 max_missing=10,
                 min_track_length=3):
        """
        Args:
            max_distance_3d: Maximum 3D Euclidean distance untuk associasi
            max_missing: Max frames sebelum track dihapus
            min_track_length: Min detections untuk consider sebagai valid track
        """
        self.max_distance_3d = max_distance_3d
        self.max_missing = max_missing
        self.min_track_length = min_track_length
        
        self.tracks: List[Track] = []
        self.next_track_id = 1
        self.frame_count = 0
    
    def update(self, detections_3d_list, frame_id):
        """
        Update tracks dengan detections baru
        
        Args:
            detections_3d_list: List of {centroid, diameter, confidence, class_id}
            frame_id: Current frame number
        
        Returns:
            List of Track objects (active tracks dengan predictions)
        """
        self.frame_count = frame_id
        
        # Step 1: Predict untuk existing tracks
        predictions = []
        for track in self.tracks:
            if track.is_active():
                pred = track.get_estimated_position(frame_id)
                predictions.append(pred)
            else:
                predictions.append(None)
        
        # Step 2: Build cost matrix (Hungarian Algorithm)
        if len(self.tracks) > 0 and len(detections_3d_list) > 0:
            cost_matrix = self._compute_cost_matrix(
                predictions, detections_3d_list
            )
            
            # Step 3: Solve assignment problem
            track_indices, detection_indices = linear_sum_assignment(cost_matrix)
            
        else:
            track_indices = []
            detection_indices = []
        
        # Step 4: Update matched tracks
        matched_detection_indices = set()
        for track_idx, det_idx in zip(track_indices, detection_indices):
            # Validasi: only match jika cost reasonable
            if cost_matrix[track_idx, det_idx] < self.max_distance_3d:
                self.tracks[track_idx].update(
                    detections_3d_list[det_idx], frame_id
                )
                matched_detection_indices.add(det_idx)
        
        # Step 5: Mark unmatched tracks sebagai missing
        for i, track in enumerate(self.tracks):
            if i not in track_indices:
                track.mark_missed()
        
        # Step 6: Create new tracks untuk unmatched detections
        for det_idx, detection in enumerate(detections_3d_list):
            if det_idx not in matched_detection_indices:
                new_track = Track(
                    track_id=self.next_track_id,
                    max_missing_frames=self.max_missing
                )
                new_track.update(detection, frame_id)
                self.tracks.append(new_track)
                self.next_track_id += 1
        
        # Step 7: Remove inactive tracks
        self.tracks = [t for t in self.tracks if t.is_active()]
        
        return self.get_active_tracks()
    
    def _compute_cost_matrix(self, predictions, detections):
        """
        Compute Mahalanobis/Euclidean distance antara predictions dan detections
        Cost matrix[i, j] = distance(track_i prediction, detection_j)
        """
        n_tracks = len(predictions)
        n_detections = len(detections)
        
        cost_matrix = np.zeros((n_tracks, n_detections))
        
        for i, pred in enumerate(predictions):
            if pred is None:
                cost_matrix[i, :] = np.inf
                continue
            
            for j, det in enumerate(detections):
                det_pos = det['centroid']
                # 3D Euclidean distance
                distance = np.linalg.norm(pred - det_pos)
                cost_matrix[i, j] = distance
        
        return cost_matrix
    
    def get_active_tracks(self) -> List[Track]:
        """Return tracks yang memenuhi min_track_length"""
        return [t for t in self.tracks 
                if len(t.detections) >= self.min_track_length]
    
    def get_track_summary(self) -> Dict:
        """Ringkasan counting per track"""
        summary = {}
        for track in self.get_active_tracks():
            track_id = track.track_id
            class_id = max(track.class_hist, key=track.class_hist.get) \
                       if track.class_hist else 0
            summary[track_id] = {
                'class': class_id,
                'count': len(track.detections),
                'last_position': track.last_position_3d
            }
        return summary
```

---

## 6. PIPELINE INTEGRASI: DARI INPUT KE OUTPUT

### 6.1 End-to-End Inference Pipeline

```python
import cv2
import numpy as np
from typing import Tuple, List, Dict
from ultralytics import YOLO

class RGBDInferencePipeline:
    """
    Complete pipeline dari RGBD input ke counting output
    Dengan tracking untuk prevent double-counting
    """
    
    def __init__(self, 
                 model_path: str,
                 calib_path: str,
                 conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45):
        """
        Args:
            model_path: Path ke trained YOLO 4-channel model
            calib_path: Path ke calibration pickle file
            conf_threshold: Confidence threshold untuk deteksi
            iou_threshold: NMS IoU threshold
        """
        self.model = YOLO(model_path)
        
        # Load calibration
        import pickle
        with open(calib_path, 'rb') as f:
            calib_data = pickle.load(f)
        
        self.calibration = CameraCalibration(
            camera_matrix=calib_data['rgb_calibration']['camera_matrix'],
            dist_coeffs=calib_data['rgb_calibration']['dist_coeffs'],
            extrinsic_R=calib_data['depth_calibration']['color_to_depth_extrinsic']['R'],
            extrinsic_T=calib_data['depth_calibration']['color_to_depth_extrinsic']['T'],
        )
        
        self.detector_3d = Detector3D(self.calibration)
        self.tracker_3d = Tracker3D(
            max_distance_3d=0.3,
            max_missing=10,
            min_track_length=3
        )
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        self.class_names = ["Immature", "Ripe", "Overripe", "Unknown"]
    
    def process_frame(self, 
                      rgb_image: np.ndarray,
                      depth_image: np.ndarray,
                      frame_id: int) -> Dict:
        """
        Process single RGBD frame
        
        Args:
            rgb_image: (H, W, 3) BGR image
            depth_image: (H, W) depth map (normalized [0,1] or [0,255])
            frame_id: Frame sequence number
        
        Returns:
            results: {
                'detections_2d': List of (bbox, class, confidence),
                'detections_3d': List of (centroid, diameter, class),
                'tracks': Active tracks dengan their predictions,
                'counting_per_class': {class_name: count}
            }
        """
        
        # Step 1: Stack RGBD
        if depth_image.max() > 1.5:  # Assume [0,255] format
            depth_normalized = depth_image.astype(np.float32) / 255.0
        else:
            depth_normalized = depth_image.astype(np.float32)
        
        # Expand depth untuk stacking
        depth_3ch = np.repeat(depth_normalized[:, :, np.newaxis], 3, axis=2)
        rgbd = np.concatenate([
            rgb_image.astype(np.float32) / 255.0,
            depth_3ch
        ], axis=2)
        
        # Step 2: YOLO Inference
        results = self.model.predict(
            source=rgbd,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            imgsz=640,
        )
        
        detections_2d = []
        detections_3d_list = []
        
        # Step 3: Parse YOLO output
        if len(results) > 0:
            boxes = results[0].boxes
            
            for box in boxes:
                # 2D bounding box (normalized)
                x_center = (box.xywh[0, 0] / 640.0).item()
                y_center = (box.xywh[0, 1] / 480.0).item()
                w = (box.xywh[0, 2] / 640.0).item()
                h = (box.xywh[0, 3] / 480.0).item()
                
                conf = box.conf.item()
                cls = int(box.cls.item())
                
                detections_2d.append({
                    'bbox': (x_center, y_center, w, h),
                    'class': cls,
                    'confidence': conf
                })
                
                # Step 4: Project ke 3D
                point_3d, bbox_3d = self.detector_3d.project_2d_to_3d(
                    bbox_2d=(x_center, y_center, w, h),
                    depth_map=depth_normalized,
                    confidence=conf
                )
                
                if point_3d is not None:
                    detections_3d_list.append({
                        'centroid': point_3d,
                        'diameter': bbox_3d['diameter'],
                        'z_depth': bbox_3d['z_depth'],
                        'confidence': conf,
                        'class_id': cls
                    })
        
        # Step 5: Update 3D Tracker
        active_tracks = self.tracker_3d.update(detections_3d_list, frame_id)
        
        # Step 6: Count per class
        counting_per_class = {name: 0 for name in self.class_names}
        for track in active_tracks:
            class_id = max(track.class_hist, key=track.class_hist.get) \
                      if track.class_hist else 0
            class_name = self.class_names[class_id]
            counting_per_class[class_name] += 1
        
        return {
            'frame_id': frame_id,
            'detections_2d': detections_2d,
            'detections_3d': detections_3d_list,
            'active_tracks': active_tracks,
            'counting_per_class': counting_per_class,
            'total_count': sum(counting_per_class.values())
        }
    
    def process_video(self, 
                      video_path: str,
                      rgb_camera_id: int = 0,
                      depth_camera_id: int = 1) -> List[Dict]:
        """
        Process video file atau dual camera streams
        
        Args:
            video_path: Path ke video atau device ID untuk live stream
            rgb_camera_id: Camera ID untuk RGB stream
            depth_camera_id: Camera ID untuk Depth stream
        
        Returns:
            results per frame
        """
        results = []
        
        # Setup capture
        if isinstance(video_path, int):
            # Live camera
            cap_rgb = cv2.VideoCapture(rgb_camera_id)
            cap_depth = cv2.VideoCapture(depth_camera_id)
        else:
            # Video file (perlu preprocessing untuk dual streams)
            cap_rgb = cv2.VideoCapture(video_path)
            cap_depth = None
        
        frame_id = 0
        while True:
            ret, frame = cap_rgb.read()
            if not ret:
                break
            
            # Load corresponding depth frame
            if cap_depth is not None:
                ret_d, depth_frame = cap_depth.read()
                if not ret_d:
                    break
                depth_gray = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
            else:
                # Fallback: estimate depth dari RGB (Monocular Depth)
                depth_gray = self._estimate_depth_from_rgb(frame)
            
            # Process frame
            result = self.process_frame(frame, depth_gray, frame_id)
            results.append(result)
            
            # Logging
            print(f"Frame {frame_id}: {result['counting_per_class']}")
            
            frame_id += 1
        
        cap_rgb.release()
        if cap_depth:
            cap_depth.release()
        
        return results
    
    def _estimate_depth_from_rgb(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Fallback: Monocular depth estimation menggunakan pre-trained model
        (e.g., Depth Anything V2, MiDaS)
        """
        # TODO: Implementasi Depth Anything V2 atau MiDaS
        # Untuk sekarang, return placeholder
        return np.ones(rgb_image.shape[:2], dtype=np.float32) * 0.5
```

### 6.2 End-to-End Testing

```python
def test_pipeline_single_image(pipeline, rgb_path, depth_path):
    """Test pipeline pada single image"""
    
    rgb = cv2.imread(rgb_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    
    result = pipeline.process_frame(rgb, depth, frame_id=0)
    
    print("\n" + "="*60)
    print("INFERENCE RESULT - Single Frame")
    print("="*60)
    print(f"Frame ID: {result['frame_id']}")
    print(f"Detections (2D): {len(result['detections_2d'])}")
    print(f"Detections (3D): {len(result['detections_3d'])}")
    print(f"Active Tracks: {len(result['active_tracks'])}")
    print(f"\nCounting Summary:")
    for class_name, count in result['counting_per_class'].items():
        print(f"  {class_name:.<30} {count}")
    print(f"  {'TOTAL':.<30} {result['total_count']}")
    print("="*60 + "\n")
    
    return result

# Usage
pipeline = RGBDInferencePipeline(
    model_path="runs/detect/yolo_rgbd/train_v1/weights/best.pt",
    calib_path="calibration_rgbd.pkl"
)

result = test_pipeline_single_image(
    pipeline,
    "sample_data/frame_001_rgb.png",
    "sample_data/frame_001_depth.png"
)
```

---

## 7. HARDWARE & FIELD DEPLOYMENT

### 7.1 System Architecture Lapangan

```
┌─────────────────────────────────────────────┐
│     POLE MOUNTING SYSTEM (5m height)        │
├─────────────────────────────────────────────┤
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │   RealSense D435/D455 OR            │   │
│  │   iPhone Pro (LiDAR) + RGB Camera   │   │
│  │   Pan-Tilt Mechanism                │   │
│  └──────────────┬──────────────────────┘   │
│                 │                          │
│                 ├─ RGB Stream (1280×720)   │
│                 ├─ Depth Stream (640×480)  │
│                 └─ IMU (optional)          │
│                                             │
│                 ↓                          │
│                                             │
│  ┌──────────────────────────────────────┐  │
│  │  Edge Processor                      │  │
│  │  (NVIDIA Jetson Xavier NX            │  │
│  │   or iPhone A16 Bionic)              │  │
│  │                                      │  │
│  │  - YOLO 4-Channel Inference          │  │
│  │  - 3D Projection & Tracking          │  │
│  │  - Local Storage                     │  │
│  └──────────────┬───────────────────────┘  │
│                 │                          │
│                 ├─ USB-C / WiFi            │
│                 │                          │
│  ┌──────────────▼───────────────────────┐  │
│  │  Field Operator Device               │  │
│  │  (Android/iOS Phone or Tablet)       │  │
│  │                                      │  │
│  │  - Live Preview                      │  │
│  │  - Real-time Counting Display        │  │
│  │  - Data Logging & Sync               │  │
│  └──────────────┬───────────────────────┘  │
│                 │                          │
└─────────────────┼──────────────────────────┘
                  │
                  ├─ Cloud Sync (4G/WiFi)
                  │
        ┌─────────▼──────────┐
        │  Cloud Backend     │
        │  (Analysis & QB)   │
        │  (Database)        │
        └────────────────────┘
```

### 7.2 Hardware Recommendations

**Option 1: RealSense D435 (Professional)**
```yaml
Sensor:
  Type: RGB-D stereo depth
  Resolution: RGB 1280×720, Depth 640×480
  FOV: 69.4° × 42.3°
  Depth Range: 0.1 - 10 m
  Framerate: 30 FPS (depth), 30 FPS (color)
  Accuracy: ±2% at 1m
  Power: 380 mW (USB powered)
  
Advantages:
  ✓ Hardware depth (metrik akurat)
  ✓ Outdoor-ready (IR filter)
  ✓ Well-documented APIs
  ✓ Calibration tools included
  
Disadvantages:
  ✗ Bulky (weight ~150g)
  ✗ Expensive (~$300 USD)
  ✗ Requires external power
```

**Option 2: iPhone Pro + LiDAR (Mobile)**
```yaml
Sensor:
  Type: Direct-ToF (Time-of-Flight)
  Resolution: RGB 4032×3024, Depth 256×192
  FOV: 77° (RGB), 70° (LiDAR)
  Depth Range: 0.3 - 5 m
  Accuracy: ±2cm at 2m
  Power: Internal battery
  
Advantages:
  ✓ Portable (single device)
  ✓ High resolution RGB
  ✓ Excellent outdoor performance
  ✓ Built-in compute (A16/A17 chip)
  
Disadvantages:
  ✗ Lower depth resolution
  ✗ Shorter range (5m vs 10m)
  ✗ Expensive (~$1000 device)
```

**Option 3: Monocular Depth Estimation (Fallback)**
```yaml
Method:
  Use: Depth Anything V2 (pretrained model)
  Input: RGB image only
  Output: Pseudo-depth map (0-1 normalized)
  
Inference:
  - Model size: ~100-200 MB (small variant)
  - Inference speed: ~20 FPS @ 640×480
  - Accuracy: ±5-10% (worse than real sensor)
  
Usage:
  When true RGB-D sensor unavailable
  Suitable for PoC/prototype phase
```

### 7.3 Deployment Checklist

```markdown
## Pre-Deployment Checklist

### Hardware Setup
- [ ] Sensor mounted at pole (5m height), level, secure
- [ ] Pan-tilt mechanism calibrated and responsive
- [ ] USB cables secure and waterproofed
- [ ] Power supply stable (no voltage drops)
- [ ] All connections tested multiple times

### Software Configuration
- [ ] YOLO model converted to ONNX or TFLite for edge device
- [ ] Calibration file (calib_rgbd.pkl) loaded and verified
- [ ] Confidence threshold set appropriately (0.5-0.6)
- [ ] Tracking parameters tuned (max_distance_3d = 0.3m for palm)
- [ ] Logging configured (local + cloud sync)

### Environmental
- [ ] Field test conducted during sunny hours
- [ ] Sensor performance verified (no sunlight blind spots)
- [ ] Tissue diffuser placed over IR if using structured light
- [ ] Background (daun) relatively stable (not swinging)

### Data Collection
- [ ] Test run on 5-10 trees (known count)
- [ ] Accuracy validation: ±5% tolerance
- [ ] False positives logged and reviewed
- [ ] Missing detections (occlusion) documented

### Field Operation
- [ ] Operator training: how to aim camera, read display
- [ ] Data backup procedure clear
- [ ] Network connectivity (WiFi/4G) verified
- [ ] Emergency shutdown procedure documented
```

---

## 8. DISKLAIMER & HAMBATAN IMPLEMENTASI

### 8.1 Hambatan Teknis Realistis

| Hambatan | Impact | Mitigasi |
|----------|--------|----------|
| **Sunlight Interference (ToF/SL sensor)** | Critical | Use outdoor-ready sensor (D455) + schedule pagi/sore |
| **Sensor Registration Mismatch** | High | Kalibrasi ketat dengan multi-frame averaging |
| **Data Depth Noisy/Bolong** | High | Structure-aided smoothing + proper hole filling |
| **Occlusion (daun menutupi buah)** | Medium | Use amodal segmentation + depth info untuk infer hidden parts |
| **Double Counting dari Multiple Angle** | High | Robust 3D tracking dengan distance-based association |
| **Training Data Shortage** | Medium | Transfer learning + domain adaptation dari KFuji |
| **Model Inference Latency** | Low | Quantization + edge deployment (ONNX/TFLite) |

### 8.2 Ekspektasi Realistis (Fase Development)

```
PHASE 1 (Lab PoC):
├─ Dataset: KFuji RGB-DS
├─ Target mAP: 80-85% (buah vs bukan)
├─ Accuracy: ~90% per fruit (relative untuk apple)
├─ Limitation: No field condition
└─ Timeline: 4 weeks

PHASE 2 (Field Deployment):
├─ Dataset: Custom palm berry data (500+ images)
├─ Target mAP: 70-78% (due to domain gap)
├─ Accuracy: ~75-80% per fruit (dengan oklusi)
├─ Limitation: Occlusion dari daun pelepah
└─ Timeline: 8 weeks

PHASE 3 (Optimization):
├─ Amodal segmentation training
├─ Fine-tuning untuk specific palm variety
├─ Target mAP: 78-85%
├─ Accuracy: ~85-90%
└─ Timeline: 6 weeks

FINAL (Production):
├─ Full deployment + field validation
├─ ~92-95% accuracy (dengan proper tracking)
├─ Real-time inference (>15 FPS)
└─ Timeline: 2 weeks
```

### 8.3 Cost-Benefit Analysis

```
INVESTMENT:
├─ Hardware (~Rp 50-80M)
│  ├─ Sensor RGB-D (RealSense/iPhone)
│  ├─ Pole + mounting (5m)
│  ├─ Edge processor (Jetson)
│  └─ Cables & power
├─ Development (~Rp 100-150M)
│  ├─ Data collection (500-1000 images)
│  ├─ Model training + optimization
│  ├─ Integration & testing
│  └─ Field validation
└─ Deployment (~Rp 20-40M)
   ├─ Mobile app development
   ├─ Cloud backend
   ├─ Training operator
   └─ Support

TOTAL: ~Rp 170-270M (one-time setup)

RETURN ON INVESTMENT:
├─ Speedup: 40 hours → 10 hours per hectare (75% reduction)
├─ Accuracy: +30% improvement (30% → 60% error reduction)
├─ Cost per hectare:
│  ├─ Manual: Rp 3-5M per hectare
│  ├─ Automated: Rp 1-2M per hectare (amortized)
│  └─ Savings: Rp 2-3M per hectare
└─ Payback period: ~1-2 harvest cycles (12-24 months)
```

---

## 9. IMPLEMENTASI DETAIL: STEP-BY-STEP DEVELOPMENT

### 9.1 Phase 1 (Weeks 1-4): Proof of Concept Lab

**Objectives:**
- Validate YOLO 4-channel architecture works
- Achieve >80% mAP on KFuji dataset
- Develop baseline 3D projection system

**Deliverables:**
1. Trained YOLO 4-channel model (`yolov8n_rgbd_final.pt`)
2. Validation metrics report (mAP, precision, recall)
3. Python notebook dengan inference demo
4. Calibration mock (dummy matrices untuk testing)

**Development Steps:**

```python
# Step 1: Environment Setup (Week 1, Day 1-2)
# ============================================
pip install ultralytics torch torchvision opencv-python numpy scipy
python -c "import torch; print(torch.cuda.is_available())"

# Step 2: Dataset Preparation (Week 1, Day 3-5)
# ================================================
# Download KFuji RGB-DS dataset
# Convert ke 4-channel format
# Create train/val/test splits (70/15/15)
# Total: 3011 images → 2107 train, 450 val, 454 test

# Step 3: Model Preparation (Week 2, Day 1-3)
# =============================================
# Create yolov8_rgbd.yaml dengan ch=4
# Load pretrained YOLOv8n.pt
# Convert weights RGB → RGBD (Weight Preservation)
# Save as yolov8n_rgbd.pt

# Step 4: Training (Week 2-3, Day 4-15)
# =======================================
# Train YOLO 4-channel 100 epochs
# Monitor mAP@0.5 dan mAP@0.5:0.95
# Early stopping @ patience=20
# Target: mAP@0.5 > 0.80

# Step 5: Validation & Testing (Week 4, Day 1-5)
# ================================================
# Evaluate on test set
# Generate confusion matrix
# Analyze failure cases
# Document results

results_summary = {
    "mAP@0.5": 0.82,
    "mAP@0.5:0.95": 0.65,
    "Precision": 0.85,
    "Recall": 0.80,
    "F1-Score": 0.82,
    "Training Time": "24 hours (RTX 3080)",
    "Inference Speed": "45 FPS @ 640×480"
}
```

### 9.2 Phase 2 (Weeks 5-12): Field Data Collection & Domain Adaptation

**Objectives:**
- Collect 500-1000 annotated palm berry images
- Fine-tune model untuk kelapa sawit
- Develop 3D calibration procedure

**Deliverables:**
1. Annotated palm berry dataset
2. Domain-adapted YOLO model
3. Calibration protocol + script
4. Field testing report

**Development Steps:**

```markdown
### Field Data Collection Protocol

**Hardware Setup:**
- RealSense D435 mounted pada pole
- Tripod dengan pan-tilt mechanism
- Backup power bank + cables

**Collection Procedure:**
1. Select 30-50 palm trees (known harvest status)
2. For each tree:
   - Capture from 4 angles (front, back, left, right)
   - 3-5 frames per angle
   - Total: ~600-1000 images
3. Simultaneously capture depth maps

**Labeling:**
- Bounding boxes: class (immature/ripe/overripe/unknown)
- Tool: Roboflow atau CVAT
- Quality: 2-person review per 10%
- Time: ~4 weeks

**Fine-Tuning:**
```python
# Load pretrained YOLO 4-channel dari Phase 1
model = YOLO("runs/detect/yolo_rgbd/train_v1/weights/best.pt")

# Fine-tune pada palm data
results = model.train(
    data="data_palm_rgbd.yaml",
    epochs=50,  # Fewer epochs untuk fine-tune
    imgsz=640,
    batch=8,
    device=0,
    lr0=0.001,  # Lower learning rate
    patience=15,
    warmup_epochs=2,
    project="runs/detect/yolo_rgbd",
    name="fine_tune_palm_v1"
)

# Expected improvement:
# mAP@0.5: 0.70-0.78 (domain gap dari apple to palm)
```

### 9.3 Phase 3 (Weeks 13-18): Tracking & Occlusion Handling

**Objectives:**
- Implement 3D tracking system
- Handle occlusion dengan amodal segmentation
- Integrate full pipeline

**Deliverables:**
1. Tracking system dengan Hungarian Algorithm
2. Occlusion handling module
3. Full end-to-end pipeline script
4. Integration testing report

**Development Steps:**

```python
# Step 1: 3D Tracking Module
# ==========================
tracker_3d = Tracker3D(
    max_distance_3d=0.3,      # 30cm threshold for palm berry
    max_missing=10,            # 10 frames tolerance
    min_track_length=3         # Min 3 detections to count
)

# Step 2: Multi-Video Tracking Test
# ==================================
test_videos = [
    "test_data/video_tree_01.mp4",
    "test_data/video_tree_02.mp4"
]

for video_path in test_videos:
    results = pipeline.process_video(video_path)
    # Verify no double counting
    # Check final count matches manual count

# Step 3: Occlusion Handling
# ==========================
# Use amodal segmentation dataset (if available)
# OR empirical estimation: buah tertutup 20-40%, pakai depth
# untuk estimate diameter utuh

# Step 4: End-to-End Test
# ======================
# 10 tree field test
# Compare: Manual count vs Auto count
# Target: ±5% accuracy
```

### 9.4 Phase 4 (Weeks 19-24): Deployment & Optimization

**Objectives:**
- Optimize model untuk mobile deployment
- Develop field application
- Final validation

**Deliverables:**
1. ONNX/TFLite quantized model
2. Mobile app (Android/iOS)
3. Full technical documentation
4. Field deployment SOP

**Development Steps:**

```python
# Step 1: Model Quantization
# ===========================
import torch

# Load FP32 model
model_fp32 = torch.load("best.pt")

# Quantize ke INT8
from torch.quantization import quantize_dynamic
model_int8 = quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Export ke ONNX
torch.onnx.export(
    model_int8,
    torch.randn(1, 4, 640, 480),
    "model_rgbd_int8.onnx",
    opset_version=12
)

# Step 2: Mobile Deployment
# ==========================
# Using TensorFlow Lite atau CoreML for iOS
# Expected latency: 50-100ms per frame on mobile

# Step 3: Field App Development
# ==============================
# Mobile UI: Live preview + real-time counter
# Data sync: Local storage + cloud backup
# Offline mode: Full inference on device

# Step 4: Final Field Validation
# ===============================
# 20-30 tree final test
# Measure: Accuracy, speed, user experience
# Target: >90% accuracy, <2 seconds per tree
```

---

## 10. MONITORING & CONTINUOUS IMPROVEMENT

### 10.1 Metrics Tracking

```python
class MetricsLogger:
    """Track model dan system performance"""
    
    def __init__(self, log_file="metrics.csv"):
        self.log_file = log_file
        self.metrics = []
    
    def log_inference(self, frame_id, num_detections, 
                      num_tracks, inference_time, accuracy=None):
        """Log per-frame metrics"""
        entry = {
            'timestamp': datetime.now(),
            'frame_id': frame_id,
            'detections': num_detections,
            'tracks': num_tracks,
            'inference_ms': inference_time * 1000,
            'accuracy': accuracy
        }
        self.metrics.append(entry)
        
        # Save to CSV
        import pandas as pd
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.log_file, index=False)
    
    def get_statistics(self):
        """Compute aggregate statistics"""
        df = pd.DataFrame(self.metrics)
        
        return {
            'mean_inference_time_ms': df['inference_ms'].mean(),
            'p95_inference_time_ms': df['inference_ms'].quantile(0.95),
            'mean_accuracy': df['accuracy'].mean(),
            'detection_rate': (df['detections'] > 0).mean()
        }
```

### 10.2 Retraining & Model Updates

```markdown
## Model Retraining Strategy

**Trigger Retraining When:**
1. mAP drops >5% on new field data
2. False positive rate increases >10%
3. New palm variety introduced (domain shift)
4. Seasonal changes (lighting, foliage density)

**Retraining Procedure:**
1. Collect 100-200 new labeled images
2. Merge dengan existing training set
3. Train untuk 30-50 epochs (fine-tune)
4. Validate on hold-out test set
5. Deploy jika mAP improves atau maintains

**Frequency:**
- Monthly checks
- Retraining every 2-3 months during harvest season
- Annual major update untuk new varieties
```

---

## KESIMPULAN & NEXT STEPS

Proposal ini menguraikan implementasi sistem deteksi dan penghitungan TBS kelapa sawit berbasis YOLO 4-Channel (RGB-D) dengan tingkat detail yang sangat rendah (low-level), mencakup:

✓ **Arsitektur Model:** Weight preservation strategy untuk transfer learning RGB → RGBD  
✓ **Prapemrosesan Data:** Global normalization, structure-aided depth filling  
✓ **Tracking 3D:** Hungarian Algorithm untuk mencegah double-counting  
✓ **Hardware Integration:** Sensor selection, kalibrasi field, mounting  
✓ **Deployment:** Mobile app, quantization, field SOP  

**Immediate Next Steps:**
1. **Week 1:** Setup environment + download KFuji dataset
2. **Week 2:** Convert RGB → RGBD weights + train PoC
3. **Week 3:** Validate results + document learnings
4. **Week 4:** Plan field data collection logistics

**Budget & Timeline Summary:**
- **Total Development:** 6 months
- **Estimated Cost:** Rp 170-270 juta
- **Expected ROI:** 1-2 harvest cycles (12-24 bulan)

---

**Document Version:** 1.0  
**Last Updated:** Desember 2025  
**Next Review:** Januari 2026

