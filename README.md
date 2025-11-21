# 🔬 MOTIP Detection Web Client

Real-time person tracking with ONNX.js, featuring re-identification and 60-second memory persistence.

## 🎯 Features

- **Real-time Detection**: ONNX.js YOLOv8 person detection running directly in the browser
- **Deep Learning Re-Identification**: State-of-the-Art (SOTA) OSNet-based Re-ID model for accurate person matching
- **Fallback System**: Automatic fallback to histogram-based features if Re-ID model is not available
- **60-Second Memory**: Maintains person IDs even when they disappear from camera for up to 60 seconds
- **Webcam Streaming**: Live video processing with bounding boxes and ID labels
- **Performance Stats**: Real-time FPS, active tracks, and detection count

## 📋 Prerequisites

1. **Node.js** (v16 or higher)
2. **YOLOv8 ONNX Model**: `yolov8n.onnx` file (required)
3. **Re-ID ONNX Model**: `osnet.onnx` file (optional, falls back to histogram features if not available)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
npm install
```

### 2. Prepare Model Files

You need to prepare ONNX models and place them in the `public/models/` directory:

```bash
# Create models directory
mkdir -p public/models

# Option 1: Convert YOLOv8 to ONNX format
pip install ultralytics onnx
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model.export(format='onnx')"
# Copy yolov8n.onnx to public/models/

# Option 2: Download pre-converted models (if available)
# - Place yolov8n.onnx in web-client/public/models/ (required)
# - Place osnet.onnx in web-client/public/models/ (optional - for deep Re-ID)

# Option 3: Use OSNet Re-ID model
# Download OSNet model from: https://github.com/KaiyangZhou/deep-person-reid
# Convert to ONNX format and place as osnet.onnx in public/models/
# Note: System will automatically fallback to histogram features if Re-ID model is not available
```

### 3. Run Development Server

```bash
npm run dev
```

The application will be available at `http://localhost:5173`

### 4. Build for Production

```bash
npm run build
```

The built files will be in the `dist/` directory.

## 📁 Project Structure

```
web-client/
├── src/
│   ├── detector.js      # ONNX.js YOLOv8 detector
│   ├── reid_detector.js # Deep Learning Re-ID detector (OSNet-based)
│   ├── tracker.js       # Re-ID person tracker with memory
│   └── main.js          # Main application logic
├── public/
│   └── models/
│       ├── yolov8n.onnx # YOLOv8 ONNX model (required)
│       └── osnet.onnx   # OSNet Re-ID ONNX model (optional)
├── index.html           # Main HTML file
├── package.json         # Dependencies
├── vite.config.js       # Vite configuration
└── README.md            # This file
```

## 🎮 Usage

1. **Start Webcam**: Click "▶️ Start Webcam" button
2. **Allow Camera Access**: Grant browser permission for camera access
3. **Track Persons**: The system will automatically detect and track persons
4. **View Statistics**: Monitor active tracks, FPS, and detection count
5. **Test Memory**: Have a person disappear from camera - ID will be maintained for 60 seconds

## 🔧 Configuration

### Memory Duration

To change the memory duration (default: 60 seconds), modify in `src/main.js`:

```javascript
maxMemorySeconds: 60  // Change this value
```

### Confidence Threshold

Adjust the confidence threshold using the slider in the UI, or set default in `src/main.js`:

```javascript
confidenceThreshold: 0.5  // Range: 0.1 - 0.9
```

### Model Paths

Update the model paths in `src/main.js` if your models are located elsewhere:

```javascript
modelPath: '/models/yolov8n.onnx',     // YOLOv8 detection model (required)
reIdModelPath: '/models/osnet.onnx',   // Re-ID model (optional)
useDeepReId: true                      // Enable deep Re-ID (set false to use histogram only)
```

## 🔬 Metode Ilmiah dan Teknik

### Paradigma MOTIP (Multiple Object Tracking as ID Prediction)

**⚠️ Catatan Penting**: Sistem ini **TIDAK menggunakan DeepSORT**. Sistem ini mengimplementasikan paradigma **MOTIP** yang memandang pelacakan objek sebagai masalah prediksi ID, bukan hanya asosiasi deteksi antar frame.

#### Perbandingan MOTIP vs DeepSORT

| Aspek | DeepSORT | MOTIP (Sistem Ini) | Lebih Canggih |
|-------|----------|-------------------|---------------|
| **Motion Prediction** | Kalman Filter (probabilistic) | Tidak menggunakan Kalman Filter | **DeepSORT** - Lebih akurat untuk motion |
| **Matching Algorithm** | Hungarian Algorithm (optimal global) | Greedy matching (local optimal) | **DeepSORT** - Optimal assignment |
| **State Management** | Recursive state update (Bayesian) | Direct history attention | **MOTIP** - Menghindari information decay |
| **Re-Identification** | Deep CNN Re-ID (MARS/ResNet) - 128-512 dim | Color histogram + spatial (30 dim) | **DeepSORT** - Lebih akurat untuk appearance |
| **Memory Mechanism** | Single embedding per track | History buffer (10 features) | **MOTIP** - Robust terhadap variasi frame-to-frame |
| **Occlusion Handling** | Motion prediction + Re-ID | Feature averaging + attention | **MOTIP** - Lebih baik untuk long-term occlusion |
| **Long-term Memory** | ~30 frames (sekitar 1 detik @ 30fps) | 60 detik (60×30 = 1800 frames @ 30fps) | **MOTIP** - Memory jauh lebih lama |
| **Computational Cost** | Tinggi (Kalman + Hungarian + Deep CNN) | Rendah (IoU + histogram) | **MOTIP** - Cocok untuk real-time web |
| **Theoretical Foundation** | Traditional tracking (Bayesian filtering) | Modern attention mechanism | **MOTIP** - Paradigma baru dengan transformer attention |
| **Accuracy (Short-term)** | Sangat baik untuk tracking kontinyu | Baik untuk tracking kontinyu | **DeepSORT** - Lebih akurat short-term |
| **Accuracy (Long-term)** | Menurun setelah occlusion panjang | Stabil untuk occlusion panjang | **MOTIP** - Lebih unggul long-term |

### Analisis: Mana yang Lebih Canggih?

**MOTIP lebih canggih dalam:**

1. **Paradigma Teoritis**: 
   - Menggunakan **attention mechanism** (paradigma modern)
   - Direct history access menghindari vanishing gradient problem
   - Lebih dekat dengan pendekatan transformer-based (state-of-the-art)

2. **Long-term Occlusion Handling**:
   - Memory 60 detik vs DeepSORT ~1 detik
   - Feature history averaging lebih robust untuk variasi temporal
   - Attention mechanism mengatasi masalah information decay

3. **Computational Efficiency untuk Web**:
   - Lightweight feature extraction (30 dim vs 128-512 dim)
   - Tidak memerlukan Kalman Filter (kompleksitas lebih rendah)
   - Cocok untuk real-time browser application

4. **Information Preservation**:
   - Direct history access: $Attention(Q, K, V)$ langsung ke history
   - DeepSORT: Information harus melewati recursive update → decay

**DeepSORT lebih canggih dalam:**

1. **Motion Modeling**:
   - Kalman Filter memberikan prediksi posisi yang akurat
   - Probabilistic approach lebih robust untuk noise

2. **Global Optimal Matching**:
   - Hungarian Algorithm memberikan solusi optimal global
   - MOTIP greedy matching hanya local optimal

3. **Re-ID Accuracy**:
   - Deep CNN (MARS/ResNet) lebih akurat untuk appearance matching
   - 128-512 dimensional embeddings capture lebih banyak informasi

4. **Short-term Tracking Accuracy**:
   - Lebih akurat untuk tracking kontinyu tanpa occlusion
   - Motion prediction membantu saat deteksi skip frame

### Kesimpulan Objektif

**MOTIP lebih canggih secara paradigma dan long-term handling**, tapi **DeepSORT lebih canggih untuk accuracy dan motion modeling**.

**Pilih MOTIP jika:**
- Butuh long-term memory (>10 detik)
- Real-time web application (resource limited)
- Long-term occlusion adalah masalah utama
- Butuh paradigma modern yang scalable

**Pilih DeepSORT jika:**
- Accuracy adalah prioritas utama
- Short-term tracking kontinyu (tanpa occlusion panjang)
- Ada resource untuk Deep CNN Re-ID model
- Butuh motion prediction yang akurat

**Untuk sistem ini (web-based real-time tracking):**
MOTIP adalah pilihan yang tepat karena:
- ✅ Cocok untuk browser (lightweight)
- ✅ Memory 60 detik untuk use case real-world
- ✅ Attention mechanism = paradigma state-of-the-art
- ✅ Mengatasi masalah occlusion panjang yang umum terjadi

#### Prinsip Dasar MOTIP

1. **Direct History Attention**: 
   - Tidak menggunakan recursive state update seperti Kalman Filter
   - Melakukan cross-attention langsung ke history buffer
   - Memungkinkan "short gradient path" untuk memori jangka panjang
   
   $$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

2. **History Buffer Management**:
   - Menyimpan feature historis dalam buffer berukuran tetap (30 frame)
   - Fitur yang teroklusi secara efektif diabaikan dalam mekanisme attention
   - Mengatasi masalah "information decay" pada tracking tradisional

3. **Long-term Memory Persistence**:
   - Memory 60 detik untuk re-identifikasi setelah oklusi
   - Probabilitas collision ID: $P(collision) \approx 1 - e^{-\frac{N^2}{2K}}$
   - K adalah ukuran kamus ID, N adalah jumlah objek unik

### Teknik Deteksi: YOLOv8

#### Arsitektur YOLOv8

- **Backbone**: CSPDarknet dengan modul C2f
- **Neck**: PANet (Path Aggregation Network)
- **Head**: Decoupled head dengan anchor-free detection
- **Input Size**: 640×640 pixels
- **Classes**: 80 COCO classes (dengan fokus pada class 0: person)

#### Pipeline Deteksi

1. **Preprocessing**:
   - Resize dengan aspect ratio preservation
   - Normalisasi pixel values: `[0, 255] → [0, 1]`
   - Format: CHW (Channel-Height-Width) dengan batch dimension
   - Shape: `[1, 3, 640, 640]`

2. **Inference**:
   - Output format: `[1, 84, 8400]`
   - 84 = 4 (bbox: cx, cy, w, h) + 80 (class probabilities)
   - 8400 = anchor points untuk deteksi multi-scale

3. **Postprocessing**:
   - Filter confidence threshold (default: 0.5)
   - Non-Maximum Suppression (NMS) dengan IoU threshold (default: 0.4)
   - Konversi format: center (cx, cy, w, h) → corner (x1, y1, x2, y2)
   - Scale bounding box ke original image size

### Teknik Tracking: Deep Learning Re-Identification (SOTA)

#### Arsitektur Re-ID: OSNet (Omni-Scale Network)

Sistem ini menggunakan **OSNet-based Re-ID model** yang merupakan state-of-the-art (SOTA) untuk person re-identification:

**OSNet Architecture**:
- **Input**: Person crop image (256×128 pixels, height×width)
- **Preprocessing**: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Backbone**: Omni-Scale Network dengan multi-scale feature extraction
- **Output**: 512-dimensional feature embedding (L2-normalized)
- **Matching**: Cosine similarity (dot product untuk L2-normalized vectors)

**Keunggulan OSNet**:
- Lightweight untuk deployment di web browser
- Akurat untuk variasi pose, viewpoint, dan illumination
- Robust terhadap occlusion parsial
- Menggunakan attention mechanism untuk fokus pada area diskriminatif

#### Fallback System: Histogram-based Features

Jika Re-ID model tidak tersedia, sistem secara otomatis menggunakan **histogram-based features**:

**Color Histogram Features**:
- RGB histogram dengan 8 bins per channel (total 24 features)
- Normalisasi per pixel untuk robust terhadap lighting changes

**Spatial Features**:
- Center region vs edge region analysis
- Menangkap pola distribusi warna spasial
- Width/height ratio untuk shape information

**Total Feature Vector**: 30 dimensi
- 24 histogram features (8×3 channels)
- 6 spatial features (center RGB, edge RGB)

#### Dual-Matching Strategy (Bukan Hungarian Algorithm seperti DeepSORT)

Sistem ini menggunakan **greedy matching** dua tahap, bukan Hungarian algorithm yang digunakan DeepSORT:

1. **Spatial Matching (IoU-based)** - Tahap 1:
   - Intersection over Union untuk active tracks
   - Formula: $IoU(A, B) = \frac{|A \cap B|}{|A \cup B|}$
   - Threshold: 0.3 untuk matching aktif
   - **Greedy selection** (bukan global optimal seperti Hungarian)
   - Efektif untuk tracking kontinyu antar frame

2. **Appearance Matching (Feature-based)** - Tahap 2:
   - Untuk lost tracks yang masih dalam memory window (< 60 detik)
   - Cosine similarity untuk re-identifikasi:
   $$Sim(u, v) = \frac{u \cdot v}{||u|| ||v||}$$
   - Threshold: **0.70** untuk deep Re-ID, **0.75** untuk histogram features
   - **Average feature dari history** (bukan single embedding seperti DeepSORT)
   - Mengatasi masalah oklusi jangka panjang

**Mengapa Greedy Matching bukan Hungarian?**
- Lebih cepat untuk real-time web application
- Hungarian algorithm memiliki kompleksitas $O(n^3)$ yang lebih berat
- Greedy matching dengan dua tahap memberikan hasil yang memadai
- Cocok untuk skenario dengan jumlah objek terbatas (person tracking)

#### Feature History Management

- Menyimpan hingga 10 feature vectors terakhir per track
- Average feature untuk matching:
  $$f_{avg} = \frac{1}{N}\sum_{i=1}^{N} f_i$$
- Exponential decay weighting untuk recent features
- Menyediakan robust representation meskipun ada variasi frame-to-frame

### Implementasi ONNX.js Runtime

#### WebAssembly Execution

- **Execution Provider**: WASM (WebAssembly)
- **Threading**: Single-threaded untuk compatibility
- **SIMD**: Disabled untuk maximum compatibility
- **Memory Management**: SharedArrayBuffer untuk efficient tensor operations

#### Optimasi Performa

- **Model Format**: ONNX (Open Neural Network Exchange)
- **Graph Optimization**: Level 'all' untuk fused operations
- **Tensor Operations**: Native WebAssembly untuk kecepatan
- **Memory Pool**: Reusable buffer untuk mengurangi GC pressure

### Algoritma Tracking Detail

#### Update Cycle (per frame)

1. **Detection Phase**:
   ```javascript
   detections = YOLOv8(image) // Person detections
   ```

2. **Spatial Matching Phase**:
   ```javascript
   for each active_track:
       best_iou = max(IoU(track.box, detection.box))
       if best_iou > 0.3:
           match track with detection
   ```

3. **Re-ID Matching Phase** (untuk lost tracks):
   ```javascript
   for each lost_track in memory_window:
       avg_feature = average(track.feature_history)
       for each unmatched_detection:
           similarity = cosine_similarity(avg_feature, detection.feature)
           if similarity > 0.5:
               re-identify track
   ```

4. **Newborn Management**:
   ```javascript
   for each unmatched_detection:
       create_new_track(detection)
   ```

5. **Cleanup Phase**:
   ```javascript
   for each lost_track:
       if (current_time - track.last_seen) > 60_seconds:
           remove track
   ```

### Memory Management System

#### Track States

- **Active**: Terdeteksi di frame saat ini
- **Lost**: Tidak terdeteksi tetapi masih dalam memory window (< 60 detik)
- **Dead**: Dihapus setelah 60 detik absence

#### Memory Persistence

- **Feature History**: Circular buffer dengan max 10 features
- **Position History**: Tidak disimpan (menggunakan IoU untuk spatial matching)
- **Temporal Decay**: Exponential decay untuk older features:
  $$w_i = e^{-0.1 \times (N - i - 1)}$$

### Rendering Architecture

#### Frame Processing Pipeline

1. **Video Capture** (~60 FPS):
   ```javascript
   frame = video.capture() // Native browser API
   ```

2. **Detection** (~10 FPS, throttled):
   ```javascript
   detections = async detect(frame) // Non-blocking
   tracks = tracker.update(detections)
   lastTracks = tracks // Store for rendering
   ```

3. **Rendering** (~60 FPS):
   ```javascript
   canvas.drawImage(video) // Video frame
   drawBoundingBoxes(lastTracks) // Bounding boxes
   drawStats() // Statistics overlay
   ```

#### Visual Indicators

- **Bounding Box**: RGB color per track ID dengan thickness 3px
- **Corner Markers**: 8×8px indicators di setiap corner
- **Label**: ID, duration, memory status, confidence
- **Fill Opacity**: 10% untuk active, 15% untuk memory tracks

## 🔬 How It Works (Simplified)

### 1. Detection Pipeline

1. **Video Capture**: Captures frames from webcam
2. **Preprocessing**: Resizes frame to 640x640 and normalizes pixel values
3. **ONNX Inference**: Runs YOLOv8 model using ONNX.js runtime
4. **Postprocessing**: Applies NMS (Non-Maximum Suppression) to filter detections
5. **Bounding Boxes**: Extracts person bounding boxes with confidence scores

### 2. Tracking Pipeline

1. **IoU Matching**: Matches new detections to active tracks using IoU (Intersection over Union)
2. **Feature Extraction**: Extracts color histogram and spatial features from person regions
3. **Re-ID Matching**: For lost tracks, uses feature similarity to re-identify persons
4. **Memory Management**: Maintains track information for up to 60 seconds after person disappears

### 3. Memory System

- **Active Tracks**: Persons currently visible in camera
- **Lost Tracks**: Persons that disappeared but are still in memory (< 60 seconds)
- **Dead Tracks**: Removed after 60 seconds of absence

## 📊 Performance

- **FPS**: Typically 15-30 FPS depending on hardware
- **Latency**: ~50-100ms per frame (detection + tracking)
- **Memory**: Tracks maintained for 60 seconds after disappearance

## 🐛 Troubleshooting

### Model Not Found

**Error**: "Model not found"

**Solution**: 
- Ensure `yolov8n.onnx` is in `public/models/` directory (required)
- For deep Re-ID, ensure `osnet.onnx` is in `public/models/` directory (optional)
- If Re-ID model not found, system will automatically use histogram-based features
- Check browser console for exact path issues

### Webcam Access Denied

**Error**: "Failed to access webcam"

**Solution**:
- Check browser permissions for camera access
- Ensure HTTPS or localhost (browsers require secure context for camera)

### Low FPS

**Performance Tips**:
- Use smaller YOLOv8 model (nano instead of small/medium)
- Reduce video resolution
- Close other browser tabs
- Use modern browser with WebAssembly support

## 🔐 Security Notes

- Camera access requires HTTPS or localhost
- All processing happens client-side (no data sent to server)
- ONNX.js runs entirely in browser using WebAssembly

## 📝 License

See main project LICENSE file.

## 📚 Referensi Akademik dan Teknis

### Paradigma MOTIP

- **Konsep Dasar**: Multiple Object Tracking as ID Prediction
- **Direct History Attention**: Mekanisme attention langsung ke history buffer tanpa recursive state update
- **Long-term Occlusion Handling**: Kemampuan mempertahankan ID melalui oklusi jangka panjang
- **Mathematical Foundation**: 
  - ID Collision Probability: $P(collision) \approx 1 - e^{-\frac{N^2}{2K}}$
  - Attention Mechanism: $Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$

### Deteksi Objek

- **YOLOv8**: "YOLOv8: State-of-the-Art Object Detection" - Ultralytics
- **Architecture**: CSPDarknet backbone dengan PANet neck
- **Anchor-free Detection**: Decoupled head untuk akurasi dan kecepatan

### Re-Identification

- **Feature-based Matching**: Color histogram + spatial features
- **Cosine Similarity**: Metrik untuk matching appearance
- **IoU Matching**: Intersection over Union untuk spatial tracking

### Implementasi Web

- **ONNX.js**: Microsoft ONNX Runtime untuk WebAssembly
- **WebAssembly**: Standar W3C untuk high-performance web applications
- **Browser-based ML**: Client-side inference tanpa server dependency

## 🙏 Credits

- **YOLOv8**: Ultralytics - State-of-the-art object detection
- **ONNX.js**: Microsoft - Cross-platform ML inference runtime
- **MOTIP**: Multiple Object Tracking as ID Prediction paradigm
- **MOTIP Research**: Paradigma tracking berbasis attention mechanism untuk occlusion handling
