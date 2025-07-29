# 🖼️ YOLO11 + SAHI

![YOLO11_SAHI Demo](assets/output.gif)

This project implements `object detection` using **YOLOv11** with **SAHI (Slicing Aided Hyper Inference)** for efficient tiled inference on videos. It processes input videos, detects objects using a YOLOv11 model, and applies SAHI to handle large images or videos by slicing them into smaller patches for improved accuracy and performance.

## 🎡 Features

- Perform object detection on videos using YOLOv11 with SAHI.
- Display real-time FPS and bounding boxes with class labels.
- Save processed videos with detection results.
- Configurable slice sizes and device selection (CPU/GPU).

## 💼 Prerequisites

- Python 3.8 or higher
- A compatible video file (e.g., `input_video/input_video_1.mp4`) for inference
- YOLOv11 model weights (e.g., `yolo11n.pt`) from Ultralytics
- Optional: GPU with CUDA support for faster inference

## ✳️ Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/ehsan-golinia/YOLO11_SAHI.git
   ```
   then
   ```bash
   cd YOLO11_SAHI
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   ```
   on linux
   ```bash
   source venv/bin/activate
   ```
   on windows
   ```bash
   venv\Scripts\activate
   ```

3. **Install dependencies**:Ensure you have a `requirements.txt` file with the necessary packages

   ```bash
   python -m pip install -r requirements.txt
   ```

## 💻 Usage

Run the inference script with the default settings:

```bash
python main.py
```

This will:

- Process `input_video/input_video_1.mp4` using `models/yolo11n.pt`.
- Display the video with bounding boxes and FPS.
- Save the output video to `output_video/output.mp4`.

### Optional Arguments

Modify `main.py` main function:

- `source`: Path to the input video (default: `input_video/input_video_1.mp4`)
- `weights`: YOLOv11 model weights (default: `yolo11n.pt`)
- `device`: Device for inference (`cpu` or `cuda`, default: `cpu`)
- `view_img`: Display video during processing (default: `True`)
- `save_img`: Save output video (default: `True`)
- `slice_size`: Size of SAHI slices (default: `(512, 512)`)
- `output_video_name`: Output video filename (default: `output.mp4`)

## 📂 Project Structure

```
YOLO11_SAHI/
├── input_video/          # Directory for input videos
├── models/               # Directory for YOLOv11 model weights
├── output_video/         # Directory for output videos
├── main.py               # Main script for SAHI + YOLOv11 inference
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## 🗒️ Notes
- **Performance:** Use a GPU (`device="cuda"`) for faster inference if available.
See https://pytorch.org/get-started/locally/ for up-to-date torch install instructions if no CUDA devices are seen by torch.
- **SAHI Configuration:** Adjust `slice_size` based on your input resolution and hardware capabilities for optimal results. See [SAHI documentation](https://obss.github.io/sahi/) for details.
- **Ultralytics Guide:** Refer to [Ultralytics SAHI guide](https://docs.ultralytics.com/guides/sahi-tiled-inference/) for advanced SAHI configurations.

Contributing

Contributions are welcome! Please submit a pull request or open an issue for bugs, features, or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for details.