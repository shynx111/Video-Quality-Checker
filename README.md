# Video Quality & Face Framing Checker

A Python command-line tool that scans individual video files or entire directories to detect common visual, lighting, and face-framing issues, exporting a structured report to an Excel spreadsheet (`.xlsx`).

## Features

The script samples frames throughout each video (automatically skipping the first **20 seconds** to bypass intros or setup adjustments) and evaluates:

* **Image Quality & Lighting:**
  * **Grain / Noise:** Detects high-frequency sensor noise or compression grain.
  * **Darkness:** Flags underexposed footage (measured on the face when detected, or the full frame otherwise).
  * **Color Cast:** Identifies strong warm/orange or cool/blue white-balance shifts.
  * **Glare:** Detects blown-out highlights on the subject's face.
  * **Face & Neck Shadows:** Measures uneven left/right lighting asymmetry across the face and neck, as well as heavy under-chin shadows.
* **Framing & Head Pose** *(requires YuNet face model)*:
  * **Cropping:** Flags if the subject's forehead or chin is cut off by the frame edges.
  * **Head Angle:** Detects head tilt (roll) and side-to-side turning (yaw: *Facing Left* / *Facing Right*).
  * **Excessive Movement:** Flags when the subject shifts position or distance significantly across sampled frames.

**Supported Video Formats:** `.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v`, `.wmv`, `.flv`, `.webm`

---

## Prerequisites & Installation

1. **Install Python dependencies:**
   Ensure you have Python 3.8+ installed, then install the required libraries:
   ```bash
   pip install opencv-python numpy openpyxl
   ```

2. **Add the Face Detection Model (Recommended):**
   To enable face, shadow, glare, framing, and angle checks, download the OpenCV YuNet ONNX model (`face_detection_yunet.onnx` / `face_detection_yunet_2023mar.onnx`) from the [OpenCV Zoo repository](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), rename it to `face_detection_yunet.onnx`, and place it in the **same folder** as the Python script.

   *(Note: If the model file is missing, the script will still run and evaluate overall grain, brightness, and color cast.)*

---

## Usage

Run the script from your terminal, pointing `--input` to either a single video or a folder of videos:

```bash
python check_videos.py --input path/to/videos
```

### Command-Line Arguments

| Argument | Short | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--input` | `-i` | **Yes** | — | Path to a single video file or a directory (searched recursively). |
| `--output` | `-o` | No | `grain_results.xlsx` | Path to the output Excel file. Appends rows if the file already exists. |
| `--interval` | `-n` | No | `100` | Frame sampling step (e.g., `100` checks 1 out of every 100 frames). |

### Examples

**Scan a folder with default settings:**
```bash
python check_videos.py -i ./recordings
```

**Scan a single video, sampling every 30 frames, and save to a custom report:**
```bash
python check_videos.py -i ./interview.mp4 -o quality_report.xlsx -n 30
```

---

## Output Format

The generated Excel file contains one row per video with the following columns:

* **Filename:** Name of the video file.
* **Framing:** `Forehead Cut`, `Chin Cut`, `Face Moved Too Much`, or blank if acceptable.
* **Face Shadow:** `Shadow` if uneven side-to-side facial lighting is detected.
* **Neck Shadow:** `Shadow` if uneven neck lighting or heavy chin shadow is detected.
* **Glare:** `Glare` if overexposed highlights appear on the face.
* **Angle:** `Tilted Left/Right`, `Facing Left/Right`, `Face Moved Too Much`, or blank if centered.
* **Darkness:** `Dark` if average brightness is below the threshold.
* **Colour:** `Orange/Warm`, `Blue/Cool`, or blank if balanced.
* **Grain:** `Grainy` if noise exceeds the threshold.

> **Tip:** Make sure the target `.xlsx` file is closed in Excel before running the script, otherwise saving will fail with a permission error.
