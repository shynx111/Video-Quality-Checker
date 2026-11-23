"""
Video Quality & Face Framing Checker

Scans a video file or folder of videos (skipping frames before the 20-second mark)
to flag common visual and framing issues—such as grain/noise, darkness, color cast,
glare, face/neck lighting asymmetry, cropped faces/chins, head tilt/yaw, and
excessive movement—and writes the results to an Excel spreadsheet.

Requirements:
    - Python packages: opencv-python, numpy, openpyxl
    - Optional face model: place `face_detection_yunet.onnx` in the same directory
      as this script to enable face, lighting, framing, and angle checks.

Usage:
    python check_videos.py --input <video_or_folder> [--output results.xlsx] [--interval 100]

Arguments:
    --input, -i     Path to a single video file or a directory to scan recursively.
    --output, -o    Output .xlsx file path (creates or appends to existing;
                    default: grain_results.xlsx).
    --interval, -n  Sample every Nth frame across the video (default: 100).
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from openpyxl import Workbook, load_workbook

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".flv", ".webm"}

NOISE_THRESHOLD = 0.28
DARKNESS_THRESHOLD = 105.0
COLOR_CAST_THRESHOLD = 40.0
GLARE_BRIGHTNESS = 245
GLARE_PCT_THRESHOLD = 10.0
FACE_SHADOW_ASYM_THRESHOLD = 25.0
NECK_SHADOW_ASYM_THRESHOLD = 15.0
NECK_SHADOW_BRIGHTNESS_DIFF_THRESHOLD = 20.0
ISSUE_REPORT_THRESHOLD_PCT = 0.30
FOREHEAD_CUT_MARGIN_PX = 5
YAW_OFFSET_RATIO_THRESHOLD = 0.35
MIN_EYE_DISTANCE_PX = 15
ROLL_ANGLE_THRESHOLD_DEG = 5.0

FACE_MODEL_PATH = Path(__file__).resolve().parent / "face_detection_yunet.onnx"
_face_detector = None
_face_detector_size = None


def get_face_detector(width, height):
    global _face_detector, _face_detector_size
    if not FACE_MODEL_PATH.exists():
        return None
    if _face_detector is None:
        create_fn = getattr(cv2, "FaceDetectorYN", None)
        if create_fn is not None and hasattr(create_fn, "create"):
            _face_detector = cv2.FaceDetectorYN.create(
                str(FACE_MODEL_PATH), "", (width, height),
                score_threshold=0.6, nms_threshold=0.3, top_k=5000,
            )
        else:
            _face_detector = cv2.FaceDetectorYN_create(
                str(FACE_MODEL_PATH), "", (width, height), 0.6, 0.3, 5000,
            )
        _face_detector_size = (width, height)
    if _face_detector_size != (width, height):
        _face_detector.setInputSize((width, height))
        _face_detector_size = (width, height)
    return _face_detector


def estimate_noise(gray_frame):
    h, w = gray_frame.shape
    if h < 3 or w < 3:
        return 0.0
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = cv2.filter2D(gray_frame.astype(np.float64), -1, kernel)
    return float(np.sum(np.abs(conv)) * math.sqrt(0.5 * math.pi) / (6 * (w - 2) * (h - 2)))


def estimate_brightness(gray_frame):
    return float(np.mean(gray_frame))


def estimate_color_cast(bgr_frame):
    mean_b = float(np.mean(bgr_frame[:, :, 0]))
    mean_r = float(np.mean(bgr_frame[:, :, 2]))
    return mean_r - mean_b


def classify_color_cast(value):
    if value >= COLOR_CAST_THRESHOLD:
        return "Orange/Warm"
    if value <= -COLOR_CAST_THRESHOLD:
        return "Blue/Cool"
    return ""


def summarize_issues(issues_all, total_frames, none_label):
    from collections import Counter
    flat = [issue for sub in issues_all for issue in sub]
    if not flat or total_frames == 0:
        return none_label
    counts = Counter(flat)
    significant = [issue for issue in counts if counts[issue] / total_frames >= ISSUE_REPORT_THRESHOLD_PCT]
    if not significant:
        return counts.most_common(1)[0][0]
    significant.sort(key=lambda issue: counts[issue], reverse=True)
    return "; ".join(significant)


def analyze_framing(x, y, fw, fh, frame_w, frame_h, landmarks):
    issues = []
    top_lm = np.min(landmarks[:, 1])
    if top_lm < -FOREHEAD_CUT_MARGIN_PX:
        issues.append("Forehead Cut")
    if (y + fh) >= frame_h:
        issues.append("Chin Cut")
    return issues


def analyze_angle(landmarks, face_box):
    roll_angle_out = None
    yaw_ratio_out = None
    if len(landmarks) < 5:
        return roll_angle_out, yaw_ratio_out
    right_eye = landmarks[0]
    left_eye = landmarks[1]
    nose = landmarks[2]
    eye_dx = left_eye[0] - right_eye[0]
    eye_dy = left_eye[1] - right_eye[1]
    eye_distance = float(np.hypot(eye_dx, eye_dy))
    if eye_distance < MIN_EYE_DISTANCE_PX:
        return roll_angle_out, yaw_ratio_out
    roll_angle = np.degrees(np.arctan2(eye_dy, eye_dx))
    roll_angle_out = float(roll_angle)
    eye_mid_x = (left_eye[0] + right_eye[0]) / 2
    nose_x = nose[0]
    yaw_ratio_out = float((nose_x - eye_mid_x) / eye_distance)
    return roll_angle_out, yaw_ratio_out


def detect_regions(bgr_frame):
    h, w = bgr_frame.shape[:2]
    detector = get_face_detector(w, h)
    if detector is None:
        return None
    retval, faces = detector.detect(bgr_frame)
    if faces is None or len(faces) == 0:
        return None
    best = max(faces, key=lambda f: f[2] * f[3])
    x, y, fw, fh = [int(round(v)) for v in best[:4]]
    lm = best[4:]
    num_vals = len(lm)
    if num_vals % 2 != 0:
        lm = lm[:-1]
    num_points = len(lm) // 2
    landmarks = lm.reshape(num_points, 2)
    x = max(x, 0)
    y = max(y, 0)
    fw = max(fw, 1)
    fh = max(fh, 1)
    x2 = min(bgr_frame.shape[1], x + fw)
    y_face_end = min(bgr_frame.shape[0], y + fh)
    fw = x2 - x
    fh = y_face_end - y
    face_roi = bgr_frame[y:y_face_end, x:x2]
    neck_extension = int(fh * 0.6)
    y_neck_end = min(bgr_frame.shape[0], y_face_end + neck_extension)
    neck_roi = bgr_frame[y_face_end:y_neck_end, x:x2] if y_neck_end > y_face_end else None
    return {
        "face_roi": face_roi, "neck_roi": neck_roi,
        "face_box": (x, y, fw, fh), "frame_size": (w, h), "landmarks": landmarks,
    }


def analyze_glare(roi_bgr):
    if roi_bgr.size == 0:
        return 0.0
    glare_mask = (
        (roi_bgr[:, :, 0] >= 240) | (roi_bgr[:, :, 1] >= 240) | (roi_bgr[:, :, 2] >= 240)
    )
    glare_pixels = np.sum(glare_mask)
    total_pixels = roi_bgr.shape[0] * roi_bgr.shape[1]
    return 100.0 * glare_pixels / total_pixels


def analyze_asymmetry(roi_bgr):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if gray.size == 0:
        return 0.0
    left_mean = float(np.mean(gray[:, :max(w // 2, 1)]))
    right_mean = float(np.mean(gray[:, w // 2:])) if w > 1 else left_mean
    return abs(left_mean - right_mean)


def check_video(path, interval=30):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  Could not open video: {path}")
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = range(0, max(total_frames, 1), interval)
    noise_scores, brightness_scores, cast_scores = [], [], []
    glare_scores, face_asymmetry_scores, face_brightness_scores = [], [], []
    neck_asymmetry_scores, neck_brightness_scores = [], []
    framing_issues_all, roll_angle_scores, yaw_ratio_scores = [], [], []
    movement_flags = []
    prev_box = None
    frames_with_face = 0
    frames_with_neck = 0
    frames_checked = 0

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if timestamp < 20.0:
            continue
        frames_checked += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        noise_scores.append(estimate_noise(gray))
        cast_scores.append(estimate_color_cast(frame))
        regions = detect_regions(frame)
        if regions is not None:
            face_gray = cv2.cvtColor(regions["face_roi"], cv2.COLOR_BGR2GRAY)
            brightness_scores.append(estimate_brightness(face_gray))
        else:
            brightness_scores.append(estimate_brightness(gray))
        if regions is not None:
            frames_with_face += 1
            glare_scores.append(analyze_glare(regions["face_roi"]))
            f_asym = analyze_asymmetry(regions["face_roi"])
            face_asymmetry_scores.append(f_asym)
            face_brightness_scores.append(
                estimate_brightness(cv2.cvtColor(regions["face_roi"], cv2.COLOR_BGR2GRAY))
            )
            x, y, fw, fh = regions["face_box"]
            frame_w, frame_h = regions["frame_size"]
            if prev_box is not None:
                px, py, pfw, pfh = prev_box
                move_dist = abs(x - px) + abs(y - py)
                size_change = abs(fw - pfw) + abs(fh - pfh)
                if move_dist > 200 or size_change > 200:
                    movement_flags.append(True)
                else:
                    movement_flags.append(False)
            prev_box = (x, y, fw, fh)
            framing_issues_all.append(
                analyze_framing(x, y, fw, fh, frame_w, frame_h, regions["landmarks"])
            )
            roll_angle, yaw_ratio = analyze_angle(regions["landmarks"], regions["face_box"])
            if roll_angle is not None:
                roll_angle_scores.append(roll_angle)
            if yaw_ratio is not None:
                yaw_ratio_scores.append(yaw_ratio)
            if regions["neck_roi"] is not None and regions["neck_roi"].size > 0:
                frames_with_neck += 1
                n_asym = analyze_asymmetry(regions["neck_roi"])
                neck_asymmetry_scores.append(n_asym)
                neck_brightness_scores.append(
                    estimate_brightness(cv2.cvtColor(regions["neck_roi"], cv2.COLOR_BGR2GRAY))
                )

    cap.release()
    if not noise_scores:
        print(f"  No frames could be scored for: {path}")
        return None

    avg_noise = round(sum(noise_scores) / len(noise_scores), 2)
    avg_brightness = round(sum(brightness_scores) / len(brightness_scores), 2)
    avg_cast = round(sum(cast_scores) / len(cast_scores), 2)
    grain_result = "Grainy" if avg_noise >= NOISE_THRESHOLD else ""
    darkness_result = "Dark" if avg_brightness <= DARKNESS_THRESHOLD else ""
    cast_result = classify_color_cast(avg_cast)

    result = {
        "avg_noise": avg_noise, "grain_result": grain_result,
        "avg_brightness": avg_brightness, "darkness_result": darkness_result,
        "avg_cast": avg_cast, "cast_result": cast_result,
    }

    if not FACE_MODEL_PATH.exists():
        result["face_result"] = "Face model not found"
        result.update({
            "avg_glare": None, "glare_result": "Face Model Missing",
            "avg_face_asymmetry": None, "face_shadow_result": "Face Model Missing",
            "avg_neck_asymmetry": None, "avg_neck_vs_face_diff": None,
            "neck_shadow_result": "Face Model Missing",
            "framing_result": "Face Model Missing", "angle_result": "Face Model Missing",
            "avg_yaw_ratio": None,
        })
        return result

    if frames_with_face == 0:
        result["face_result"] = "No Face Detected"
        result.update({
            "avg_glare": None, "glare_result": "No Face Detected",
            "avg_face_asymmetry": None, "face_shadow_result": "No Face Detected",
            "avg_neck_asymmetry": None, "avg_neck_vs_face_diff": None,
            "neck_shadow_result": "No Face Detected",
            "framing_result": "No Face Detected", "angle_result": "No Face Detected",
            "avg_yaw_ratio": None,
        })
        return result

    result["face_result"] = f"Face found in {frames_with_face}/{frames_checked} frames"
    avg_glare = round(sum(glare_scores) / len(glare_scores), 2)
    result["avg_glare"] = avg_glare
    result["glare_result"] = "Glare" if avg_glare >= GLARE_PCT_THRESHOLD else ""

    avg_face_asym = round(sum(face_asymmetry_scores) / len(face_asymmetry_scores), 2)
    result["avg_face_asymmetry"] = avg_face_asym
    result["face_shadow_result"] = "Shadow" if avg_face_asym >= FACE_SHADOW_ASYM_THRESHOLD else ""

    if frames_with_neck == 0:
        result["avg_neck_asymmetry"] = None
        result["avg_neck_vs_face_diff"] = None
        result["neck_shadow_result"] = "N/A"
    else:
        avg_neck_asym = round(sum(neck_asymmetry_scores) / len(neck_asymmetry_scores), 2)
        avg_face_brightness_local = sum(face_brightness_scores) / len(face_brightness_scores)
        avg_neck_brightness_local = sum(neck_brightness_scores) / len(neck_brightness_scores)
        avg_neck_vs_face_diff = round(avg_face_brightness_local - avg_neck_brightness_local, 2)
        result["avg_neck_asymmetry"] = avg_neck_asym
        result["avg_neck_vs_face_diff"] = avg_neck_vs_face_diff
        if avg_neck_asym >= NECK_SHADOW_ASYM_THRESHOLD:
            result["neck_shadow_result"] = "Shadow"
        elif avg_neck_vs_face_diff >= NECK_SHADOW_BRIGHTNESS_DIFF_THRESHOLD:
            result["neck_shadow_result"] = "Shadow"
        else:
            result["neck_shadow_result"] = ""

    result["framing_result"] = summarize_issues(framing_issues_all, frames_with_face, "")
    yaw_ratio_avg = round(sum(yaw_ratio_scores) / len(yaw_ratio_scores), 3) if yaw_ratio_scores else None
    if yaw_ratio_avg is not None and yaw_ratio_avg > YAW_OFFSET_RATIO_THRESHOLD:
        yaw_result = "Facing Right"
    elif yaw_ratio_avg is not None and yaw_ratio_avg < -YAW_OFFSET_RATIO_THRESHOLD:
        yaw_result = "Facing Left"
    else:
        yaw_result = ""
    result["avg_yaw_ratio"] = yaw_ratio_avg

    avg_tilt_angle = round(sum(roll_angle_scores) / len(roll_angle_scores), 2) if roll_angle_scores else None
    if avg_tilt_angle is not None and avg_tilt_angle > ROLL_ANGLE_THRESHOLD_DEG:
        tilt_result = "Tilted Right"
    elif avg_tilt_angle is not None and avg_tilt_angle < -ROLL_ANGLE_THRESHOLD_DEG:
        tilt_result = "Tilted Left"
    else:
        tilt_result = ""
    combined_angle = [issue for issue in (tilt_result, yaw_result) if issue]
    result["angle_result"] = "; ".join(combined_angle)

    if movement_flags:
        movement_ratio = sum(movement_flags) / len(movement_flags)
        if movement_ratio > 0.30:
            result["framing_result"] = "Face Moved Too Much"
            result["angle_result"] = "Face Moved Too Much"

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", default="grain_results.xlsx")
    parser.add_argument("--interval", "-n", type=int, default=100)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Path does not exist: {input_path}")
        return
    if input_path.is_file():
        videos = [input_path]
    else:
        videos = sorted(p for p in input_path.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)
    print(f"Found {len(videos)} video(s).")
    if not videos:
        print("Nothing to do.")
        return

    output_path = Path(args.output)
    if output_path.exists():
        wb = load_workbook(output_path)
        ws = wb.active
    else:
        wb = Workbook()
        ws = wb.active
        ws.append([
            "Filename", "Framing", "Face Shadow", "Neck Shadow", "Glare",
            "Angle", "Darkness", "Colour", "Grain",
        ])

    for v in videos:
        print(f"Checking {v.name}...")
        r = check_video(v, interval=args.interval)
        if r is None:
            ws.append([v.name, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "Error"])
        else:
            ws.append([
                v.name, r.get("framing_result", ""), r.get("face_shadow_result", ""),
                r.get("neck_shadow_result", ""), r.get("glare_result", ""),
                r.get("angle_result", ""),
                r["darkness_result"], r["cast_result"], r["grain_result"],
            ])

    try:
        wb.save(output_path)
    except PermissionError:
        print(f"ERROR: Couldn't save. Close Excel and retry.")
        return
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()