"""
yolo_ncnn_infer.py
==================
Lightweight YOLO11 inference for Raspberry Pi 5.
Uses the ncnn Python bindings directly — no ultralytics, no torch, no opencv-dnn.

Install on Pi:
  pip install ncnn opencv-python numpy pyyaml

Usage examples:
  python yolo_ncnn_infer.py --model yolo11n_ncnn_model --source test.jpg
  python yolo_ncnn_infer.py --model yolo11n_ncnn_model --source images_dir
  python yolo_ncnn_infer.py --model yolo11n_ncnn_model --source video.mp4
  python yolo_ncnn_infer.py --model yolo11n_ncnn_model --source usb0
  python yolo_ncnn_infer.py --model yolo11n_ncnn_model --source picamera0 --resolution 640x480
  python yolo_ncnn_infer.py --model yolo11n_ncnn_model --source picamera0 --resolution 1280x720 --display 640x480
  python yolo_ncnn_infer.py --model yolo11n_ncnn_model --source usb0 --resolution 1280x720 --record
"""

import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
import yaml
import ncnn

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model',      required=True,
                    help='Path to NCNN model folder  (contains model.ncnn.param, '
                         'model.ncnn.bin, metadata.yaml)')
parser.add_argument('--source',     required=True,
                    help='Image file, image folder, video file, usb<N>, or picamera<N>')
parser.add_argument('--thresh',     type=float, default=0.5,
                    help='Confidence threshold  (default: 0.5)')
parser.add_argument('--iou',        type=float, default=0.45,
                    help='NMS IoU threshold  (default: 0.45)')
parser.add_argument('--resolution',
                    help='Capture resolution WxH, e.g. 1280x720  (required for picamera/usb/record)')
parser.add_argument('--display',
                    help='Display window resolution WxH, e.g. 640x480  '
                         '(optional — defaults to capture resolution if not set)')
parser.add_argument('--record',     action='store_true',
                    help='Record output to demo1.avi  (requires --resolution)')
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Locate model files
# ---------------------------------------------------------------------------
model_dir     = args.model
param_path    = os.path.join(model_dir, 'model.ncnn.param')
bin_path      = os.path.join(model_dir, 'model.ncnn.bin')
metadata_path = os.path.join(model_dir, 'metadata.yaml')

for fpath, name in [(param_path,    'model.ncnn.param'),
                    (bin_path,      'model.ncnn.bin'),
                    (metadata_path, 'metadata.yaml')]:
    if not os.path.exists(fpath):
        print(f'ERROR: {name} not found in "{model_dir}"')
        sys.exit(1)

# ---------------------------------------------------------------------------
# Parse metadata.yaml  →  class labels + input size
# ---------------------------------------------------------------------------
with open(metadata_path) as f:
    meta = yaml.safe_load(f)

names_dict  = meta['names']                           # {0: 'person', 1: 'bicycle', …}
labels      = [names_dict[i] for i in sorted(names_dict)]
num_classes = len(labels)

imgsz   = meta.get('imgsz', [640, 640])
INPUT_H, INPUT_W = (imgsz, imgsz) if isinstance(imgsz, int) else (imgsz[0], imgsz[1])

print(f'[INFO] {num_classes} classes | input size: {INPUT_W}x{INPUT_H}')
print(f'[INFO] Loading model from "{model_dir}" ...')

# ---------------------------------------------------------------------------
# Load NCNN model (single Net instance, reused every frame)
# ---------------------------------------------------------------------------
net = ncnn.Net()
net.load_param(param_path)
net.load_model(bin_path)
print('[INFO] Model loaded.')

# ---------------------------------------------------------------------------
# Pre-process: letterbox frame  →  ncnn.Mat (3, INPUT_H, INPUT_W) float32
# Returns the Mat plus the inverse-mapping params needed to convert
# detected box coords back to the original frame space.
# ---------------------------------------------------------------------------
def preprocess(frame: np.ndarray):
    h, w = frame.shape[:2]
    scale  = min(INPUT_W / w, INPUT_H / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    pad_top    = (INPUT_H - new_h) // 2
    pad_bottom =  INPUT_H - new_h - pad_top
    pad_left   = (INPUT_W - new_w) // 2
    pad_right  =  INPUT_W - new_w - pad_left

    padded = cv2.copyMakeBorder(resized,
                                pad_top, pad_bottom, pad_left, pad_right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # Use ncnn.Mat.from_pixels_resize for safe pixel ingestion
    # from_pixels expects HWC uint8 BGR, and handles normalisation internally
    # We normalise manually via from_pixels + mean/norm params
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)  # HWC uint8 RGB

    # from_pixels: HWC uint8 → ncnn.Mat (CHW float), pixel_type=PIXEL_RGB
    mat_in = ncnn.Mat.from_pixels(rgb, ncnn.Mat.PixelType.PIXEL_RGB,
                                  INPUT_W, INPUT_H)

    # Normalise to [0, 1]: x = (x - mean) * norm  →  mean=0, norm=1/255
    mean_vals = [0.0, 0.0, 0.0]
    norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]
    mat_in.substract_mean_normalize(mean_vals, norm_vals)

    return mat_in, scale, pad_left, pad_top


# ---------------------------------------------------------------------------
# Post-process: raw YOLO11 output  →  list of (xmin, ymin, xmax, ymax, cls, conf)
#
# YOLO11 NCNN out0 shape (verified from model_ncnn.py):
#   After unsqueeze(0) → (1, 4 + num_classes, num_anchors)
#   rows 0-3  : cx, cy, w, h  in INPUT_W/H pixel space (not normalised)
#   rows 4..  : per-class scores  (no separate objectness — YOLO11 style)
# ---------------------------------------------------------------------------
def postprocess(raw: np.ndarray, scale, pad_left, pad_top,
                orig_w, orig_h, conf_thresh, iou_thresh):

    out = raw[0]                            # (4+nc, num_anchors)
    box_data     = out[:4, :]               # (4, A)
    class_scores = out[4:4 + num_classes, :] # (nc, A)

    # Vectorised best-class selection
    class_ids   = class_scores.argmax(axis=0)  # (A,)
    confidences = class_scores.max(axis=0)     # (A,)

    # Early filter — avoids looping over thousands of weak anchors
    mask = confidences >= conf_thresh
    if not mask.any():
        return []

    box_data    = box_data[:, mask]
    class_ids   = class_ids[mask]
    confidences = confidences[mask]

    # cx,cy,w,h (padded-input space) → corner coords (original image space)
    cx, cy, bw, bh = box_data
    cx = (cx - pad_left) / scale
    cy = (cy - pad_top)  / scale
    bw /= scale
    bh /= scale

    xmins = np.clip(cx - bw / 2, 0, orig_w - 1).astype(int)
    ymins = np.clip(cy - bh / 2, 0, orig_h - 1).astype(int)
    xmaxs = np.clip(cx + bw / 2, 0, orig_w - 1).astype(int)
    ymaxs = np.clip(cy + bh / 2, 0, orig_h - 1).astype(int)

    # cv2.dnn.NMSBoxes wants [x, y, w, h]
    boxes_xywh = [[int(x), int(y), int(x2 - x), int(y2 - y)]
                  for x, y, x2, y2 in zip(xmins, ymins, xmaxs, ymaxs)]
    conf_list  = confidences.tolist()

    indices = cv2.dnn.NMSBoxes(boxes_xywh, conf_list, conf_thresh, iou_thresh)
    if len(indices) == 0:
        return []

    results = []
    for idx in indices.flatten():
        x, y, w, h = boxes_xywh[idx]
        results.append((x, y, x + w, y + h,
                        int(class_ids[idx]), conf_list[idx]))
    return results


# ---------------------------------------------------------------------------
# Bounding-box colours — Tableau-10 palette
# ---------------------------------------------------------------------------
BBOX_COLORS = [
    (164,120, 87), ( 68,148,228), ( 93, 97,209), (178,182,133), ( 88,159,106),
    ( 96,202,231), (159,124,168), (169,162,241), ( 98,118,150), (172,176,184),
]

# ---------------------------------------------------------------------------
# Source detection
# ---------------------------------------------------------------------------
IMG_EXTS = {'.jpg','.jpeg','.png','.bmp','.JPG','.JPEG','.PNG','.BMP'}
VID_EXTS = {'.avi','.mov','.mp4','.mkv','.wmv'}

img_source = args.source
if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    ext = os.path.splitext(img_source)[1]
    if ext in IMG_EXTS:
        source_type = 'image'
    elif ext in VID_EXTS:
        source_type = 'video'
    else:
        print(f'ERROR: Unsupported file extension: {ext}')
        sys.exit(1)
elif img_source.startswith('usb'):
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif img_source.startswith('picamera'):
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'ERROR: Unrecognised source: {img_source}')
    sys.exit(1)

# ---------------------------------------------------------------------------
# Resolution / recording setup
# ---------------------------------------------------------------------------

# --- Capture resolution (what the camera grabs / frame is resized to) ---
resize = False
resW = resH = None
if args.resolution:
    resize = True
    resW, resH = (int(v) for v in args.resolution.split('x'))

# Picamera always needs an explicit capture resolution
if source_type == 'picamera' and not args.resolution:
    print('ERROR: --resolution is required for picamera source  (e.g. --resolution 640x480)')
    sys.exit(1)

# --- Display resolution (what is shown in the OpenCV window) ---
dispW = dispH = None
if args.display:
    dispW, dispH = (int(v) for v in args.display.split('x'))
elif args.resolution:
    # Default display = capture resolution
    dispW, dispH = resW, resH
# If neither is set (image/folder sources), display at native frame size

if args.record:
    if source_type not in ('video', 'usb'):
        print('ERROR: Recording only works for video/camera sources.')
        sys.exit(1)
    if not args.resolution:
        print('ERROR: Please specify --resolution to record.')
        sys.exit(1)
    # Record at capture resolution, not display resolution
    recorder = cv2.VideoWriter('demo1.avi',
                               cv2.VideoWriter_fourcc(*'MJPG'),
                               30, (resW, resH))

# ---------------------------------------------------------------------------
# Open source
# ---------------------------------------------------------------------------
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(os.path.join(img_source, '*'))
                 if os.path.splitext(f)[1] in IMG_EXTS]
    if not imgs_list:
        print(f'ERROR: No supported images found in "{img_source}"')
        sys.exit(1)
elif source_type in ('video', 'usb'):
    cap_arg = img_source if source_type == 'video' else usb_idx
    cap = cv2.VideoCapture(cap_arg)
    if resize:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  resW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(
        main={"format": 'RGB888', "size": (resW, resH)}))
    cap.start()

# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------
avg_frame_rate    = 0.0
frame_rate_buffer = []
FPS_AVG_LEN       = 200
img_count         = 0

while True:
    t_start = time.perf_counter()

    # --- Load frame ---------------------------------------------------------
    if source_type in ('image', 'folder'):
        if img_count >= len(imgs_list):
            print('All images processed. Exiting.')
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1
        if frame is None:
            print(f'WARNING: Could not read {imgs_list[img_count-1]}, skipping.')
            continue

    elif source_type == 'video':
        ret, frame = cap.read()
        if not ret:
            print('Reached end of video. Exiting.')
            break

    elif source_type == 'usb':
        ret, frame = cap.read()
        if not ret or frame is None:
            print('Camera read failed. Exiting.')
            break

    elif source_type == 'picamera':
        frame = cap.capture_array()
        if frame is None:
            print('Picamera read failed. Exiting.')
            break

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    orig_h, orig_w = frame.shape[:2]

    # --- Pre-process --------------------------------------------------------
    mat_in, scale, pad_left, pad_top = preprocess(frame)

    # --- Inference ----------------------------------------------------------
    # create_extractor() is lightweight — a thin stateless view over the net
    with net.create_extractor() as ex:
        ex.input("in0", mat_in)
        _, out0 = ex.extract("out0")

    # out0 is an ncnn.Mat; convert to numpy then reshape to (1, 4+nc, anchors)
    raw = np.array(out0).reshape(1, 4 + num_classes, -1)

    # --- Post-process -------------------------------------------------------
    detections = postprocess(raw, scale, pad_left, pad_top,
                             orig_w, orig_h, args.thresh, args.iou)

    # --- Draw results -------------------------------------------------------
    object_count = 0
    for (xmin, ymin, xmax, ymax, class_id, conf) in detections:
        color = BBOX_COLORS[class_id % 10]
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

        label = f'{labels[class_id]}: {int(conf * 100)}%'
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_ymin = max(ymin, lh + 10)
        cv2.rectangle(frame,
                      (xmin, label_ymin - lh - 10),
                      (xmin + lw,  label_ymin + baseline - 10),
                      color, cv2.FILLED)
        cv2.putText(frame, label, (xmin, label_ymin - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        object_count += 1

    if source_type in ('video', 'usb', 'picamera'):
        cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}',
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(frame, f'Objects: {object_count}',
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Resize to display resolution if it differs from capture resolution
    display_frame = frame
    if dispW and dispH and (dispW != frame.shape[1] or dispH != frame.shape[0]):
        display_frame = cv2.resize(frame, (dispW, dispH))

    cv2.imshow('YOLO11n NCNN', display_frame)

    if args.record:
        recorder.write(frame)

    # --- Key handling -------------------------------------------------------
    wait_ms = 0 if source_type in ('image', 'folder') else 5
    key = cv2.waitKey(wait_ms)
    if   key in (ord('q'), ord('Q')):   # quit
        break
    elif key in (ord('s'), ord('S')):   # pause / unpause
        cv2.waitKey()
    elif key in (ord('p'), ord('P')):   # screenshot
        cv2.imwrite('capture.png', frame)
        print('Saved capture.png')

    # --- FPS bookkeeping ----------------------------------------------------
    t_stop  = time.perf_counter()
    fps_now = 1.0 / max(t_stop - t_start, 1e-9)
    if len(frame_rate_buffer) >= FPS_AVG_LEN:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(fps_now)
    avg_frame_rate = float(np.mean(frame_rate_buffer))

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
net.clear()
if source_type in ('video', 'usb'):
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if args.record:
    recorder.release()
cv2.destroyAllWindows()