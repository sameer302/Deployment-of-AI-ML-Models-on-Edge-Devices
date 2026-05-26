
"""
yolo_ultralytics_advanced.py
============================
Advanced YOLO inference pipeline using Ultralytics.

Features:
- Image / folder / video / USB camera / Picamera support
- Recording
- Display resolution control
- Graceful shutdown
- CSV metrics logging
- End-to-end latency logging
- Inference latency logging
- Rolling FPS logging
- Warmup period support

Install:
    pip install ultralytics opencv-python numpy

Usage:
    python yolo_ultralytics_advanced.py --model best.pt --source usb0 --resolution 1280x720

    python yolo_ultralytics_advanced.py \
        --model best.pt \
        --source usb0 \
        --resolution 1280x720 \
        --display 640x480 \
        --log-latency \
        --log-inference-latency \
        --log-fps \
        --log-inference-fps \
        --csv-path metrics.csv
"""

import os
import sys
import cv2
import csv
import glob
import time
import signal
import argparse
import datetime
import numpy as np

from collections import deque
from ultralytics import YOLO

# -------------------------------------------------------------------
# Graceful shutdown
# -------------------------------------------------------------------

_shutdown_requested = False

def _request_shutdown(signum, frame):
    global _shutdown_requested
    print(f'\n[INFO] Signal {signum} received. Shutting down gracefully...')
    _shutdown_requested = True

signal.signal(signal.SIGINT, _request_shutdown)
signal.signal(signal.SIGTERM, _request_shutdown)

# -------------------------------------------------------------------
# Argument parser
# -------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument('--model', required=True,
                    help='Path to YOLO model')

parser.add_argument('--source', required=True,
                    help='Image, folder, video, usb<N>, or picamera<N>')

parser.add_argument('--thresh', type=float, default=0.5,
                    help='Confidence threshold')
                    
parser.add_argument('--iou', type=float, default=0.45,
                    help='IoU threshold')

parser.add_argument('--resolution',
                    help='Capture resolution WxH')

parser.add_argument('--display',
                    help='Display resolution WxH')

parser.add_argument('--record', action='store_true',
                    help='Record output video')

# Metrics logging

parser.add_argument('--log-latency', action='store_true')
parser.add_argument('--log-inference-latency', action='store_true')
parser.add_argument('--log-fps', action='store_true')
parser.add_argument('--log-inference-fps', action='store_true')

parser.add_argument('--csv-path', type=str, default=None)

parser.add_argument('--frame-window-size', type=int, default=30)

args = parser.parse_args()

# -------------------------------------------------------------------
# Validate logging args
# -------------------------------------------------------------------

any_logging = (
    args.log_latency or
    args.log_inference_latency or
    args.log_fps or
    args.log_inference_fps
)

if any_logging and not args.csv_path:
    print('ERROR: --csv-path required when using log flags')
    sys.exit(1)

# -------------------------------------------------------------------
# Load model
# -------------------------------------------------------------------

if not os.path.exists(args.model):
    print('ERROR: Model file not found')
    sys.exit(1)

print('[INFO] Loading model...')
model = YOLO(args.model, task='detect')
labels = model.names
print('[INFO] Model loaded.')

# -------------------------------------------------------------------
# Detect source type
# -------------------------------------------------------------------

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
        print(f'Unsupported extension: {ext}')
        sys.exit(1)

elif img_source.startswith('usb'):
    source_type = 'usb'
    usb_idx = int(img_source[3:])

elif img_source.startswith('picamera'):
    source_type = 'picamera'
    picam_idx = int(img_source[8:])

else:
    print(f'Invalid source: {img_source}')
    sys.exit(1)

# -------------------------------------------------------------------
# Resolution setup
# -------------------------------------------------------------------

resize = False
resW = resH = None

if args.resolution:
    resize = True
    resW, resH = map(int, args.resolution.split('x'))

dispW = dispH = None

if args.display:
    dispW, dispH = map(int, args.display.split('x'))

elif args.resolution:
    dispW, dispH = resW, resH

# -------------------------------------------------------------------
# Recording setup
# -------------------------------------------------------------------

if args.record:

    if source_type not in ('video', 'usb'):
        print('Recording only supported for video/camera')
        sys.exit(1)

    if not args.resolution:
        print('Please specify --resolution for recording')
        sys.exit(1)

    recorder = cv2.VideoWriter(
        'demo1.avi',
        cv2.VideoWriter_fourcc(*'MJPG'),
        30,
        (resW, resH)
    )

# -------------------------------------------------------------------
# Open source
# -------------------------------------------------------------------

if source_type == 'image':

    imgs_list = [img_source]

elif source_type == 'folder':

    imgs_list = [
        f for f in glob.glob(os.path.join(img_source, '*'))
        if os.path.splitext(f)[1] in IMG_EXTS
    ]

elif source_type in ('video', 'usb'):

    cap_arg = img_source if source_type == 'video' else usb_idx

    cap = cv2.VideoCapture(cap_arg)

    if resize:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)

elif source_type == 'picamera':

    from picamera2 import Picamera2

    cap = Picamera2()

    cap.configure(
        cap.create_video_configuration(
            main={"format": 'RGB888', "size": (resW, resH)}
        )
    )

    cap.start()

# -------------------------------------------------------------------
# CSV setup
# -------------------------------------------------------------------

csv_file = None
csv_writer = None

if any_logging:

    columns = ['timestamp']

    if args.log_latency:
        columns.append('end_to_end_latency_ms')

    if args.log_inference_latency:
        columns.append('inference_latency_ms')

    if args.log_fps:
        columns.append('end_to_end_fps')

    if args.log_inference_fps:
        columns.append('inference_fps')

    csv_dir = os.path.dirname(args.csv_path)

    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)

    csv_file = open(args.csv_path, 'w', newline='')

    csv_writer = csv.DictWriter(csv_file, fieldnames=columns)

    csv_writer.writeheader()

    print(f'[INFO] Logging metrics to: {args.csv_path}')

# -------------------------------------------------------------------
# Metrics setup
# -------------------------------------------------------------------

WINDOW_SIZE = args.frame_window_size

e2e_window = deque(maxlen=WINDOW_SIZE)
infer_window = deque(maxlen=WINDOW_SIZE)

WARMUP_SECONDS = 120
loop_start_time = None

# -------------------------------------------------------------------
# Colors
# -------------------------------------------------------------------

BBOX_COLORS = [
    (164,120,87),
    (68,148,228),
    (93,97,209),
    (178,182,133),
    (88,159,106),
    (96,202,231),
    (159,124,168),
    (169,162,241),
    (98,118,150),
    (172,176,184),
]

# -------------------------------------------------------------------
# Main loop
# -------------------------------------------------------------------

avg_frame_rate = 0
frame_rate_buffer = []
FPS_AVG_LEN = 200

img_count = 0

while not _shutdown_requested:

    t_frame_start = time.perf_counter()

    if loop_start_time is None:
        loop_start_time = time.time()

    # ---------------------------------------------------------------
    # Load frame
    # ---------------------------------------------------------------

    if source_type in ('image', 'folder'):

        if img_count >= len(imgs_list):
            print('All images processed.')
            break

        frame = cv2.imread(imgs_list[img_count])

        img_count += 1

        if frame is None:
            continue

    elif source_type == 'video':

        ret, frame = cap.read()

        if not ret:
            print('Video ended.')
            break

    elif source_type == 'usb':

        ret, frame = cap.read()

        if not ret or frame is None:
            print('Camera read failed.')
            break

    elif source_type == 'picamera':

        frame = cap.capture_array()

        if frame is None:
            print('Picamera read failed.')
            break

    # ---------------------------------------------------------------
    # Resize
    # ---------------------------------------------------------------

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # ---------------------------------------------------------------
    # Inference timing
    # ---------------------------------------------------------------

    t_infer_start = time.perf_counter()

    results = model(
    frame,
    verbose=False,
    conf=args.thresh,
    iou=args.iou
    )

    t_infer_end = time.perf_counter()

    inference_latency_s = t_infer_end - t_infer_start

    detections = results[0].boxes

    # ---------------------------------------------------------------
    # Draw detections
    # ---------------------------------------------------------------

    object_count = 0

    for i in range(len(detections)):

        conf = detections[i].conf.item()

        if conf < args.thresh:
            continue

        xyxy = detections[i].xyxy.cpu().numpy().squeeze()

        xmin, ymin, xmax, ymax = xyxy.astype(int)

        classidx = int(detections[i].cls.item())

        classname = labels[classidx]

        color = BBOX_COLORS[classidx % 10]

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

        label = f'{classname}: {int(conf*100)}%'

        labelSize, baseLine = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1
        )

        label_ymin = max(ymin, labelSize[1] + 10)

        cv2.rectangle(
            frame,
            (xmin, label_ymin-labelSize[1]-10),
            (xmin+labelSize[0], label_ymin+baseLine-10),
            color,
            cv2.FILLED
        )

        cv2.putText(
            frame,
            label,
            (xmin, label_ymin-7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,0,0),
            1
        )

        object_count += 1

    # ---------------------------------------------------------------
    # FPS display
    # ---------------------------------------------------------------

    if source_type in ('video', 'usb', 'picamera'):

        cv2.putText(
            frame,
            f'FPS: {avg_frame_rate:.2f}',
            (10,20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,255),
            2
        )

    cv2.putText(
        frame,
        f'Objects: {object_count}',
        (10,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,255),
        2
    )

    # ---------------------------------------------------------------
    # Display resize
    # ---------------------------------------------------------------

    display_frame = frame

    if dispW and dispH:
        if display_frame.shape[1] != dispW or display_frame.shape[0] != dispH:
            display_frame = cv2.resize(display_frame, (dispW, dispH))

    # ---------------------------------------------------------------
    # End-to-end timing
    # ---------------------------------------------------------------

    t_frame_end = time.perf_counter()

    e2e_latency_s = t_frame_end - t_frame_start

    # ---------------------------------------------------------------
    # Show frame
    # ---------------------------------------------------------------

    cv2.imshow('Ultralytics YOLO', display_frame)

    # ---------------------------------------------------------------
    # Recording
    # ---------------------------------------------------------------

    if args.record:
        recorder.write(frame)

    # ---------------------------------------------------------------
    # Key handling
    # ---------------------------------------------------------------

    wait_ms = 0 if source_type in ('image', 'folder') else 5

    key = cv2.waitKey(wait_ms) & 0xFF

    if key in (ord('q'), ord('Q')):
        break

    elif key in (ord('s'), ord('S')):
        cv2.waitKey(0)

    elif key in (ord('p'), ord('P')):
        cv2.imwrite('capture.png', frame)
        print('Saved capture.png')

    try:
        if cv2.getWindowProperty('Ultralytics YOLO', cv2.WND_PROP_VISIBLE) < 1:
            break
    except:
        break

    # ---------------------------------------------------------------
    # FPS calculation
    # ---------------------------------------------------------------

    t_stop = time.perf_counter()

    fps_now = 1.0 / max(t_stop - t_frame_start, 1e-9)

    if len(frame_rate_buffer) >= FPS_AVG_LEN:
        frame_rate_buffer.pop(0)

    frame_rate_buffer.append(fps_now)

    avg_frame_rate = float(np.mean(frame_rate_buffer))

    # ---------------------------------------------------------------
    # Rolling windows
    # ---------------------------------------------------------------

    current_perf = time.perf_counter()

    e2e_window.append(current_perf)

    infer_window.append(t_infer_end)

    e2e_fps = None
    infer_fps = None

    if len(e2e_window) == WINDOW_SIZE:

        elapsed_e2e = e2e_window[-1] - e2e_window[0]

        if elapsed_e2e > 0:
            e2e_fps = (WINDOW_SIZE - 1) / elapsed_e2e

    if len(infer_window) == WINDOW_SIZE:

        elapsed_infer = infer_window[-1] - infer_window[0]

        if elapsed_infer > 0:
            infer_fps = (WINDOW_SIZE - 1) / elapsed_infer

    # ---------------------------------------------------------------
    # CSV logging
    # ---------------------------------------------------------------

    if any_logging and csv_writer is not None:

        elapsed_since_start = time.time() - loop_start_time

        if elapsed_since_start >= WARMUP_SECONDS:

            ts = datetime.datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S.%f'
            )[:-3]

            row = {'timestamp': ts}

            if args.log_latency:
                row['end_to_end_latency_ms'] = round(
                    e2e_latency_s * 1000,
                    3
                )

            if args.log_inference_latency:
                row['inference_latency_ms'] = round(
                    inference_latency_s * 1000,
                    3
                )

            if args.log_fps:
                row['end_to_end_fps'] = (
                    round(e2e_fps, 3)
                    if e2e_fps is not None else ''
                )

            if args.log_inference_fps:
                row['inference_fps'] = (
                    round(infer_fps, 3)
                    if infer_fps is not None else ''
                )

            csv_writer.writerow(row)

            csv_file.flush()

# -------------------------------------------------------------------
# Cleanup
# -------------------------------------------------------------------

print(f'Average pipeline FPS: {avg_frame_rate:.2f}')

if source_type in ('video', 'usb'):
    cap.release()

elif source_type == 'picamera':
    cap.stop()

if args.record:
    recorder.release()

if csv_file is not None:
    csv_file.close()
    print(f'[INFO] Metrics saved to: {args.csv_path}')

cv2.destroyAllWindows()
