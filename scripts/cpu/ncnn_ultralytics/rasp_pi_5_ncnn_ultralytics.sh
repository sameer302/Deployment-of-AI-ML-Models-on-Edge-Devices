
CURRENT_DIR=$(pwd)
cd /home/sameer/Desktop/Edge_AI/Deployment-of-AI-ML-Models-on-Edge-Devices/deployment_pipelines/cpu/raspberry_pi_5/ncnn_ultralytics
source venv/bin/activate

# Set the duration for running the inference script (format: Xm for minutes, Xs for seconds)
# Examples: 7m (7 minutes), 30s (30 seconds), 2h (2 hours)
TIME_DURATION="7m"

# YOLOV11 inference using ncnn format (ultralytics library)
timeout $TIME_DURATION python yolo11_ultralytics_ncnn_inference.py \
    --model /home/sameer/Desktop/Edge_AI/Deployment-of-AI-ML-Models-on-Edge-Devices/aiml_models/pre-built_models/vision/object_detection/cpu/yolo11n_ncnn_model \
    --source picamera0 \
    --thresh 0.5 \
    --iou 0.45 \
    --resolution 1296x972 \
    --log-latency \
    --log-inference-latency \
    --log-fps \
    --log-inference-fps \
    --csv-path /home/sameer/Desktop/Edge_AI/Deployment-of-AI-ML-Models-on-Edge-Devices/deployment_pipelines/cpu/raspberry_pi_5/ncnn_ultralytics/yolo11_ultralytics_ncnn_metrics.csv
    # --frame-window-size 30
    # --display 640x360 \
    # --record

deactivate
cd $CURRENT_DIR
