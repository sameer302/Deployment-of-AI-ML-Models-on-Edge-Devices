
CURRENT_DIR=$(pwd)
cd /home/sameer/Desktop/Edge_AI/Deployment-of-AI-ML-Models-on-Edge-Devices/deployment_pipelines/cpu/raspberry_pi_5/ncnn
source venv/bin/activate

# Set the duration for running the inference script (format: Xm for minutes, Xs for seconds)
# Examples: 7m (7 minutes), 30s (30 seconds), 2h (2 hours)
TIME_DURATION="7m"

# Uncomment other inference scripts and run only the one you want to test. Make sure to update the model path and source as needed.

# YOLOV11 inference using ncnn format
# The script will run for the specified TIME_DURATION and then automatically stop
timeout $TIME_DURATION python yolo11_ncnn_inference.py \
    --model /home/sameer/Desktop/Edge_AI/Deployment-of-AI-ML-Models-on-Edge-Devices/aiml_models/pre-built_models/vision/object_detection/cpu/yolo11n_ncnn_model \
    --source picamera0 \
    --thresh 0.5 \
    --iou 0.45 \
    --resolution 1296x972 \
    --log-latency \
    --log-inference-latency \
    --log-fps \
    --log-inference-fps \
    --csv-path /home/sameer/Desktop/Edge_AI/Deployment-of-AI-ML-Models-on-Edge-Devices/deployment_pipelines/cpu/raspberry_pi_5/ncnn/yolo11_ncnn_metrics.csv \
    # --frame-window-size 30
    # --display 640x360 \
    # --record

deactivate
cd $CURRENT_DIR
