# Don't uncomment the below section of code.

CURRENT_DIR=$(pwd)
cd /home/sameer/Desktop/Edge_AI/Deployment_Examples/deployment_pipelines/cpu
source ./venv/bin/activate
cd raspberry_pi_5

# Uncomment other inference scripts and run only the one you want to test. Make sure to update the model path and source as needed.

# YOLOV11 inference using ncnn format
python yolo11_ncnn_inference.py \
    --model /home/sameer/Desktop/Edge_AI/Deployment_Examples/aiml_models/pre-built_models/vision/object_detection/cpu/yolo11n_ncnn_model \
    --source picamera0 \
    --thresh 0.5 \
    --iou 0.45 \
    --resolution 1296x972 \
    # --display 640x360 \
    # --record

# Don't uncomment the below section of code

cd $CURRENT_DIR