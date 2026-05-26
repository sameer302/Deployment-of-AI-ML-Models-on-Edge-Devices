## Deploying YOLO models on raspberry pi 5 CPU

- The specifications of hardware being used here is as follows:-
    1) Raspberry Pi 5, 16GB RAM, SBC. [Product sheet](https://pip-assets.raspberrypi.com/categories/892-raspberry-pi-5/documents/RP-008348-DS-6-raspberry-pi-5-product-brief.pdf?disposition=inline)
    2) Arducam 5MP ov5647 camera module. [Product sheet.](https://docs.arducam.com/Raspberry-Pi-Camera/Native-camera/5MP-OV5647/#raspberry-pi-compute-module-3-4) 
    3) CSI ribbon to connect camera module with raspberry pi 5. 
    4) 27W USB-C official raspberry pi 5 adapter and charger.
    5) micro HDMI to HDMI cable to connect with display monitor. 

- Here we have inference.py code files to run inference on raspberry pi 5 cpu using various model formats such as ncnn, onnx, pt, etc. 

- For each model architecture and format combination, we have different inference code files. 

- We can export the model to particular format (ncnn, onnx, etc) on some other device as it would need using the heavy ultralytics module which would be memory heavy to install on pi 5. The exported model then can be copied to aiml models folder which will then be used in the inference.py script. 

### Deploying YOLOv11 models in ncnn format on raspberry pi 5 CPU

- Following guide from [EjTech - How to Run YOLO Detection Models on the Raspberry Pi](https://www.ejtech.io/learn/yolo-on-raspberry-pi)

- The yolo_detect.py file provided in the above blog has been modified by removing dependency on Ultralytics library as it is very heavy in terms of memory requirements and necessary for training the model. For running inference, I added custom pre and post processing code and finally made the yolo11_ncnn_inference.py file. 

- Here we have freedom to choose following arguments for each parameter:-
    1) --model
        -  Path to NCNN model folder (contains model.ncnn.param, model.ncnn.bin, metadata.yaml)
        - we can use any size variant of YOLOv11 model such as n, s, m, l, x depending on accuracy and speed need.
        - we can use YOLOv11 model with different input resolution and number of classes as the script directly reads this from the metadata.yaml file
    2) --source 
        - Image file, image folder, video file, usb<N>, or picamera<N>
    3) --thresh
        - Confidence threshold (default: 0.5)
    4) --iou
        - NMS IoU threshold (default: 0.45)
    5) --resolution
        - Capture resolution WxH, e.g. 1280x720 (required for picamera/usb/record)
        - We can use any available resolution with our camera. 
    6) --display
        - Display window resolution WxH, e.g. 640x480 (optional — defaults to capture resolution if not set)
    7) --record
        -  Record output to demo1.avi (requires --resolution)

- The `yolo11_ncnn_inference.py` script can be used only for models whose input and output tensor formats match to yolo11. The inference engine ncnn.net is generic but preprocessing + output decoding are model-specific. Hence to adopt this script to other detectors we would need to modify preprocessing, layer names and postprocess decoding logic. 