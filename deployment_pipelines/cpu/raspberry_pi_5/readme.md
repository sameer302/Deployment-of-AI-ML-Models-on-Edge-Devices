## Deploying YOLO Models on raspberry pi 5 CPU

- Following guide from [EjTech - How to Run YOLO Detection Models on the Raspberry Pi](https://www.ejtech.io/learn/yolo-on-raspberry-pi)

- Here we have inference.py code files to run inference on raspberry pi 5 cpu using various model formats such as ncnn, onnx, pt, etc. 
- We can export the model to particular format (ncnn, onnx, etc) on some other device as it would need using the heavy ultralytics module which would be memory heavy to install on pi 5. The exported model then can be copied to aiml models folder which will then be used in the inference.py script. 