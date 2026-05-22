# Deployment-of-AI-ML-Models-on-Edge-Devices

## quick start guide

- Navigate to the desired hardware deployment setting folder in the scripts folder.
- Open the bash script (.sh) file and choose the appropriate inference instance and comment out the other examples. Read more about the inference instance from deployment_setting folder so you can modify the arguments as per need. 
- Finally run the bash script using the command `bash ./file_name.sh`.

## aiml_models

- This folder acts like a repository for AIML models that we will use to run inference on edge devices.
- This folder has sub-folders on the basis of AIML application domain such as computer-vision, natural language processing, speech processing , etc. 
- Then inside each domain we have folders for particular task being performed such as in computer vision domain we have object classification, detection, segmentation, etc. 
- Further for each task we have sub-folders to classify the model files on the basis of processor hardware on which they will be executed. For example for object detection I have two sub-folders named cpu and hailo8_npu.
- Finally for each processor hardware, inside we have the necessary model files which need to be used whhile runing the inference. For the same processor hardware we may have more than one model format. For e.g., we have various model formats to un inference on CPU such as .pt, onnx, ncnn, etc. 

## deployment_pipelines

- In this folder first we navigate to the folder representing the processor on which we run the inference for e.g., cpu, npu, etc. Then we navigate to the folder representing the host device on which we are running the inference for e.g., raspberry pi 5.
- Then inside each deployment setting we will have a setup.sh file to ensure relevant hardware settings to run the inference, we will have requirements.txt file to note down the necessary dependencies, we will have the inference code file named as (model_name)_(model_format)_inference.py and finally we wil have the readme.md file which will contain information about this particular deployment pipeline. 

## scripts

- In this folder we navigate down to each deployment setting such as cpu and then inside it for raspberry pi 5
- For each of these setting we then have a bash script which contains code to run inference for different scenarios as mentioned in the corresponding bash script. 
 
2) learn how to write custom inference scripts in order to minimize the memory overhead while running inference code and also to addmore functionality to our code such as lets say for an object etection model, when more than 5 people are detected it will send an alarm or take photo or something like that. 
3) make a setup.sh file for each deployment pipeline setting. 
