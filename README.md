# Deployment-of-AI-ML-Models-on-Edge-Devices

## aiml_models

- This folder acts like a repository for AIML models that we will use to run inference on edge devices.
- This folder has sub-folders on the basis of AIML application domain such as computer-vision, natural language processing, speech processing , etc. 
- Then inside each domain we have folders for particular task being performed such as in computer vision domain we have object classification, detection, segmentation, etc. 
- Further for each task we have sub-folders to classify the model files on the basis of processor hardware on which they will be executed. For example for object detection I have two sub-folders named cpu and hailo8_npu.
- Finally for each processor hardware, inside we have the necessary model files which need to be used whhile runing the inference. For the same processor hardware we may have more than one model format. For e.g., we have various model formats to un inference on CPU such as .pt, onnx, ncnn, etc. 

## deployment_pipelines

- In this folder first we navigate to the exact deployment setting so we start with the host device we are using (for e.g., raspberry_pi_5) and the processor we will use to offload the inference computation (for e.g., cpu / npu).
- Then inside each deployment setting we will have a setup.sh file to ensure relevant hardware settings to run the inference, we will have requirements.txt file to note down the necessary dependencies, we will have the inference code file named as (model_name)_(model_format)_inference.py and finally we wil have the readme.md file which will contain information about this particular deployment pipeline. 

1) look into how to clean the libraries and packages and modules installed in the root repo.
2) learn how to write custom inference scripts in order to minimize the memory overhead while running inference code and also to addmore functionality to our code such as lets say for an object etection model, when more than 5 people are detected it will send an alarm or take photo or something like that. 
3) make a setup.sh file for each deployment pipeline setting. 