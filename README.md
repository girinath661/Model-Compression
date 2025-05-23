# Model Compression & Quantization with TensorFlow Lite
This repository demonstrates how to apply  Post-Training Quantization (PTQ)  to compress and optimize a trained Keras model using TensorFlow Lite. The resulting models are lightweight, faster, and ideal for deployment on edge devices like mobile phones, IoT boards, and embedded systems.
# What can a Compressed and Quantized model do for me?
  •	Smaller Model Size: Converts float32 weights to lower precision (e.g., int8 or float16), significantly reducing model size.
  •	Faster Inference: Lwer-precision computations lead to faster execution on supported hardware.
  •	Lower Power Consumption: Ideal for battery-powered or resource-constrained devices.
  •	No Retraining Required: Quantization is applied after model training, making it quick and easy to implement.
# Getting started:
## 1.	 Step 1: Open Your Project in VS Code
  •	Launch Visual Studio Code.
  •	Open the folder where your project is located (File > Open Folder).
## 2.	 Step 2: Open a Terminal in VS Code
  •	Go to Terminal > New Terminal (or press Ctrl + `).
## 3.	 Step 3: Create the Environment
     •	Conda create -n environment_name python=3.12
## 4.	 Step 4: Activate Environment
     •	Conda activate environment_name
  •	You can see as {(environment_name) PS C:\Users\prade\OneDrive\Desktop\git quantization>}
## 5.	 Step 5: Install Project Dependencies
     •	pip install -r requirements.txt
