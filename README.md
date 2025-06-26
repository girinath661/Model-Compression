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

## Contents of requirements.txt:
```
tensorflow
tensorflow-model-optimization
pandas
numpy
scikit-learn
```
📁 File Descriptions
- model.py: Trains a binary classification model using clustering on a preprocessed dataset.

- compressed.py: Compresses the trained model using TensorFlow Lite optimization.

- compressed_quantization.py: Applies quantization-aware training to further optimize the model.

- normal.h5: Output model from training.

- normal_compressed.tflite: Compressed TFLite model.

- normal_compressed_quantized.tflite: Quantized and compressed TFLite model.

⚙️ Step-by-Step Instructions
1. Clone the Repository and Prepare Environment
   git clone <your-repo-url>
   cd <your-repo-dir>
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt

2. Train the Model
Ensure Assessment - Form Responses.csv is available in the same directory as model.py.
```
python model.py
```
This will generate a model file: normal.h5.
3. Compress the Trained Model
```
python compressed.py --weightpath normal.h5
```
This will create: normal_compressed.tflite
4. Quantize the Compressed Model
```
python compressed_quantization.py --weightpath normal.h5
```
This will produce: normal_compressed_quantized.tflite

✅ Notes
- Make sure the normal.h5 model exists before running compression or quantization scripts.

- TFLite models are suitable for deployment on edge devices with reduced size and inference time.



