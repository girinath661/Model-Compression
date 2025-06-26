import argparse
import tensorflow
from tensorflow_model_optimization.python.core.keras.compat import keras
import tensorflow_model_optimization as tfmot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weightpath",type = str,required = True,help = "model weight path (normal.h5)")

    args = parser.parse_args()

    weightpath = args.weightpath

    model1 = keras.models.load_model(weightpath)

    # Convert the model
    converter = tensorflow.lite.TFLiteConverter.from_keras_model(model1) # loaded model
    converter.optimizations = [tensorflow.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()

    # Save the model
    with open(("normal_compressed.tflite"),mode="wb") as file:
        file.write(tflite_quant_model)
        
