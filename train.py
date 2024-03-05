import os
import tensorflow as tf
import numpy as np

from utils import load_dataset, load_yaml_file
from models.cnn import CNN

def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
      yield [input_value]

if __name__ == '__main__':
  yaml_file_path = 'config.yaml'  # Path to your YAML file
  config = load_yaml_file(yaml_file_path)

  tflite_models_dir = config['MODEL_DIR']

  if not os.path.exists(tflite_models_dir):
    try:
        os.makedirs(tflite_models_dir)
        print(f"Directory '{tflite_models_dir}' created successfully.")
    except OSError as e:
        print(f"Error creating directory '{tflite_models_dir}': {e}")
  else:
      print(f"Directory '{tflite_models_dir}' already exists.")

  # Load MNIST dataset
  train_images, train_labels, test_images, test_labels = load_dataset(config['DATASET'])

  cnn = CNN()
  cnn.make_model()
  cnn.train_model(train_images, train_labels, test_images, test_labels, epochs=config['EPOCH'])

  # Convert and save the model from keras to tflite
  tflite_model_path = cnn.convert_to_tflite(model_name=config['MODEL_FILE'], model_dir=tflite_models_dir)
  print("Saved TensorFlow Lite model to:", tflite_model_path)

  # Convert and save the model from keras to tflite and quantize to 8-bit
  # pass the calibration dataset for quantization to 8-bit
  tflite_quant_model_path = cnn.convert_to_tflite_quant(representative_data_gen, model_name=config['QUANT_MODEL_FILE'], model_dir=tflite_models_dir)
  print("Saved TensorFlow Lite quantized model to:", tflite_quant_model_path)