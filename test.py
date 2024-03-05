import os
import tensorflow as tf
import numpy as np
from utils import test_model, load_dataset, evaluate_model, load_yaml_file

if __name__ == "__main__":
    yaml_file_path = 'config.yaml'  # Path to your YAML file
    config = load_yaml_file(yaml_file_path)

    # Load MNIST dataset
    train_images, train_labels, test_images, test_labels = load_dataset('mnist')

    # Change this to test a different image
    test_image_index = 1

    tflite_model_file = os.path.join(config['MODEL_DIR'], config['MODEL_FILE'])
    tflite_model_quant_path = os.path.join(config['MODEL_DIR'], config['QUANT_MODEL_FILE'])

    # test model on images and visualize the results
    test_model(tflite_model_file, test_image_index, model_type="Float", test_images=test_images, test_labels=test_labels)

    # test quantized model on images and visualize the results
    test_model(tflite_model_quant_path, test_image_index, model_type="Quantized", test_images=test_images, test_labels=test_labels)

    # evaluate model performance and benchmark
    print("Evaluation float model...")
    evaluate_model(tflite_model_file, model_type="Float", test_images=test_images, test_labels=test_labels)

    # evaluate quantized model performance and benchmark
    print("Evaluation quantized model...")
    evaluate_model(tflite_model_quant_path, model_type="Quantized", test_images=test_images, test_labels=test_labels)