import yaml
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf

def load_yaml_file(file_path):
    with open(file_path, 'r') as yaml_file:
        try:
            data = yaml.safe_load(yaml_file)
            return data
        except yaml.YAMLError as e:
            print("Error loading YAML file:", e)
            return None
        
def load_dataset(dataset):
    
    if dataset == 'mnist':
        # Load MNIST dataset
        mnist = tf.keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        # Normalize the input image so that each pixel value is between 0 to 1.
        train_images = train_images.astype(np.float32) / 255.0
        test_images = test_images.astype(np.float32) / 255.0
        return train_images, train_labels, test_images, test_labels
    else:
        return None, None, None, None
     
# Helper function to run inference on a TFLite model
def run_tflite_model(tflite_file, test_image_indices, test_images):

  # Initialize the interpreter
  interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  predictions = np.zeros((len(test_image_indices),), dtype=int)
  for i, test_image_index in enumerate(test_image_indices):
    test_image = test_images[test_image_index]

    # Check if the input type is quantized, then rescale input data to uint8
    if input_details['dtype'] == np.uint8:
      input_scale, input_zero_point = input_details["quantization"]
      test_image = test_image / input_scale + input_zero_point

    test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]

    predictions[i] = output.argmax()

  return predictions

## Helper function to test the models on one image
def test_model(tflite_file, test_image_index, model_type, test_images, test_labels):

  predictions = run_tflite_model(tflite_file, [test_image_index], test_images)

  plt.imshow(test_images[test_image_index])
  template = model_type + " Model \n True:{true}, Predicted:{predict}"
  _ = plt.title(template.format(true= str(test_labels[test_image_index]), predict=str(predictions[0])))
  plt.grid(False)
  plt.show()

# Helper function to evaluate a TFLite model on all images
def evaluate_model(tflite_file, model_type, test_images, test_labels):

  test_image_indices = range(test_images.shape[0])
  predictions = run_tflite_model(tflite_file, test_image_indices, test_images)

  accuracy = (np.sum(test_labels== predictions) * 100) / len(test_images)

  print('%s model accuracy is %.4f%% (Number of test samples=%d)' % (
      model_type, accuracy, len(test_images)))