from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import tensorflow as tf
import pathlib

if __name__ == "__main__":

    # Load the pre-trained ResNet50 model weights (without the top classification layer)
    model = ResNet50(weights='imagenet', include_top=True)

    # Load an image for classification
    img_path = 'data/imagenet/1.JPEG'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Make predictions
    predictions = model.predict(x)

    # Decode and print the top 5 predicted classes
    decoded_predictions = decode_predictions(predictions, top=5)[0]
    print('Predictions:')
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"{i+1}: {label} ({score:.2f})")

    model_dir='mnist_tflite_models'
    model_name = 'Resnet50.tflite'
    tflite_models_dir = pathlib.Path(model_dir)
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    tflite_model_file = tflite_models_dir / model_name

    # Convert the model to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_model_file.write_bytes(tflite_model)


    def representative_data_gen(train_images):
        for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
            yield [input_value]


    # Convert the model to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model_quant = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
    input_type = interpreter.get_input_details()[0]['dtype']
    print('input: ', input_type)
    output_type = interpreter.get_output_details()[0]['dtype']
    print('output: ', output_type)

    tflite_model_quant_file.write_bytes(tflite_model_quant)

