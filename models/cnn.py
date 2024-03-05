import tensorflow as tf
import numpy as np
import pathlib

class CNN:
    def __init__(self):
        self.model = None

    def make_model(self):
        # Define the model architecture
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28, 28)),
            tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10)
        ])

    def train_model(self, train_images, train_labels, test_images, test_labels, epochs=5):
        # Train the digit classification model
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        self.model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))

    def representative_data_gen(train_images):
        for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
            yield [input_value]

    def convert_to_tflite(self, model_name, model_dir='mnist_tflite_models'):
        # Save the TensorFlow Lite model
        tflite_models_dir = pathlib.Path(model_dir)
        tflite_models_dir.mkdir(exist_ok=True, parents=True)
        tflite_model_file = tflite_models_dir / model_name

        # Convert the model to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        tflite_model_file.write_bytes(tflite_model)

        return tflite_model_file

    def convert_to_tflite_quant(self, representative_data_gen, model_name, model_dir='mnist_tflite_models'):
        # Save the TensorFlow Lite model
        tflite_models_dir = pathlib.Path(model_dir)
        tflite_models_dir.mkdir(exist_ok=True, parents=True)
        tflite_model_quant_file = tflite_models_dir / model_name

        # Convert the model to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
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

        return tflite_model_quant_file