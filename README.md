# TensorflowLite (8-bit) Quantization

This repository is related to the quantization of convolution neural networks using TensorFlow Lite library. In this tutorial, we will use MNIST dataset for faster training of a simple CNN for image classification task.

## Directory Structure
After cloning the repository, the directory structure will look like this,

```
├── config.yaml
├── models
│   ├── cnn.py
├── README.md
├── test.py
├── train.py
└── utils.py
```

## Train
Set up the path in the [config.yaml](/config.yaml) and run the following command to start training,
```
python train.py
```
This script will train a simple CNN model, convert into `.tflite` model format, and quantize it into `8-bit`.

## Test and Evaluate
Run the following command to start testing,
```
python test.py
```
This script will compare the float32 and 8-bit (quantized) model.

## Model customization
You can change the model in [models/cnn.py](/models/cnn.py).

## Upcoming
- Support for CIFAR10/CIFAR100 dataset.
- Support for ImageNet dataset.
- Support for latest CNN's.

---

Please feel free to contribute. Thanks!