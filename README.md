# targeted-adversarial-mnist
Adversarial attack on a CNN trained on MNIST dataset using Iterative Targeted Fast Gradient Sign Method

### Dependencies
- Tensorflow
- numpy

The `model.py` file defines the architecture and saves the trained model.

## Architecture

- Convolutional layer 1: 32 `5x5x1` kernels
- Relu activation
- Standard Max Pooling
- Convolutional layer 2: 64 `5x5x32` kernels
- Relu activation
- Standard Max Pooling
- Fully Connected Layer 1 with `1024` out units
- Relu activation
- Dropout
- Fully Connected Layer 2 with `10` out units (representing 10 classes of the dataset)

The `adversary.py` file creates the adversarial examples.  It takes 2 arguments
- `--input_class` or `-i`
- `--target_class` or `-t`

Input class is the actual label of the input image
Target class is the label that we want the network to predict for the input image

The image is modified by taking the gradient of the cost function w.r.t the input.
![equation](https://image.ibb.co/cHaamS/ifgsm.png)

The pre-trained model is present in the `model` folder. So, the adversary script can be run directly.
`python adversary.py -i 2 -t 6`

The default parameters are: `e=0.01` and `num_steps=25`.

## Result
![result](image.jpg)

This result can be fine tuned by conditional update depending on the classification outcome. 

### TO-DO
- Refactor
- One pixel attack with Differential Evolution
