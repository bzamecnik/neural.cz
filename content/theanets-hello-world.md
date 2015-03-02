Title: Theanets Hello world: simple classification example
Slug: theanets-hello-world
Date: 2015-03-03
Category: Machine Learning
Tags: Python, Neural networks, Theano, Theanets, Machine Learning
Summary: <div class="row">
		<div class="col-md-6">
			<img src="http://i.neural.cz/theanets-hello-world/theanets_console.png" class="thumbnail">
		</div>
		<div class="col-md-6">
			<p>How to train a simple neural network with the theanets library in a few lines of code?</p>
			<p>In this working example we generate simple synthetic data, create a classifier, train it and evaluate it. Then you can start with your own experiments with much more powerful networks.</p>
		</div>
	</div>

<div class="row">
	<div class="col-md-6">
		<img src="http://i.neural.cz/theanets-hello-world/theanets_console.png" class="thumbnail">
	</div>
	<div class="col-md-6">
		<p>How to train a simple neural network with the theanets library in a few lines of code?</p>
		<p>In this working example we generate simple synthetic data, create a classifier, train it and evaluate it. Then you can start with your own experiments with much more powerful networks.</p>
	</div>
</div>

It assumes knowing a bit of theory of machine learning and neural networks. After the [Andrew Ng's Machine Learning course](https://www.coursera.org/course/ml) at Coursera you shoul feel at home.

## Why neural networks and `theanets`?

Neural networks are a very powerful tool in machine learning. Recently with the advent of deep learning architectures and efficient algorithms for training them, deep neural networks allowed solving complicated problems such as speech or face recognition with state-of-the-art results. Thanks to vast amounts of training data, GPUs (pocket-sized super-computers) and a lot of software available we can apply deep neural network models to problems in many domains.

However, a difficult problem for a beginner is to start and get the algorithm working. There's an abundance of papers and software libraries available and it might be hard to decide where to start. Although it is possible to write a some neural network model from scratch there's a lot of tricky details so it's much easier to pick a some existing library and learn from it as much as possible, and later extend it or write one's own implementation.

In this article we'll go through a basic example of using the [theanets](https://github.com/lmjohns3/theanets) library which allows to build and train
various neural networks on top of the [theano](https://github.com/Theano/Theano)
compiler. Altough we write pure Python code Theano compiles it and allows to run the code either on a CPU or a GPU (it also takes care of automatic differentiation, memory management and other goodies). Theanets then allows to build models of neural networks with various architectures (shallow/deep, feed-forward/recurrent), activation functions, training algorithms, etc. Besides it we'll use bits of the [scikit-learn](http://scikit-learn.org) library, in this case for generating synthetic data and for model evaluation.

## Installation

I was able to install and run theano, theanets and scikit-learns on Python 3.4 on Mac OS X Yosemite. You need the classic SciPy stack (numpy, scipy, matplotlib). Scikit-learn and IPython and optional but recommended. Usage of pyvenv/virtualenv is also recommended.

[Theano installation instructions](http://deeplearning.net/software/theano/install.html). You can either install the release from the repository or the development code from GitHub. Python 3 seems to be supported since version 0.6.0rc5 (not in 0.6.0 - the last "stable" release).

```bash
pip install theano==0.6.0rc5
# or
pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
```

Then test it via:

```bash
$ python
```
```python
>>> import theano
```

[Theanets installation instructions](https://github.com/lmjohns3/theanets). Install theanets release or get the development code via git:

	pip install theanets

Voilà.

## Overview

The goal is to get familiar with theanets on some simple example. You can
modify this example bit by bit to work on more complex data and models.

In this example we generate some synthetic data (via scikit-learn) - two 2D
blobs with Gaussian distribution which are in addition linearly separable.
Thus any classification model should have no problem with such data.

We create a neural network with three layers and train it.

Finally, we evaluate the model with functions provided by scikit-learn.

## The code

The complete runnable code from this article is available on GitHub as a single Python script.

- [theanets_hello_world.py](https://github.com/bzamecnik/ml-playground/blob/master/theanets/theanets_hello_world.py)

### Preparing data

We prepare very simple data - two linearly separable 2D blobs.

```python
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
n_samples = 1000
# centers - number of classes
# n_features - dimension of the data
X, y = make_blobs(n_samples=n_samples, centers=2, n_features=2, \
    cluster_std=0.5, random_state=0)
```

We need to convert the features and targets to the 32-bit format suitable for the Theano. It is possible to configure the float precision used by Theano, but 32-bit precision is enough in this case.

```python
X = X.astype(np.float32)
y = y.astype(np.int32)
```

Tip: See what happens (later when you create the model) when you pass it data in a not supported format. (An exeption is thrown.)

### Visualizing data

To get familiar with the data and assure that it is alrigth we might visualize it. In this 2D case it is quite simple - a scatter plot is sufficient. We can make the points transparent and get rid of the border to better see the density.

```python
import matplotlib.pyplot as plt
def plot_2d_blobs(dataset):
    X, y = dataset
    plt.axis('equal')
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.1, edgecolors='none')

plot_2d_blobs((X, y))
```

<img src="http://i.neural.cz/theanets-hello-world/2d_blobs.png" class="thumbnail">

Tip: For more examples of assesing the dataset and visualizing it for better understanding you can check out the previous article [Dataset exploration: Boston house pricing](dataset-exploration-boston-house-pricing.html)

### Splitting it to datasets

In order to train and evaluate the model we need to split the data set into
training, validation and testing set. Training set can be used to optimize general model parameters and validation training set for optimizing some hyperparameters. Testing set can be used only for evaluation of performance on unseen data, never for optimization (including training).

Note that for splitting the data must be randomly shuffled, in this case the data was already shuffled when being generated. Scikit-learn provides [ShuffleSplit](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.ShuffleSplit.html) for this purpose. Unfortunately it only splits data into two sets (training, test), not three. So I wrote this simple splitting function instead.

```python
import numpy as np
def split_data(X, y, slices):
    '''
    Splits the data into training, validation and test sets.
    slices - relative sizes of each set (training, validation, test)
        test - provide None, since it is computed automatically
    '''
    datasets = {}
    starts = np.floor(np.cumsum(len(X) * np.hstack([0, slices[:-1]])))
    slices = {
        'training': slice(starts[0], starts[1]),
        'validation': slice(starts[1], starts[2]),
        'test': slice(starts[2], None)}
    data = X, y
    def slice_data(data, sl):
        return tuple(d[sl] for d in data)
    for label in slices:
        datasets[label] = slice_data(data, slices[label])
    return datasets
```

Generally speaking, split to ratios 60% / 20% / 20% are a good start. The dataset can be split into three parts with the above ratios like this:

```python
datasets = split_data(X, y, (0.6, 0.2, None))
## the data is then structured like this:
# datasets['training'] == (X_training, y_training)
# datasets['validation'] == (X_validation, y_validation)
# datasets['test'] == (X_test, y_test)
```

### Creating the model

First we setup logging and argument handling for theanets via the `climate` library which it uses.

```python
import climate # some utilities for command line interfaces
climate.enable_default_logging()
```

We make a `Classifier` model and wrap it into an `Experiment` which allows to train the model. The architecture of the network can be specified via the `layers` parameter.

We create a neural network with three layers - input, hidden and output - each with two dimensions (2D featues, two target classes). The input and hidden layer has by default sigmoid activation, the output classification layer has softmax actiovation by default. In addition we specify a bit of L1 regularization (though in this simple example it is not necessary to prefer a sparser model).

```python
import theanets

exp = theanets.Experiment(
    theanets.Classifier,
    layers=(2, 2, 2),
    hidden_l1=0.1)
```

### Training the model

Once we created the model we can train it using the provided data. In this case we train the network via the basic stochastic gradient descent (SGD) method. We can specify its parameters quite easily. There's available a lot more optimization algorithms and their options so you can play and don't need to implement everything from scratch. Just consult the docs and possibly extent the code if needed.

```python
exp.train(
    datasets['training'],
    datasets['validation'],
    optimize='sgd',
    learning_rate=0.01,
    momentum=0.5)
```

### Evaluating the classification

Finally we look at how good the model is using functions provided by scikit-learn. For classification we can measure [precision and recall](http://en.wikipedia.org/wiki/Precision_and_recall) or combine it into a single [f1-score](http://en.wikipedia.org/wiki/F1_Score).

Also we can compute the whole [confusion matrix](http://en.wikipedia.org/wiki/Confusion_matrix) as it might give us a more complete understanding which inputs were classified correctly (counts on diagonal) and what were the errors.

```python
from sklearn.metrics import classification_report, confusion_matrix
X_test, y_test = datasets['test']
y_pred = exp.network.classify(X_test)

print('classification_report:\n', classification_report(y_test, y_pred))
print('confusion_matrix:\n', confusion_matrix(y_test, y_pred))
```

Note that we evaluate the model on against the test dataset, ie. data which was *not* used for training. Otherwise we would get too optimistic result.

In case we're happy with the model, we can use the `exp.network.classify()` function to classify the data we wish to classify (which is usually the ultimate purpose of training the model).

### Results

In case we run the whole example script...

- [theanets_hello_world.py](https://github.com/bzamecnik/ml-playground/blob/master/theanets/theanets_hello_world.py)

```bash
python ./theanets_hello_world.py
```

... we can get output like the following (training output was shortened). Plus imaging some nice colors thanks to the logger setting :)

```text
I 2015-03-02 09:03:55 theanets.layers:370 layer hid1: 2 -> 2, sigmoid, 6 parameters
I 2015-03-02 09:03:55 theanets.layers:370 layer out: 2 -> 2, softmax, 6 parameters
I 2015-03-02 09:03:55 theanets.dataset:158 valid: 4 of 4 mini-batches of (64, 2) -> (64,)
I 2015-03-02 09:03:55 theanets.dataset:158 train: 10 of 10 mini-batches of (64, 2) -> (64,)
I 2015-03-02 09:03:55 theanets.main:195 creating trainer <class 'theanets.trainer.SGD'>
I 2015-03-02 09:03:55 theanets.main:197 --batch_size = 64
I 2015-03-02 09:03:55 theanets.main:197 --cg_batches = None
I 2015-03-02 09:03:55 theanets.main:197 --contractive = 0
I 2015-03-02 09:03:55 theanets.main:197 --decode_from = 1
I 2015-03-02 09:03:55 theanets.main:197 --global_backtracking = False
I 2015-03-02 09:03:55 theanets.main:197 --gradient_clip = 1000000.0
I 2015-03-02 09:03:55 theanets.main:197 --help_activation = False
I 2015-03-02 09:03:55 theanets.main:197 --help_optimize = False
I 2015-03-02 09:03:55 theanets.main:197 --hidden_activation = logistic
I 2015-03-02 09:03:55 theanets.main:197 --hidden_dropouts = 0
I 2015-03-02 09:03:55 theanets.main:197 --hidden_l1 = 0.1
I 2015-03-02 09:03:55 theanets.main:197 --hidden_l2 = 0
I 2015-03-02 09:03:55 theanets.main:197 --hidden_noise = 0
I 2015-03-02 09:03:55 theanets.main:197 --initial_lambda = 1.0
I 2015-03-02 09:03:55 theanets.main:197 --input_dropouts = 0
I 2015-03-02 09:03:55 theanets.main:197 --input_noise = 0
I 2015-03-02 09:03:55 theanets.main:197 --layers = (2, 2, 2)
I 2015-03-02 09:03:55 theanets.main:197 --learning_rate = 0.01
I 2015-03-02 09:03:55 theanets.main:197 --max_gradient_norm = 1000000.0
I 2015-03-02 09:03:55 theanets.main:197 --min_improvement = 0.01
I 2015-03-02 09:03:55 theanets.main:197 --momentum = 0.5
I 2015-03-02 09:03:55 theanets.main:197 --optimize = ()
I 2015-03-02 09:03:55 theanets.main:197 --output_activation = linear
I 2015-03-02 09:03:55 theanets.main:197 --patience = 4
I 2015-03-02 09:03:55 theanets.main:197 --preconditioner = False
I 2015-03-02 09:03:55 theanets.main:197 --recurrent_error_start = 3
I 2015-03-02 09:03:55 theanets.main:197 --rms_halflife = 7
I 2015-03-02 09:03:55 theanets.main:197 --rprop_decrease = 0.99
I 2015-03-02 09:03:55 theanets.main:197 --rprop_increase = 1.01
I 2015-03-02 09:03:55 theanets.main:197 --rprop_max_step = 1.0
I 2015-03-02 09:03:55 theanets.main:197 --rprop_min_step = 0.0
I 2015-03-02 09:03:55 theanets.main:197 --save_every = 0
I 2015-03-02 09:03:55 theanets.main:197 --save_progress = None
I 2015-03-02 09:03:55 theanets.main:197 --tied_weights = False
I 2015-03-02 09:03:55 theanets.main:197 --train_batches = None
I 2015-03-02 09:03:55 theanets.main:197 --valid_batches = None
I 2015-03-02 09:03:55 theanets.main:197 --validate_every = 10
I 2015-03-02 09:03:55 theanets.main:197 --weight_l1 = 0
I 2015-03-02 09:03:55 theanets.main:197 --weight_l2 = 0
I 2015-03-02 09:03:55 theanets.trainer:129 compiling evaluation function
I 2015-03-02 09:03:55 theanets.trainer:296 compiling SGD learning function
I 2015-03-02 09:03:57 theanets.trainer:169 validation 0 loss=1.601823 err=1.52 hid1<0.1=0.0 hid1<0.9=100.0 out<0.1=50.0 out<0.9=50.0 acc=48.83 *
I 2015-03-02 09:03:57 theanets.trainer:169 SGD 1 loss=1.468738 err=1.39 hid1<0.1=0.0 hid1<0.9=100.0 out<0.1=50.0 out<0.9=50.0 acc=51.61
I 2015-03-02 09:03:57 theanets.trainer:169 SGD 2 loss=1.348126 err=1.27 hid1<0.1=0.0 hid1<0.9=100.0 out<0.1=50.0 out<0.9=50.0 acc=51.61
I 2015-03-02 09:03:57 theanets.trainer:169 SGD 3 loss=1.239856 err=1.16 hid1<0.1=0.0 hid1<0.9=100.0 out<0.1=22.4 out<0.9=77.6 acc=51.61
I 2015-03-02 09:03:57 theanets.trainer:169 SGD 4 loss=1.144667 err=1.07 hid1<0.1=0.0 hid1<0.9=100.0 out<0.1=0.0 out<0.9=100.0 acc=51.61
I 2015-03-02 09:03:57 theanets.trainer:169 SGD 5 loss=1.061087 err=0.98 hid1<0.1=0.0 hid1<0.9=100.0 out<0.1=0.0 out<0.9=100.0 acc=51.61
I 2015-03-02 09:03:57 theanets.trainer:169 SGD 6 loss=0.987163 err=0.91 hid1<0.1=0.0 hid1<0.9=100.0 out<0.1=0.0 out<0.9=100.0 acc=51.61
I 2015-03-02 09:03:57 theanets.trainer:169 SGD 7 loss=0.920878 err=0.84 hid1<0.1=0.0 hid1<0.9=100.0 out<0.1=0.0 out<0.9=100.0 acc=51.61
I 2015-03-02 09:03:57 theanets.trainer:169 SGD 8 loss=0.863152 err=0.78 hid1<0.1=0.0 hid1<0.9=100.0 out<0.1=0.0 out<0.9=100.0 acc=51.61
I 2015-03-02 09:03:57 theanets.trainer:169 SGD 9 loss=0.812307 err=0.73 hid1<0.1=0.0 hid1<0.9=100.0 out<0.1=0.0 out<0.9=100.0 acc=51.61
I 2015-03-02 09:03:57 theanets.trainer:169 SGD 10 loss=0.769680 err=0.68 hid1<0.1=0.0 hid1<0.9=100.0 out<0.1=0.0 out<0.9=100.0 acc=51.61
I 2015-03-02 09:03:57 theanets.trainer:169 validation 1 loss=0.765381 err=0.67 hid1<0.1=0.0 hid1<0.9=100.0 out<0.1=0.0 out<0.9=100.0 acc=48.83 *

[...]

I 2015-03-02 09:04:00 theanets.trainer:169 SGD 1189 loss=0.064307 err=0.01 hid1<0.1=74.6 hid1<0.9=76.6 out<0.1=49.5 out<0.9=50.5 acc=99.84
I 2015-03-02 09:04:00 theanets.trainer:169 SGD 1190 loss=0.064297 err=0.01 hid1<0.1=74.6 hid1<0.9=76.6 out<0.1=49.5 out<0.9=50.5 acc=99.84
I 2015-03-02 09:04:00 theanets.trainer:169 validation 119 loss=0.065886 err=0.01 hid1<0.1=73.2 hid1<0.9=75.0 out<0.1=49.4 out<0.9=50.6 acc=100.00
I 2015-03-02 09:04:00 theanets.trainer:253 patience elapsed!
```

```text
classification_report:
              precision    recall  f1-score   support

          0       1.00      1.00      1.00        97
          1       1.00      1.00      1.00       103

avg / total       1.00      1.00      1.00       200

confusion_matrix:
 [[ 97   0]
 [  0 103]]
```

As we can see, the model was trained rather quickly and was able to perfectly learn how to classify this dataset. This is due to the very simplicity of this data.

In real world the performance is usually not perfect. And the challenge it to train models that are both precise and can generalize well to unseen data.

## Conclusion and further challenges

The goal of this tutorial was to get into a running state quickly. In further articles I'd like to provide more complex examples of theanets as well as some real-world applications.

If you like this article and would like to be notified you might follow me on Twitter: [@bzamecnik](https://twitter.com/bzamecnik).

Now you can try to iteratively modify this code and experiment with progressively harder tasks. Just a few tips for inspiration:

- go through the [theanets documentation](http://theanets.readthedocs.org/en/stable/index.html) to better understand what's going on and how to use the tool
- generate data with higher input dimension (and add more input neurons)
- generate data with more classes (and add more output neurons)
- generate data that is not so easily separable
- try some real-world data
	- eg. [MNIST hand-written digit classification tutorial](http://theanets.readthedocs.org/en/stable/quickstart.html)
- [generate more complex data](http://scikit-learn.org/stable/datasets/index.html#sample-generators) (other than simple gaussian blobs)
- try various training algorithms
	- do you see any differences in training speed or classification results?
- try various architectures
	- more layers, more neurons
- try pretraining with [autoencoders](http://theanets.readthedocs.org/en/stable/generated/theanets.feedforward.Autoencoder.html#theanets.feedforward.Autoencoder)
- try other machine learning tasks, eg.
	- regression - with some regression data and [Regressor](http://theanets.readthedocs.org/en/stable/generated/theanets.feedforward.Regressor.html#theanets.feedforward.Regressor)
	- dimension reduction - with autoencoders
- try passing thenets model and training hyperparameters via command-line arguments
- try to run the code on the GPU and compare the speed vs. plain CPU
- compare the model with other models provided by scikit-learn (logistic regression, SVC, k-means, etc.) + combine it with some dimensionality reduction (eg. PCA, Kernel PCA, FastICA, etc.)
- [theano tutorial](http://deeplearning.net/software/theano/tutorial/)
- [deep learning tutorial](http://deeplearning.net/tutorial/)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html) online book

