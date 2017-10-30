# NOvA Deep Learning Workshop Keras Tutorials

+ `mnist_quick_start.ipynb` introduces the sequential API and shows how to train a model.
+ `mnist_advanced.ipynb` introduces the functional API and builds a Siamese architecture that uses a data generator to train.


## Setup

Let's start by making a folder for this tutorial. Open the terminal, go to a fitting folder and run
```
mkdir keras-tutorials
cd keras-tutorials
```

### MacOS and Python 3

```
sudo easy_install pip
sudo pip install --upgrade virtualenv
virtualenv --system-site-packages -p python3 kerasenv
```

and

```
source ./kerasenv/bin/activate
```

then install TensorFlow

```
easy_install -U pip
pip3 install --upgrade tensorflow
```

and Keras

```
pip3 install keras
```

To check whether it worked run `python` and type in `import keras`.
