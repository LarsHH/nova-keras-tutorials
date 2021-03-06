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

Finally clone this repository by running `git clone https://github.com/LarsHH/nova-keras-tutorials.git`.

In order to run the notebooks you will also need to install Jupyter and Matplotlib:

```
pip3 install jupyter
pip3 install matplotlib
```

## Running the Notebooks
In order to go through the notebooks call `jupyter notebook` from the folder that you cloned the repository into. This will open the Jupyter Notebook portal in your browser. From the Jupyter Notebook portal you can open the individual notebooks and run each piece of code using CTRL+Return.
