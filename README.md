# Installation

On Debian:

```sh
# scikit-learn
$ sudo apt-get install python3-sklearn

# keras
$ sudo apt-get install python3-pkgconfig python3-h5py python3-pip
$ pip3 install pip --user
$ pip3 install keras --user
$ pip3 install tensorflow --user
```

The reason for installing `pip` through `pip` is because the Debian
package applies patches that breaks `pip` in several ways.  One of these
patches causes `pip` to always use `ignore-installed` -- which results
in the Debian-packaged pip to ignore dependencies that's already been
installed (and `keras` has some heavy dependencies, eg. `numpy`).

See: https://github.com/pypa/pip/issues/4222


# MNIST ConvNet

```sh
./app.py --mnist-conv-net
[...]
ConvNet (44.24 seconds):
acc: 0.8628
loss: 0.03871329088807106
```


# Spiral

```sh
$ ./app.py --spiral-values values.png --spiral-metrics metrics.png
```

![spiral values](/data/img/spiral-values.png)
![spiral metrics](/data/img/spiral-metrics.png)
