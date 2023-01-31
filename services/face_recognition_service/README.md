# Face Recognition Service

## Startup

```shell
python3 main.py
```

# Setup Guide

1. create new conda environment

```shell
conda create -n <ENV-NAME> python=3.7 anaconda
```

2. install `tensorflow`

```shell
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
python3 -m pip install tensorflow
```

3. install additional `keras` libraries

```shell
pip install keras_applications
pip install keras_preprocessing
```

4. edit the `vgg_face` library

```python
# edit the line in the following file:
# <ANACONDA-DIR>/envs/<ENV-NAME>/lib/python3.7/site-packages/keras_vggface/models.py

# original line:
from keras.engine.topology import get_source_inputs

# replace with:
from keras.utils.layer_utils import get_source_inputs
```

5. install `dlib`

```shell
conda install -c conda-forge dlib
```

