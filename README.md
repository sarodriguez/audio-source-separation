This repo contains a re-implementation from scratch of some of the existing methods for music source separation. We implement [Conditioned-U-Net](https://arxiv.org/abs/1907.01277) and [LASAFT](https://arxiv.org/abs/2010.11631v2).
In this project we used WAV version of the [musdb dataset](https://sigsep.github.io/datasets/musdb.html). We converted it using musdb's [musdbconvert](https://pypi.org/project/musdb/).

There are two main scripts to execute in order to be able to interact with the project, the train.py script contains all the training logic, while the evaluate.py is only for doing inference using an existing model. Using [effortless-config](https://pypi.org/project/effortless-config/) we define all the arguments in the config.py file. These should be modified to match your project structure and desired behavior.

### Models
Contains the implemented models and its mechanisms

### Dataset
Contains the musdb dataset, which is defined as a PyTorch dataset.

### Utils
Contains additional utilities used during training, such as predefined PyTorch object getters and a 'spectrogramer' used during training an evaluation as an intermediate step between the raw audio inut and the models.
