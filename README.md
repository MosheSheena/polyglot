# Bsc-Final-Project
Bsc SW Eng. degree final project - improving performance for an RNN based LSTM language model using TensorFlow for Python

## Installation
### Using Docker
1. Install Docker https://docs.docker.com/install/

### Manual
1. Install TensorFlow version: 1.12 - https://www.tensorflow.org/install
2. Install Python dependancies:
```shell
pip install -r requirements.txt
```

## How to Run
### Using Docker
1. Build the image (notice the dot at the end): 
```shell
docker build -t lstm_fast .
```
2. Run the image: 
```shell
docker run lstm_fast
```

### Manual
Make sure your current working directory is the project's root folder
```shell
python main.py
```

## Limitations
We assume of the following:
1. Your vocab file fits in memory (train, test and validation datasets are unlimited)
2. We need the num batches of each {train, test, validation} to be inserted into hyperparameters.json
this is due to the fact that tf.data.Dataset loads in mini batches your data without taking into how much data there is.

