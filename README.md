# Bsc-Final-Project
Bsc SW Eng. degree final project - improving performance for an RNN based LSTM language model using TensorFlow for Python

## Dependencies
TensorFlow version: 1.12
Install the following:
```shell
pip install -r requirements.txt
```

Open Python console and run:
```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

## How to Run
Make sure your current working directory is the project's root folder
```shell
python main.py
```

## Assumptions
we assume of the following:
1. Your vocab file fits in memory (train, test and validation datasets are unlimited)
2. We need the num batches of each {train, test, validation} to be inserted into hyperparameters.json
this is due to the fact that tf.data.Dataset loads in mini batches your data without taking into how much data there is.


## Configuring the JSON Schema

### Using PyCharm

1. Open the experiment_config.json in Pycharm

2. You should see something like this:

3. Click on the "No Json Schema" button (marked red)

4. Choose "New Schema Mapping":

5. Give the mapping any name you wish

6. Choose the file schema - schema.json:

7. Choose Schema version - "JSON Schema Version 7"

8. Your final settings should look this:


### Using The Web

