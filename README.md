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

2. Click on the "No Json Schema" button (marked red):
![Imgur](https://i.imgur.com/LGnibsl.png)

3. Choose "New Schema Mapping":
![Imgur](https://i.imgur.com/aTavzTS.png)

4. Give the mapping any name you wish

5. Choose the file schema - schema.json:
![Imgur](https://i.imgur.com/0QpjEno.png)

6. Choose Schema version - "JSON Schema Version 7"

7. Your final settings should look this:
![Imgur](https://i.imgur.com/4xygoCl.png)

### Using The Web
1. Go to https://www.jsonschemavalidator.net/
2. Copy the content of schema.json to the left side of the web page
3. Copy the content of experiments_config.json to the right side of the web page
4. If your JSON is valid you should see the green message:
![Imgur](https://i.imgur.com/EXc3bdM.png)

5. If your json is invalid you should see the red message indication the error:
![Imgur](https://i.imgur.com/lq4Jbi8.png)
