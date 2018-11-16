"""
A config file containing definitions required for generating words

"""
# Tensors required to load from model
TENSORS_OF_MODEL_DICT = {
    "word_in": "Test/Model/lstm_fast/test_word_in",
    "word_out": "Test/Model/lstm_fast/test_word_out",
    "initial_state": "Test/Model/lstm_fast/test_initial_state",
    "state_in": "Test/Model/lstm_fast/test_state_in",
    "state_out": "Test/Model/lstm_fast/test_state_out",
    "cell_in": "Test/Model/lstm_fast/test_cell_in",
    "cell_out": "Test/Model/lstm_fast/test_cell_out",
    "test_out": "Test/Model/lstm_fast/test_out",
    "softmax_w": "Model/lstm_fast/softmax_w",
    "softmax_b": "Model/lstm_fast/softmax_b"
}

GEN_DICT = {
    "weights": "Model/lstm_fast/softmax_w",
    "biases": "Model/lstm_fast/softmax_b",
    "initial_state": "Test/Model/lstm_fast/test_initial_state",
    "state_in": "Test/Model/lstm_fast/test_state_in"
}
# changes for commit
