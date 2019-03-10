import tensorflow as tf


def data_type(hyperparams):
    return tf.float16 if hyperparams.train.get_or_default(key="use_fp16", default=False) else tf.float32


def lstm_cell(num_neurons_in_layer):
    """
    Wrapper for BasicLSTMCell creator

    Args:
        num_neurons_in_layer(int): number of neurons in a single hidden layer

    Returns:
        BasicLSTMCell
    """
    return tf.nn.rnn_cell.BasicLSTMCell(
        num_units=num_neurons_in_layer,
        forget_bias=0.0,
        state_is_tuple=True,
        reuse=tf.get_variable_scope().reuse
    )


def lstm_cell_with_dropout(num_neurons_in_layer, keep_prob):
    """
    Create a rnn cell with dropout
    Args:
        num_neurons_in_layer(int): number of neurons in a single hidden layer
        keep_prob(float): dropout probability between 0-1

    Returns:
        tf.nn.rnn_cell.DropoutWrapper - the rnn cell with dropout
    """
    return tf.nn.rnn_cell.DropoutWrapper(
        cell=lstm_cell(num_neurons_in_layer),
        output_keep_prob=keep_prob
    )


def _create_multi_rnn_cell(num_neurons_in_hidden_layer,
                           num_hidden_layers,
                           keep_prob,
                           batch_size,
                           dtype,
                           mode):
    """
    Creates RNN cells and combines them to a single object.
    The individual cells will have dropout if keep_prob is less than 1 and
    mode is in training
    Args:
        num_neurons_in_hidden_layer(int): number of neurons in a single hidden layer
        num_hidden_layers(int): how many hidden layers
        keep_prob(int): for dropout, if less than 1 and mode is training,
         an lstm with dropout will be used
        batch_size(int):
        dtype(tf.type): the dtype of the zero state of the multi rnn cells
        mode(tf.estimator.ModeKeys): if in training and keep_prob is less than 1 than
         a dropout lstm cell will be used.

    Returns:
        a tuple of two:
          1. The multi rnn cell object
          2. It's initial state
    """
    if mode == tf.estimator.ModeKeys.TRAIN and keep_prob < 1:
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell_with_dropout(num_neurons_in_hidden_layer, keep_prob) for _ in range(num_hidden_layers)],
            state_is_tuple=True
        )
    else:
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell(num_neurons_in_hidden_layer) for _ in range(num_hidden_layers)],
            state_is_tuple=True
        )

    initial_state = multi_rnn_cell.zero_state(batch_size, dtype)

    tf.summary.histogram("initial_state", initial_state)

    return multi_rnn_cell, initial_state


def _create_input_test_tensors(multi_rnn_cell,
                               num_neurons_in_layer,
                               num_hidden_layers,
                               dtype):
    """
    Create the input tensors for the test phase.
    Kaldi uses these tensors, it searches it by name scope
    Do NOT change these tensors names
    Args:
        multi_rnn_cell(tf.nn.rnn_cell.MultiRNNCell): the multi rnn cell object
        num_neurons_in_layer(int): num neurons in a single layer
        num_hidden_layers(int): number of hidden layers
        dtype(tf.dtype): the type of the zero state tensor of the multi rnn cell

    Returns:
        A tuple of 2:
        1. Input tensor for the word to test
        2. Input tensor for the state (context) of that word
    """

    # create a single input tensor with shape of (1, 20)
    initial_state_single = multi_rnn_cell.zero_state(1, dtype)

    test_initial_state = tf.reshape(
        tf.stack(axis=0, values=initial_state_single),
        [num_hidden_layers, 2, 1, num_neurons_in_layer],
        name="test_initial_state"
    )

    test_word_in = tf.placeholder(
        tf.int32,
        [1, 1],
        name="test_word_in"
    )

    state_placeholder = tf.placeholder(
        tf.float32,
        [num_hidden_layers, 2, 1, num_neurons_in_layer],
        name="test_state_in"
    )

    # partition each state of each cell to a group of c states and group of h states
    partition_c_and_m = tf.unstack(state_placeholder, axis=0)

    # iterate each states for multi rnn cell
    # each iteration defines a placeholder for the c and h states
    test_input_state = tuple(
        [tf.nn.rnn_cell.LSTMStateTuple(partition_c_and_m[idx][0], partition_c_and_m[idx][1])
         for idx in range(num_hidden_layers)]
    )

    return test_word_in, test_input_state


def _create_embeddings_layer(input_tensor,
                             test_inputs,
                             vocab_size,
                             num_neurons_in_layer,
                             dtype):
    with tf.device("/cpu:0"):
        embedding = tf.get_variable(
            "embedding", [vocab_size, num_neurons_in_layer], dtype=dtype)

        input_embeddings = tf.nn.embedding_lookup(embedding,
                                                  input_tensor)
        test_embeddings = tf.nn.embedding_lookup(embedding,
                                                 test_inputs)

        return input_embeddings, test_embeddings


def _create_test_cells(multi_rnn_cell,
                       num_neurons_in_layer,
                       test_inputs,
                       test_input_state,
                       num_hidden_layers):
    with tf.variable_scope("test_cells"):
        test_cell_output, test_output_state = multi_rnn_cell(
            test_inputs[:, 0, :],
            test_input_state
        )

    test_state_out = tf.reshape(
        tf.stack(
            axis=0,
            values=test_output_state
        ),
        [num_hidden_layers, 2, 1, num_neurons_in_layer],
        name="test_state_out"
    )

    test_cell_out = tf.reshape(
        test_cell_output,
        [1, num_neurons_in_layer],
        name="test_cell_out"
    )

    # above is the first part of the graph for test
    # test-word-in
    #               > ---- > test-state-out
    # test-state-in        > test-cell-out

    # below is the 2nd part of the graph for test
    # test-word-out
    #               > prob(word | test-word-out)
    # test-cell-in

    test_word_out = tf.placeholder(tf.int32,
                                   [1, 1],
                                   name="test_word_out")
    cell_out_placeholder = tf.placeholder(tf.float32,
                                          [1, num_neurons_in_layer],
                                          name="test_cell_in")
    return test_word_out, cell_out_placeholder


def _flow_the_data_through_the_rnn_cells(data_inputs,
                                         initial_state,
                                         multi_rnn_cells,
                                         seq_len,
                                         num_neurons_in_layer):
    outputs = []
    state = initial_state
    with tf.variable_scope("rnn_cells_forward_propagation"):
        for time_step in range(seq_len):
            if time_step > -1:
                tf.get_variable_scope().reuse_variables()
            cell_output, state = multi_rnn_cells(data_inputs[:, time_step, :],
                                                 state)
            outputs.append(cell_output)

    output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, num_neurons_in_layer])

    tf.summary.histogram("lstm_output", output)
    tf.summary.histogram("final_state", state)

    return output, state


def _create_softmax(output,
                    num_neurons_in_layer,
                    vocab_size,
                    vocab_size_pos,
                    dtype,
                    cell_out_placeholder,
                    test_word_out):
    with tf.variable_scope("lstm_fast_softmax"):
        softmax_w = tf.get_variable("softmax_w",
                                    [num_neurons_in_layer, vocab_size],
                                    dtype=dtype)
        softmax_b = tf.get_variable("softmax_b",
                                    [vocab_size],
                                    dtype=dtype)
        softmax_b = softmax_b - 9.0
        tf.summary.histogram("softmax_w_lstm", softmax_w)
        tf.summary.histogram("softmax_b_lstm", softmax_b)

    with tf.variable_scope("pos_softmax"):
        softmax_w_pos = tf.get_variable("softmax_w_pos",
                                        [num_neurons_in_layer, vocab_size_pos],
                                        dtype=dtype)
        softmax_b_pos = tf.get_variable("softmax_b_pos",
                                        [vocab_size_pos],
                                        dtype=dtype)
        tf.summary.histogram("softmax_w_pos", softmax_w_pos)
        tf.summary.histogram("softmax_b_pos", softmax_b_pos)

    with tf.variable_scope("gen_softmax"):
        num_classifications = 2  # either a sentence is generated or not
        softmax_w_gen = tf.get_variable("softmax_w_gen",
                                        [num_neurons_in_layer, num_classifications],
                                        dtype=dtype)
        softmax_b_gen = tf.get_variable("softmax_b_gen",
                                        [num_classifications],
                                        dtype=dtype)
        tf.summary.histogram("softmax_w_gen", softmax_w_gen)
        tf.summary.histogram("softmax_b_gen", softmax_b_gen)

    test_logits = tf.matmul(
        cell_out_placeholder,
        tf.transpose(
            tf.nn.embedding_lookup(
                tf.transpose(softmax_w),
                test_word_out[0]
            )
        )
    ) + softmax_b[test_word_out[0, 0]]

    p_word = test_logits[0, 0]
    test_out = tf.identity(p_word, name="test_out")

    return _create_logits(output,
                          softmax_w,
                          softmax_b,
                          softmax_w_pos,
                          softmax_b_pos,
                          softmax_w_gen,
                          softmax_b_gen)


def _create_logits(output,
                   softmax_w,
                   softmax_b,
                   softmax_w_pos,
                   softmax_b_pos,
                   softmax_w_gen,
                   softmax_b_gen):
    with tf.variable_scope("lstm_fast_logits"):
        logits = tf.matmul(output, softmax_w) + softmax_b
        tf.summary.histogram("lstm_fast_logits", logits)

    with tf.variable_scope("pos_logits"):
        logits_pos = tf.matmul(output, softmax_w_pos) + softmax_b_pos
        tf.summary.histogram("pos_logits", logits_pos)

    with tf.variable_scope("gen_logits"):
        logits_gen = tf.matmul(output, softmax_w_gen) + softmax_b_gen
        tf.summary.histogram("gen_logits", logits_gen)

    return logits, logits_pos, logits_gen


def create_model(input_tensor, mode, hyperparams, shared_hyperparams):
    """
    creates the hidden layers of the model
    Args:
        input_tensor (Tensor): the input tensor, the estimator input the features
        from the input_fn
        mode (tf.estimator.ModeKeys): Can be Train, Eval or Predict, this argument can be used
          to perform different actions when doing each of the modes
        hyperparams (Dict2obj): hyperparams of a current task
        shared_hyperparams (Dict2Obj): hyperparams that tasks share

    Returns:
        dict were each key (str) is a name of the tensor and value (Tensor) is the tensor in the model
    """

    model = dict()

    initializer = tf.random_uniform_initializer(-hyperparams.train.w_init_scale,
                                                hyperparams.train.w_init_scale)
    with tf.variable_scope("lstm_fast", reuse=tf.AUTO_REUSE, initializer=initializer):
        batch_size = hyperparams.train.batch_size
        seq_len = shared_hyperparams.arch.sequence_length
        num_neurons_in_layer = shared_hyperparams.arch.hidden_layer_size
        vocab_size = hyperparams.data.vocab_size
        vocab_size_pos = hyperparams.data.vocab_size_pos
        keep_prob = shared_hyperparams.arch.keep_prob
        num_hidden_layers = shared_hyperparams.arch.num_hidden_layers
        dtype = data_type(hyperparams)

        multi_rnn_cell, initial_state = _create_multi_rnn_cell(num_neurons_in_hidden_layer=num_neurons_in_layer,
                                                               num_hidden_layers=num_hidden_layers,
                                                               keep_prob=keep_prob,
                                                               mode=mode,
                                                               batch_size=batch_size,
                                                               dtype=dtype)
        model["initial_state"] = initial_state

        test_word_in, test_input_state = _create_input_test_tensors(multi_rnn_cell=multi_rnn_cell,
                                                                    num_neurons_in_layer=num_neurons_in_layer,
                                                                    num_hidden_layers=num_hidden_layers,
                                                                    dtype=dtype)

        input_embeddings, test_embeddings = _create_embeddings_layer(input_tensor=input_tensor,
                                                                     test_inputs=test_word_in,
                                                                     vocab_size=vocab_size,
                                                                     num_neurons_in_layer=num_neurons_in_layer,
                                                                     dtype=dtype)

        test_word_out, cell_out_placeholder = _create_test_cells(multi_rnn_cell=multi_rnn_cell,
                                                                 num_neurons_in_layer=num_neurons_in_layer,
                                                                 test_inputs=test_embeddings,
                                                                 test_input_state=test_input_state,
                                                                 num_hidden_layers=num_hidden_layers)

        if mode == tf.estimator.ModeKeys.TRAIN and shared_hyperparams.arch.keep_prob < 1:
            input_embeddings = tf.nn.dropout(input_embeddings, shared_hyperparams.arch.keep_prob)

        output, final_state = _flow_the_data_through_the_rnn_cells(data_inputs=input_embeddings,
                                                                   initial_state=initial_state,
                                                                   multi_rnn_cells=multi_rnn_cell,
                                                                   seq_len=seq_len,
                                                                   num_neurons_in_layer=num_neurons_in_layer)

        logits, logits_pos, logits_gen = _create_softmax(output=output,
                                                         num_neurons_in_layer=num_neurons_in_layer,
                                                         vocab_size=vocab_size,
                                                         vocab_size_pos=vocab_size_pos,
                                                         dtype=dtype,
                                                         cell_out_placeholder=cell_out_placeholder,
                                                         test_word_out=test_word_out)

        # Save references to tensors that will be used later on
        # The following dict can be used in create_loss and estimator_hooks
        model["logits"] = logits
        model["logits_pos"] = logits_pos
        model["logits_gen"] = logits_gen

        model["final_state"] = final_state

    return model
