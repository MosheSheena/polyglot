import tensorflow as tf


def data_type(hyperparams):
    return tf.float16 if hyperparams.train.use_fp16 else tf.float32


def lstm_cell(hyperparams):
    """
    Wrapper for BasicLSTMCell creator

    Args:
        hyperparams: (Dict2obj)

    Returns:
        BasicLSTMCell
    """
    return tf.nn.rnn_cell.BasicLSTMCell(
        num_units=hyperparams.arch.hidden_layer_size,
        forget_bias=0.0,
        state_is_tuple=True,
        reuse=tf.get_variable_scope().reuse
    )


def attn_cell(hyperparams):
    return tf.nn.rnn_cell.DropoutWrapper(
        cell=lstm_cell(hyperparams),
        output_keep_prob=hyperparams.arch.keep_prob
    )


def create_model(input_tensor, mode, hyperparams, is_training):
    """

    Args:
        input_tensor: (Tensor)
        mode: (tf.estimator.ModeKeys) Can be Train, Eval or Predict
        hyperparams: (Dict2obj)
        is_training: (bool) TODO - remove dependency

    Returns:
        dict were each key (str) is a name of the tensor and value (Tensor) is the tensor in the model
    """

    model = dict()
    with tf.variable_scope("lstm_fast", reuse=tf.AUTO_REUSE) as scope:
        # if is_training:
        #    scope.reuse_variables()

        batch_size = hyperparams.train.batch_size
        num_steps = hyperparams.arch.hidden_layer_depth
        size = hyperparams.arch.hidden_layer_size
        vocab_size = hyperparams.problem.vocab_size

        if is_training and hyperparams.arch.keep_prob < 1:
            cell_func = attn_cell
        else:
            cell_func = lstm_cell
        cell = tf.nn.rnn_cell.MultiRNNCell(
            [cell_func(hyperparams) for _ in range(hyperparams.arch.num_hidden_layers)],
            state_is_tuple=True
        )

        _initial_state = cell.zero_state(batch_size, data_type(hyperparams))
        model["initial_state"] = _initial_state
        _initial_state_single = cell.zero_state(1, data_type(hyperparams))

        initial = tf.reshape(
            tf.stack(axis=0, values=_initial_state_single),
            [hyperparams.arch.num_hidden_layers, 2, 1, size],
            name="test_initial_state"
        )

        # first implement the less efficient version
        test_word_in = tf.placeholder(
            tf.int32,
            [1, 1],
            name="test_word_in"
        )

        state_placeholder = tf.placeholder(
            tf.float32,
            [hyperparams.arch.num_hidden_layers, 2, 1, size],
            name="test_state_in"
        )

        l = tf.unstack(state_placeholder, axis=0)
        test_input_state = tuple(
            [tf.nn.rnn_cell.LSTMStateTuple(l[idx][0], l[idx][1])
                for idx in range(hyperparams.arch.num_hidden_layers)]
        )

        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [vocab_size, size], dtype=data_type(hyperparams))

            inputs = tf.nn.embedding_lookup(embedding,
                                            input_tensor)
            test_inputs = tf.nn.embedding_lookup(embedding,
                                                 test_word_in)

        # test time
        with tf.variable_scope("RNN"):
            (test_cell_output, test_output_state) = cell(
                test_inputs[:, 0, :],
                test_input_state
            )

        test_state_out = tf.reshape(
            tf.stack(
                axis=0,
                values=test_output_state
            ),
            [hyperparams.arch.num_hidden_layers, 2, 1, size], name="test_state_out"
        )

        test_cell_out = tf.reshape(
            test_cell_output,
            [1, size],
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
        cellout_placeholder = tf.placeholder(tf.float32,
                                             [1, size],
                                             name="test_cell_in")

        softmax_w = tf.get_variable("softmax_w",
                                    [size, vocab_size],
                                    dtype=data_type(hyperparams))
        softmax_b = tf.get_variable("softmax_b",
                                    [vocab_size],
                                    dtype=data_type(hyperparams))
        softmax_b = softmax_b - 9.0

        test_logits = tf.matmul(
            cellout_placeholder,
            tf.transpose(
                tf.nn.embedding_lookup(
                    tf.transpose(softmax_w),
                    test_word_out[0]
                )
            )
        ) + softmax_b[test_word_out[0, 0]]

        p_word = test_logits[0, 0]
        test_out = tf.identity(p_word, name="test_out")

        if is_training and hyperparams.arch.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, hyperparams.arch.keep_prob)

        # Simplified version of models/tutorials/rnn/rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # inputs = tf.unstack(inputs, num=num_steps, axis=1)
        # outputs, state = tf.contrib.rnn.static_rnn(
        #     cell, inputs, initial_state=self._initial_state)
        outputs = []
        state = _initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > -1:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :],
                                            state)
                outputs.append(cell_output)

        output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, size])
        logits = tf.matmul(output, softmax_w) + softmax_b
        _final_state = state
        model["final_state"] = _final_state
        model["logits"] = logits
    return model
