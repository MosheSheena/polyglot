import tensorflow as tf


def data_type(hyperparams):
    return tf.float16 if hyperparams.train.get_or_default(key="use_fp16", default=False) else tf.float32


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


def _create_multi_rnn_cell(batch_size,
                           mode,
                           hyperparams,
                           shared_hyperparams):

    if mode == tf.estimator.ModeKeys.TRAIN and shared_hyperparams.arch.keep_prob < 1:
        cell_func = attn_cell
    else:
        cell_func = lstm_cell
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(
        [cell_func(shared_hyperparams) for _ in range(shared_hyperparams.arch.num_hidden_layers)],
        state_is_tuple=True
    )

    initial_state = multi_rnn_cell.zero_state(batch_size, data_type(hyperparams))

    return multi_rnn_cell, initial_state


def _create_init_test_tensors(rnn_cell,
                              num_neurons_in_layer,
                              hyperparams,
                              shared_hyperparams):
    _initial_state_single = rnn_cell.zero_state(1, data_type(hyperparams))

    initial = tf.reshape(
        tf.stack(axis=0, values=_initial_state_single),
        [shared_hyperparams.arch.num_hidden_layers, 2, 1, num_neurons_in_layer],
        name="test_initial_state"
    )

    test_word_in = tf.placeholder(
        tf.int32,
        [1, 1],
        name="test_word_in"
    )

    state_placeholder = tf.placeholder(
        tf.float32,
        [shared_hyperparams.arch.num_hidden_layers, 2, 1, num_neurons_in_layer],
        name="test_state_in"
    )

    l = tf.unstack(state_placeholder, axis=0)
    test_input_state = tuple(
        [tf.nn.rnn_cell.LSTMStateTuple(l[idx][0], l[idx][1])
         for idx in range(shared_hyperparams.arch.num_hidden_layers)]
    )

    return test_word_in, test_input_state


def _create_embeddings_layer(input_tensor,
                             test_inputs,
                             vocab_size,
                             num_neurons_in_layer,
                             hyperparams):
    with tf.device("/cpu:0"):
        embedding = tf.get_variable(
            "embedding", [vocab_size, num_neurons_in_layer], dtype=data_type(hyperparams))

        inputs = tf.nn.embedding_lookup(embedding,
                                        input_tensor)
        test_inputs = tf.nn.embedding_lookup(embedding,
                                             test_inputs)

        return inputs, test_inputs


def _create_test_cells(cell_fn,
                       num_neurons_in_layer,
                       test_inputs,
                       test_input_state,
                       shared_hyperparams):
    with tf.variable_scope("test_cells"):
        test_cell_output, test_output_state = cell_fn(
            test_inputs[:, 0, :],
            test_input_state
        )

    test_state_out = tf.reshape(
        tf.stack(
            axis=0,
            values=test_output_state
        ),
        [shared_hyperparams.arch.num_hidden_layers, 2, 1, num_neurons_in_layer], name="test_state_out"
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


def _create_softmax(output,
                    num_neurons_in_layer,
                    vocab_size,
                    vocab_size_pos,
                    hyperparams,
                    cell_out_placeholder,
                    test_word_out):
    with tf.variable_scope("lstm_fast_softmax"):
        softmax_w = tf.get_variable("softmax_w",
                                    [num_neurons_in_layer, vocab_size],
                                    dtype=data_type(hyperparams))
        softmax_b = tf.get_variable("softmax_b",
                                    [vocab_size],
                                    dtype=data_type(hyperparams))
        softmax_b = softmax_b - 9.0

    with tf.variable_scope("pos_softmax"):
        softmax_w_pos = tf.get_variable("softmax_w_pos",
                                        [num_neurons_in_layer, vocab_size_pos],
                                        dtype=data_type(hyperparams))
        softmax_b_pos = tf.get_variable("softmax_b_pos",
                                        [vocab_size_pos],
                                        dtype=data_type(hyperparams))

    with tf.variable_scope("gen_softmax"):
        softmax_w_gen = tf.get_variable("softmax_w_gen",
                                        [num_neurons_in_layer, 2],
                                        dtype=data_type(hyperparams))
        softmax_b_gen = tf.get_variable("softmax_b_gen",
                                        [2],
                                        dtype=data_type(hyperparams))

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

    with tf.variable_scope("pos_logits"):
        logits_pos = tf.matmul(output, softmax_w_pos) + softmax_b_pos

    with tf.variable_scope("gen_logits"):
        logits_gen = tf.matmul(output, softmax_w_gen) + softmax_b_gen

    return logits, logits_pos, logits_gen


def _flow_the_data_through_the_rnn_cells(data_inputs,
                                         initial_state,
                                         cell_fn,
                                         seq_len,
                                         num_neurons_in_layer):
    outputs = []
    # TODO - check if changes original initial_state
    state = initial_state
    with tf.variable_scope("rnn_cells_forward_propagation"):
        for time_step in range(seq_len):
            if time_step > -1:
                tf.get_variable_scope().reuse_variables()
            (cell_output, state) = cell_fn(data_inputs[:, time_step, :],
                                           state)
            outputs.append(cell_output)

    output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, num_neurons_in_layer])

    return output, state


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
        vocab_size = hyperparams.problem.vocab_size
        vocab_size_pos = hyperparams.problem.vocab_size_pos

        cell, initial_state = _create_multi_rnn_cell(batch_size=batch_size,
                                                     mode=mode,
                                                     hyperparams=hyperparams,
                                                     shared_hyperparams=shared_hyperparams)
        model["initial_state"] = initial_state

        test_word_in, test_input_state = _create_init_test_tensors(rnn_cell=cell,
                                                                   num_neurons_in_layer=num_neurons_in_layer,
                                                                   hyperparams=hyperparams,
                                                                   shared_hyperparams=shared_hyperparams)

        inputs, test_inputs = _create_embeddings_layer(input_tensor=input_tensor,
                                                       test_inputs=test_word_in,
                                                       vocab_size=vocab_size,
                                                       num_neurons_in_layer=num_neurons_in_layer,
                                                       hyperparams=hyperparams)

        test_word_out, cell_out_placeholder = _create_test_cells(cell_fn=cell,
                                                                 num_neurons_in_layer=num_neurons_in_layer,
                                                                 test_inputs=test_inputs,
                                                                 test_input_state=test_input_state,
                                                                 shared_hyperparams=shared_hyperparams)

        if mode == tf.estimator.ModeKeys.TRAIN and shared_hyperparams.arch.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, shared_hyperparams.arch.keep_prob)

        output, final_state = _flow_the_data_through_the_rnn_cells(data_inputs=inputs,
                                                                   initial_state=initial_state,
                                                                   cell_fn=cell,
                                                                   seq_len=seq_len,
                                                                   num_neurons_in_layer=num_neurons_in_layer)

        logits, logits_pos, logits_gen = _create_softmax(output=output,
                                                         num_neurons_in_layer=num_neurons_in_layer,
                                                         vocab_size=vocab_size,
                                                         vocab_size_pos=vocab_size_pos,
                                                         hyperparams=hyperparams,
                                                         cell_out_placeholder=cell_out_placeholder,
                                                         test_word_out=test_word_out)

        model["logits"] = logits
        model["logits_pos"] = logits_pos
        model["logits_gen"] = logits_gen

        model["final_state"] = final_state

    return model
