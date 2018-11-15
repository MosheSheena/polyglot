from rnnlm.utils.estimator.estimator import train_and_evaluate_model


def train_classic(task):
    print("training {}".format(task.name))
    train_and_evaluate_model(create_model=task.create_model,
                             create_loss=task.create_loss,
                             create_optimizer=task.create_optimizer,
                             train_tf_record_path=task.train_tf_record_path,
                             valid_tf_record_path=task.valid_tf_record_path,
                             test_tf_record_path=task.test_tf_record_path,
                             num_epochs=task.hyperparams.train.num_epochs,
                             epoch_size_train=task.hyperparams.train.epoch_size_train,
                             epoch_size_valid=task.hyperparams.train.epoch_size_valid,
                             epoch_size_test=task.hyperparams.train.epoch_size_test,
                             hyperparams=task.hyperparams,
                             checkpoint_path=task.checkpoint_path,
                             training_hooks=task.training_hooks,
                             evaluation_hooks=task.evaluation_hooks)


def train_multitask_learning(tasks, switch_each_epoch, switch_each_batch, num_multitask_epochs):
    _verify_weight_sharing(tasks=tasks)
    for multi_task_epoch in range(num_multitask_epochs):
        print("Starting multitask epoch #{}".format(multi_task_epoch))
        for task in tasks:
            epoch_size_train = task.hyperparams.train.epoch_size_train
            num_epochs = task.hyperparams.train.num_epochs
            if switch_each_batch:
                epoch_size_train = 1
            if switch_each_epoch:
                num_epochs = 1
            print("training {} with epoch_size_train {} and num_epochs {}".format(task.name,
                                                                                  epoch_size_train,
                                                                                  num_epochs))
            train_and_evaluate_model(create_model=task.create_model,
                                     create_loss=task.create_loss,
                                     create_optimizer=task.create_optimizer,
                                     train_tf_record_path=task.train_tf_record_path,
                                     valid_tf_record_path=task.valid_tf_record_path,
                                     test_tf_record_path=task.test_tf_record_path,
                                     num_epochs=num_epochs,
                                     epoch_size_train=epoch_size_train,
                                     epoch_size_valid=task.hyperparams.train.epoch_size_valid,
                                     epoch_size_test=task.hyperparams.train.epoch_size_test,
                                     hyperparams=task.hyperparams,
                                     checkpoint_path=task.checkpoint_path,
                                     training_hooks=task.training_hooks,
                                     evaluation_hooks=task.evaluation_hooks)
        print("Finished multitask epoch #{}".format(multi_task_epoch))


def train_transfer_learning(tasks):
    _verify_weight_sharing(tasks=tasks)
    for task in tasks:
        print("training {}".format(task.name))
        train_and_evaluate_model(create_model=task.create_model,
                                 create_loss=task.create_loss,
                                 create_optimizer=task.create_optimizer,
                                 train_tf_record_path=task.train_tf_record_path,
                                 valid_tf_record_path=task.valid_tf_record_path,
                                 test_tf_record_path=task.test_tf_record_path,
                                 num_epochs=task.hyperparams.train.num_epochs,
                                 epoch_size_train=task.hyperparams.train.epoch_size_train,
                                 epoch_size_valid=task.hyperparams.train.epoch_size_valid,
                                 epoch_size_test=task.hyperparams.train.epoch_size_test,
                                 hyperparams=task.hyperparams,
                                 checkpoint_path=task.checkpoint_path,
                                 training_hooks=task.training_hooks,
                                 evaluation_hooks=task.evaluation_hooks)


def _verify_weight_sharing(tasks):
    # TODO write test
    # Check if tasks checkpoint path is the same otherwise weight sharing will not work
    previous_checkpoint_path = tasks[0].checkpoint_path
    for task in tasks[1:]:
        if previous_checkpoint_path != task.checkpoint_path:
            raise ValueError("checkpoints path must be the same for all tasks in order to share weights")
