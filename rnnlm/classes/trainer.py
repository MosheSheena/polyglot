from rnnlm.utils.estimator.estimator import train_and_evaluate_model
from rnnlm.utils.estimator.estimator_hook.learning_rate_decay import LearningRateDecayHook


class Trainer:

    def __init__(self, create_model, checkpoint_path, shared_hyperparams):
        self.create_model = create_model
        self.checkpoint_path = checkpoint_path
        self.shared_hyperparams = shared_hyperparams
        self.tasks = list()

    def add_task(self, task):
        self.tasks.append(task)

    def remove_task(self, task):
        self.tasks.remove(task)

    def remove_all_tasks(self):
        self.tasks.clear()

    def train_normal(self):
        if len(self.tasks) != 1:
            raise ValueError("can only train a single task when normal training or no tasks given")
        task = self.tasks[0]
        print("training {}".format(task.name))
        train_and_evaluate_model(create_model=self.create_model,
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
                                 shared_hyperparams=self.shared_hyperparams,
                                 checkpoint_path=self.checkpoint_path,
                                 training_hooks=task.training_hooks,
                                 evaluation_hooks=task.evaluation_hooks)

    def train_multitask(self, switch_each_epoch, switch_each_batch, num_multitask_epochs):
        if not len(self.tasks) > 1:
            raise ValueError(
                "multitask learning must have more than 1 task current is {}".format(self.tasks)
            )
        if not switch_each_epoch and not switch_each_batch:
            raise ValueError("switch_each_epoch or switch_each_batch must be True")
        for multi_task_epoch in range(num_multitask_epochs):
            print("Starting multitask epoch #{}".format(multi_task_epoch + 1))
            for task in self.tasks:
                epoch_size_train = task.hyperparams.train.epoch_size_train
                num_epochs = task.hyperparams.train.num_epochs
                if switch_each_batch:
                    epoch_size_train = 1
                if switch_each_epoch:
                    num_epochs = 1
                print("training {} with epoch_size_train {} and num_epochs {}".format(task.name,
                                                                                      epoch_size_train,
                                                                                      num_epochs))
                train_and_evaluate_model(create_model=self.create_model,
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
                                         shared_hyperparams=self.shared_hyperparams,
                                         checkpoint_path=self.checkpoint_path,
                                         training_hooks=task.training_hooks,
                                         evaluation_hooks=task.evaluation_hooks)
            print("Finished multitask epoch #{}".format(multi_task_epoch + 1))

    def train_transfer_learning(self):
        if not len(self.tasks) > 1:
            raise ValueError(
                "transfer learning must have more than 1 task current is {}".format(self.tasks)
            )
        for task in self.tasks:
            print("training {}".format(task.name))
            train_and_evaluate_model(create_model=self.create_model,
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
                                     shared_hyperparams=self.shared_hyperparams,
                                     checkpoint_path=self.checkpoint_path,
                                     training_hooks=task.training_hooks,
                                     evaluation_hooks=task.evaluation_hooks)

            # if this hook is active, the learning rate decays according to num_epoch
            # if more than one task has this hook this will decay the learning rate of the other tasks
            # unless, we reset it so each task will have it's learning rate
            LearningRateDecayHook.epoch_counter = 0
