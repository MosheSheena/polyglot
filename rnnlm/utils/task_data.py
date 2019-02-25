class TaskData:

    def __init__(self,
                 task,
                 train_tf_record_path,
                 valid_tf_record_path,
                 test_tf_record_path):
        self.task = task
        self.train_tf_record_path = train_tf_record_path
        self.valid_tf_record_path = valid_tf_record_path
        self.test_tf_record_path = test_tf_record_path
