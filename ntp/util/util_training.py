"""Helpers for training"""

import tensorflow as tf

class TrainingState:
    """Keeps track of state of training, used to implement learning rate decay"""

    def __init__(self, initial_lr, decay_type, decay_rate):

        self.decay_type_dict = {
            "exp": self.exponential_decay
        }

        self.decay_type = decay_type
        self.decay_rate = decay_rate
        self.iterations = 0
        self.lr_variable = tf.Variable(initial_lr)
        
    def update_iteration(self):
        self.iterations +=1
        if self.decay_type is not None:
            self.decay_type_dict[self.decay_type](self.iterations)

    def exponential_decay(self, iterations):
        new_value = self.lr_variable * tf.exp(-self.decay_rate)
        self.lr_variable.assign(new_value)

    





        