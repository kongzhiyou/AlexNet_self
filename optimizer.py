from cfg import config
from data_process import ImageGenerator
import math
import tensorflow as tf

class Optimizer(object):
    def __init__(self,image_generator,model,sess):
        self.batch_size = config.BATCH_SIZE
        self.num_examples = ImageGenerator.num_examples
        self.epochs = config.EPOCH
        self.num_batch_per_epoch = math.ceil(self.num_examples/self.batch_size)
        self.num_all_batch = self.epochs*self.num_batch_per_epoch
        for step in self.num_all_batch:
           X,T_true = image_generator.next_batch()
           sess.run([model.optimize,model.loss])







