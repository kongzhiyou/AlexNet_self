from model import AlexModel
import tensorflow as tf
from cfg import config
from optimizer import Optimizer
from data_process import ImageGenerator

X = tf.placeholder(dtype=tf.float32,shape=[config.BATCH_SIZE,227,227])
Y = tf.placeholder(dtype=tf.float32,shape=[None,config.NUM_CLASS])

sess = tf.Session()
model = AlexModel(X,config.KEEP_PROP,config.NUM_CLASS)
fc8 = model.create()
loss = model.loss(fc8,Y)
optimizier = model.optimize(loss)
correct_pred = tf.equal(tf.argmax(fc8,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

image_generator = ImageGenerator()
all_step = image_generator.all_step

init_op = tf.global_variables_initializer()
sess.run(init_op)
for i in range(all_step):
    train_set , label_set = image_generator.next_batch()
    loss_out, _ , acc_out = sess.run([loss,optimizier,accuracy],feed_dict={X:train_set,Y:label_set})





