from __future__ import print_function
import random
import tensorflow as tf
from utils.config import get_config
from env.GymEnvironment import GymEnvironment
from env.SimpleGymEnvironment import SimpleGymEnvironment
from dqn.agent import Agent
flags = tf.app.flags

#model
flags.DEFINE_string('model', 'm1', 'type of model')
flags.DEFINE_boolean('dueling', False, 'whether to use dueling deep q-network')
flags.DEFINE_boolean('double_q', False, 'whether to use double q-learning')

flags.DEFINE_string('env_name', 'Breakout-v0', 'the name of gym environment to use')
flags.DEFINE_integer('action_repeat', 4, 'the number of action to be repeated')

flags.DEFINE_boolean('use_gpu', False, 'whether to use gpu or not')
flags.DEFINE_string('gpu_fraction', '1/1', 'idx / # of gpu fraction')
flags.DEFINE_boolean('display', False, 'whether to do display the game screen or not')
flags.DEFINE_boolean('is_train', True, 'whether to do training or testing')
flags.DEFINE_integer('random_seed', 123, 'value of random seed')

FLAGS = flags.FLAGS
tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

if FLAGS.gpu_fraction == '':
    raise ValueError("--gpu_fraction should be defined")

def calc_gpu_fraction(fraction_string):
    idx, num = fraction_string.split("/")
    idx, num = float(idx), float(num)
    fraction = 1 / (num - idx + 1)
    print('[*] GPU : %.4f' % fraction)
    return fraction

def main(_):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=calc_gpu_fraction(FLAGS.gpu_fraction))
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        config = get_config(FLAGS) or FLAGS

        if config.env_type == "simple":
            env = SimpleGymEnvironment(config)
        else:
            env = GymEnvironment(config)

        if not tf.test.is_gpu_available() and FLAGS.use_gpu:
            raise Exception("use_gpu flag is true when no GPUs are available")

        if not FLAGS.use_gpu:
            config.cnn_format = "NHWC"

        agent = Agent(config, env, sess)

        if FLAGS.is_train:
            agent.train()
        else:
            agent.play()

if __name__ == "__main__":
    tf.app.run()

