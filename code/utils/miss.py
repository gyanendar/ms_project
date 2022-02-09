import random
import numpy as np
import tensorflow as tf


def set_seed():
  tf.random.set_seed(207241)
  np.random.seed(207241)
  random.seed(207241)