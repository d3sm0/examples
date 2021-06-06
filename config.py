import argparse
import math
import os

import jax.experimental.optimizers

import experiment_buddy

initial_lr = .0001

decay_steps = 500000
num_hidden = 1024
decay_factor = .5

batch_size = 128
momentum_mass = 0.99
weight_norm = 0.00

num_epochs = 1

experiment_buddy.register(locals())

################################################################
# Derivative parameters
################################################################
learning_rate = jax.experimental.optimizers.inverse_time_decay(initial_lr, decay_steps, decay_factor, staircase=True)
eval_every = math.ceil(num_epochs / 1000)

HOST = os.environ['DEPLOY_DESTINATION']

host_map = {
    'cluster': 'mila',
    'local': ''
}

tensorboard = experiment_buddy.deploy(host=host_map[HOST], sweep_yaml="")
