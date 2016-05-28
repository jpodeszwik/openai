import math
from random import random

import gym
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import LinearLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork


def print_net(n):
    for mod in n.modules:
        for conn in n.connections[mod]:
            print conn
            for cc in range(len(conn.params)):
                print conn.whichBuffers(cc), conn.params[cc]


max_frames = 10000


class Trainer:
    def __init__(self):
        self.best_all = 0

    def run(self):
        for generation in range(100):
            print("Generation " + str(generation))
            self.run_generation()

    def run_generation(self):
        best_t, best_set = self.run_try()

        for i in range(20):
            t, ds = self.run_try(rand_count=2, rand_count_ref=best_t)
            if t > best_t:
                best_t, best_set = t, ds

        self.run_try(render=True)

        if best_t > self.best_all:
            self.best_all = best_t

        # trainer = BackpropTrainer(net, best_set, lrdecay=0.9, momentum=0.3, weightdecay=0.01, learningrate=0.01)
        trainer = BackpropTrainer(net, best_set, learningrate=0.1)
        trainer.trainUntilConvergence(maxEpochs=20, validationProportion=0.1)
        # trainer.trainUntilConvergence(maxEpochs=20, continueEpochs=5, validationProportion=0.5)
        trainer.train()

        print_net(net)

    def run_try(self, rand_chance=0, rand_count=0, rand_count_ref=0, render=False):
        ds = SupervisedDataSet(env_size, 1)
        observation = env.reset()

        random_indexes = []

        while len(random_indexes) < rand_count:
            random_index = math.floor(random() * rand_count_ref)
            if random_index not in random_indexes:
                random_indexes.append(random_index)

        for t in range(max_frames):
            if render:
                env.render()
            # print(observation)

            action = 0 if net.activate(observation)[0] < 0 else 1

            if t in random_indexes or random() < rand_chance:
                action = (action + 1) % 1

            ds.addSample(observation, (action,))
            observation, reward, done, info = env.step(action)

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

        if t == max_frames - 1:
            print("Passed!!")
            self.run_try(render=True)

        return t, ds


env = gym.make('CartPole-v0')
env_size = 4
net = buildNetwork(env_size, 1, bias=True, outputbias=False, outclass=LinearLayer)

Trainer().run()
