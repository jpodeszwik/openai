from random import random

import gym
from pybrain import LinearLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork


def print_net(n):
    for mod in n.modules:
        for conn in n.connections[mod]:
            print conn
            for cc in range(len(conn.params)):
                print conn.whichBuffers(cc), conn.params[cc]


class Trainer:
    def __init__(self):
        self.best_all = 0

    def run(self):
        for i_episode in range(100):
            self.run_episode()

    def run_episode(self):
        best_t, best_set = self.run_try_with_random(rand_chance=0)

        for i in range(20):
            t, ds = self.run_try_with_random(1.0 / best_t)
            if t > best_t:
                best_t, best_set = t, ds

        self.run_try_with_random(0, True)

        if best_t > self.best_all:
            self.best_all = best_t

        # trainer = BackpropTrainer(net, best_set, lrdecay=0.9, momentum=0.3, weightdecay=0.01, learningrate=0.01)
        trainer = BackpropTrainer(net, best_set, learningrate=0.1)
        trainer.trainUntilConvergence(maxEpochs=20, validationProportion=0.1)
        # trainer.trainUntilConvergence(maxEpochs=20, continueEpochs=5, validationProportion=0.5)
        trainer.train()

        print_net(net)

    @staticmethod
    def run_try_with_random(rand_chance=0.01, render=False):
        max_frames = 100000
        ds = SupervisedDataSet(env_size, 1)
        observation = env.reset()
        for t in range(max_frames):
            if render:
                env.render()
            # print(observation)

            action = 0 if net.activate(observation)[0] < 0 else 1

            if random() < rand_chance:
                action = (action + 1) % 1

            ds.addSample(observation, (action,))
            observation, reward, done, info = env.step(action)

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

        if t == max_frames - 1:
            print("Passed!!")

        return t, ds


env = gym.make('CartPole-v0')
env_size = 4
net = buildNetwork(env_size, 1, bias=True, outputbias=False, outclass=LinearLayer)

Trainer().run()
