import random

import gym
from pybrain.datasets import SupervisedDataSet
from pybrain.structure.modules.tanhlayer import TanhLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork

env = gym.make('CartPole-v0')
net = buildNetwork(4, 4, 1, bias=True, outputbias=False, outclass=TanhLayer)


class Trainer:
    def __init__(self):
        pass

    best_all = 0

    def run(self):
        for i_episode in range(10):
            self.run_episode()

    def run_episode(self):
        best_t, best_set = self.run_try_with_random(False)

        for i in range(100):
            t, ds = self.run_try_with_random()
            if t > best_t:
                best_t, best_set = t, ds

        trainer = BackpropTrainer(net, best_set)
        trainer.train()

    @staticmethod
    def run_try_with_random(rand=True):
        max_frames = 1000
        ds = SupervisedDataSet(4, 1)
        observation = env.reset()
        for t in range(max_frames):
            # env.render()
            # print(observation)

            action = 0 if net.activate(observation)[0] > 0 else 1

            # randomize here
            if rand and random.random() < 0.1:
                action = (action + 1) % 1

            ds.addSample(observation, (action,))
            observation, reward, done, info = env.step(action)

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

        if t == max_frames - 1:
            print("Passed!!")

        return t, ds


Trainer().run()
