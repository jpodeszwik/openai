import copy
import gym
import numpy
import random


class GeneticNetworkHelper:
    @staticmethod
    def crossing(net1, net2):
        crossing_point = random.randint(1, 3)
        new_weights = []

        for i in range(crossing_point):
            new_weights.append(net1.weights[i])

        for i in range(crossing_point, 4):
            new_weights.append(net2.weights[i])

        return Network(new_weights)

    @staticmethod
    def mutate(net):
        mutations = random.randint(1, 3)
        mutated_genes = random.sample([0, 1, 2, 3], mutations)
        new_weights = copy.copy(net.weights)

        for idx in mutated_genes:
            new_weights[idx] = random.random() * 2 - 1

        return Network(new_weights)


class GeneticSearcher:
    def __init__(self, pop_size):
        self.pop = [Network.random_network() for i in range(pop_size)]
        self.fitness_cache = {}
        self.nt = NetTester()
        self.best = None

    def rate_network(self, net):
        return self.nt.test_n_times_and_return_min(net, 10)

    def fitness(self, net):
        if net not in self.fitness_cache:
            self.fitness_cache[net] = self.rate_network(net)

        return self.fitness_cache[net]

    def selection(self):
        population_fitness = [(net, self.fitness(net)) for net in self.pop]
        population_fitness = sorted(population_fitness, reverse=True, key=lambda x: x[1])
        pop_size = len(population_fitness)
        old_survivors = list(map(lambda x: x[0], population_fitness[:int(pop_size / 3)]))
        children = []
        while len(children) < pop_size / 3:
            parents = random.sample(set(old_survivors), 2)
            children.append(GeneticNetworkHelper.crossing(parents[0], parents[1]))

        new_generation = old_survivors + children

        while len(new_generation) < 0.9 * pop_size:
            new_generation.append(GeneticNetworkHelper.mutate(random.choice(old_survivors)))

        while len(new_generation) < pop_size:
            new_generation.append(Network.random_network())

        self.pop = new_generation

        self.best = population_fitness[0][0]
        return population_fitness[0][1]

    def show_best(self):
        self.nt.render('{}'.format(self.best))


class Network:
    def __init__(self, weights):
        self.weights = weights

    def __hash__(self):
        return hash(frozenset(self.weights))

    def __eq__(self, other):
        return self.weights.__eq__(other.weights)

    def weighted_sum(self, observation):
        s = 0.0
        for i in range(3):
            s += self.weights[i] * observation[i]

        return s + self.weights[3]

    def output(self, observation):
        val = self.weighted_sum(observation) / 2
        if val > 2:
            return 2
        elif val < -2:
            return -2

        return val

    def __str__(self):
        return str(self.weights)

    @staticmethod
    def random_network():
        return Network([random.random() * 2 - 1 for i in range(4)])


class NetTester:
    def __init__(self):
        self.env = gym.make('Pendulum-v0')

    def test_n_times_and_return_min(self, net, n):
        results = [self.test(net) for i in range(n)]
        return min(results)

    def test(self, net):
        observation = self.env.reset()
        numpy.ndarray(shape=(1, 1), dtype=float)
        action = numpy.array([0])

        res = 0.0

        for t in range(1000):
            observation, reward, done, info = self.env.step(action)

            for i in range(3):
                observation[i] /= self.env.observation_space.high[i]
            numpy.ndarray(shape=(1, 1), dtype=float)
            action = numpy.array([net.output(observation)])

            res += reward

            if done:
                break

        return reward

    def test_with_render(self, net):
        observation = self.env.reset()
        numpy.ndarray(shape=(1, 1), dtype=float)
        action = numpy.array([0])

        for t in range(1000):
            self.env.render()
            observation, reward, done, info = self.env.step(action)
            numpy.ndarray(shape=(1, 1), dtype=float)
            action = numpy.array([net.output(observation)])

            if done:
                break

        return t + 1

    def render(self, net):
        val = self.test_with_render(net)
        print('result: {}', val)


def main():
    gs = GeneticSearcher(100)

    for i in range(20):
        print('generation {}'.format(i))
        best = gs.selection()
        print('best: {}'.format(best))

        gs.show_best()

        if best == 10000:
            break


main()
