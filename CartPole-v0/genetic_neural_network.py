import gym
import time
import random
import copy

class GeneticNetworkHelper:
    def crossing(net1, net2):
        crossing_point = random.randint(1, 3)
        new_weights = []
        
        for i in range(crossing_point):
            new_weights.append(net1.weights[i])

        for i in range(crossing_point, 4):
            new_weights.append(net2.weights[i])

        return Network(new_weights)

    def mutate(net):
        mutations = random.randint(1, 3)
        mutated_genes = random.sample([0, 1, 2, 3], mutations)
        new_weights = copy.copy(net.weights)
        
        for idx in mutated_genes:
            new_weights[idx] = random.random()*2 - 1

        return Network(new_weights)

class GeneticSearcher:
    def __init__(self, pop_size):
        self.pop = [Network.random_network() for i in range(pop_size)]
        self.nt = NetTester()

    def rate_network(self, net):
        return self.nt.test_n_times_and_return_min(net, 3)

    def selection(self):
        population_fitness = [(net, self.rate_network(net)) for net in self.pop]
        population_fitness = sorted(population_fitness, reverse=True, key=lambda x: x[1])
        pop_size = len(population_fitness)
        old_survivors = list(map(lambda x: x[0], population_fitness[:int(pop_size/3)]))
        children = []
        while len(children) < pop_size/3:
            parents = random.sample(set(old_survivors), 2)
            children.append(GeneticNetworkHelper.crossing(parents[0], parents[1]))
        
        new_generation = old_survivors + children

        while len(new_generation) < pop_size:
            new_generation.append(GeneticNetworkHelper.mutate(random.choice(old_survivors)))

        self.pop = new_generation

        return population_fitness[0][1]

    def show_best(self):
        population_fitness = [(net, self.rate_network(net)) for net in self.pop]
        population_fitness = sorted(population_fitness, reverse=True, key=lambda x: x[1])
        best = population_fitness[0][0]
        self.nt.render(best)

class Network:
    def __init__(self, weights):
        self.weights = weights

    def weighted_sum(self, observation):
        sum = 0.0
        for i in range(4):
            sum += self.weights[i] * observation[i]

        return sum

    def output(self, observation):
        if self.weighted_sum(observation) > 0:
            return 1
        else:
            return 0

    def __str__(self):
        return str(self.weights)

    def random_network():
        return Network([random.random() * 2 - 1.0 for i in range(4)])

class NetTester:
    def __init__(self):
        self.env = gym.make('CartPole-v0')

    def test_n_times_and_return_min(self, net, n):
        results = [self.test(net) for i in range(n)]
        return min(results)


    def test(self, net):
        observation = self.env.reset()
        action = 0

        for t in range(10000):
            observation, reward, done, info = self.env.step(action)
            action = net.output(observation)

            if done:
                break

        return t+1

    def test_with_render(self, net):
        observation = self.env.reset()
        action = 0

        for t in range(100000):
            self.env.render()
            observation, reward, done, info = self.env.step(action)
            action = net.output(observation)

            if done:
                break

        return t+1

    def render(self, net):
        val = self.test_with_render(net)
        print ('result: {}', val)
        
gs = GeneticSearcher(20)

for i in range(20):
    print('generation {}'.format(i))
    best = gs.selection()
    print('best: {}'.format(best))

    gs.show_best()

    if best == 10000:
        break
