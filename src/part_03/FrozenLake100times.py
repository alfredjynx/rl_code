import gymnasium as gym
import numpy as np
from numpy import loadtxt
import math

env = gym.make('FrozenLake-v1', map_name='8x8',is_slippery=True, render_mode='ansi').env
q_table = loadtxt('data/q-table-frozen-lake-sarsa.csv', delimiter=',')
# q_table = loadtxt('data/q-table-frozen-lake-qlearning.csv', delimiter=',')

r = list()

minimo = math.inf
maximo = -math.inf

for _ in range(100):
    rewards = 0
    for i in range(0,100):    
        (state, _) = env.reset()
        done = False
        while not done:
            action = np.argmax(q_table[state])
            state, reward, done, _, info = env.step(action)
        rewards += reward

    if rewards>maximo:
        maximo = rewards
    elif rewards<minimo:
        minimo = rewards
    r.append(rewards)

print(np.mean(r))
print('Mínimo :', minimo)
print('Máximo :', maximo)