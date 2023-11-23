# Hyperparameter optimisation of different reinforcement learning algorithms using genetic algorithm in Webots environment

Requirements:
- gym
- numpy
- stable_baselines3
- sb3_contrib
- pygad
- pandas

The project was used with Python3.9 and Webots2023a.

# Install

`pip3 install numpy gym stable_baselines3 sb3_contrib pygad pandas`

If you want to use CUDA, then proceed with [this](https://pytorch.org/get-started/locally/).

# Usage
The framework uses stable baselines RL algorithms and pygad genetic algorithm for parameter optimisation in a Webots based environment. 
In the 'cartpole_openai' directory an example can be found, which contains a cartpole based sample Webots world (based on the openai Webots example [world](https://www.cyberbotics.com/doc/guide/samples-howto?version=master#openai_gym-wbt).

In the 'pioneer_follow' directory a custom Webots environment can be found, which contains a Pioneer 3-DX mobile robot. The goal is to follow the white cube using the following observations:
- wheel velocity
- distance from object
- relative orientation between the robot and the object

The action space can be discrete or continuous. If discrete the action is represented by 4 binary values, which are stop, move forward, turn left, turn right. If the action space is continuous, then two floats are used, which control the motors directly with values in the range of [0.0, 12.0].

The reward is shaped as follows:
- $r = r_{\phi} + r_{d}$
- if $|\phi_{rel}| <= \beta $: $r_{\phi} = \frac{\beta - |\phi_{rel}|}{2*\beta}$, else $r_{\phi} = -1 * \frac{|\phi_{rel}|}{\pi}$
- if $v_{wheels} == 0$ and $d_{goal} > 1$: $r_{d} = -1$, else if $v_{wheels} == 0$ and $d_{goal} < 1$: $r_{d} = 1$,else if $d_{goal} < 1$: $r_{d} = 0.5$, else if $v_{wheels} > 0$ and $d_{goal} > d_{init}$: $r_{d} = -0.5$

So reward is given if the relative orientation is lower or equal than 20 degrees, or the velocity is zero, but the goal is reached (distance < 1 meters), or half reward is given, when the goal is reached but the velocity is not zero. On the other hand, punishment is given when relative orientation is bigger than 20 deegrees, or the velocity is zero when the goal is not reached yet, or the robot is moving away from the goal.

The done flag is true, when the $d_{goal} >=1$ and $\bar{v} < 0.1$ or the simulation has reached a pre-set value, in this case 500 timesteps. 

Can be run after installing the dependencies and opening the Webots world. By default it will optimise the algorithms given in the `alg_list` list with 2048 steps for one simulation, the genetic algorith runs with the following parameters: 5 generation, 25 solution per population, two parents and random mutation. The gene type and the gene space is given by the params dictionary, which contains the parameters for the used RL algorithms. The optimisation is logged to the `logs\algorithm\ga_log.log` file. Also the RL training is logged to the `logs\algorithm\progress.csv` file. The final model is saved to the `models\algorithm\algorithm.zip` file.   

Currently the framework supports the following RL algorithms (from stablebaselines3 and stablebaselines3_contrib) [^1]:
- TRPO
- PPO
- DQN
- DDPG
- SAC
- TD3

One can extend this list with changing the `params` dictionary in the utils.py file and the corresponding part in the `run_env` function.  E.g. the parameters for the TRPO algoritm is the following:

```
'TRPO':
{
        'parameters': ['learning_rate', 'batch_size', 'gamma', 'n_critic_updates', 'gae_lambda', 'target_kl'],
        'ranges': [{'low': 0.0001, 'high': 0.1},
                   [64, 128, 256, 512],
                   {'low': 0.0, 'high': 1.0},
                   np.arange(10, 110, 10).tolist(),
                   {'low': 0.0, 'high': 1.0},
                   {'low': 0.0, 'high': 1.0}],
        'types': [float, int, float, int, float, float],
        "discrete": True, "total_timesteps": 10_000
 }
 ```
 In which:
 - parameters - Contains the name of the parameters
 - ranges - Contains the range for the corresponding parameter in a list, for a given parameter a list or a low-high dict can be used
 - types - Contains the type for the corresponding parameter in a list
 - discrete - Action space type of the algorithm, if uses gym.space.Discrete then it should be True
 - total timesteps - Maximum steps for one episode, the bigger it is the longer the optimisation will become
 
 PyGAD package is used for the genetic algorithm based parameter optimisation [^2]:
 
 ```
 ga_instance = pygad.GA(num_generations=5,
                       num_parents_mating=2,
                       sol_per_pop=25,
                       num_genes=len(gene_space),
                       fitness_func=fitness_function,
                       mutation_type="random",
                       gene_type=gene_types,
                       gene_space=gene_space,
                       on_generation=log_generation)
```

If you use this project, please cite: L. Schaffer, Z. Kincses, Sz. Pletl, "Hyperparameter Optimisation of Reinforcement Learning Algorithms in Webots Simulation Environment", IEEE 23rd International Symposium on Computational Intelligence and Informatics (CINTI 2023), 2023
                                
[^1]: Antonin Raffin et.al, Stable-Baselines3: Reliable Reinforcement Learning Implementations, 22(268):1−8, 2021.
[^2]: Ahmed Fawzy Gad, PyGAD: An Intuitive Genetic Algorithm Python Library, arXiv:2106.06158 
