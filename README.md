# Parameter optimisation of different reinforcement learning algorithms using genetic algorithm in Webots environment

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
The framework uses stable baselines RL algorithms and pygad genetic algorithm for parameter optimisation in a Webots based environment. In this example a cartpole based sample Webots world is used, it can be found on the [Webots webpage](https://www.cyberbotics.com/doc/guide/samples-howto?version=master#openai_gym-wbt).

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
                                
[^1]: Antonin Raffin et.al, Stable-Baselines3: Reliable Reinforcement Learning Implementations, 22(268):1−8, 2021.
[^2]: Ahmed Fawzy Gad, PyGAD: An Intuitive Genetic Algorithm Python Library, arXiv:2106.06158 
