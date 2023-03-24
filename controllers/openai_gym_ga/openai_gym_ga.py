# Copyright 1996-2022 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import logging
from logging.handlers import WatchedFileHandler
from controller import Supervisor
from utils import params, OpenAIGymEnvironment

try:
    import gym
    import numpy as np
    from stable_baselines3 import PPO, DQN, DDPG, SAC, TD3
    from stable_baselines3.common.logger import configure
    from sb3_contrib import TRPO
    from stable_baselines3.common.env_checker import check_env
    import pygad
except ImportError:
    sys.exit(
        'Please make sure you have all dependencies installed. '
        'Run: "pip3 install numpy gym stable_baselines3 sb3_contrib pygad"'
    )
    
# global variables
env_global = None
alg_global = None
train_global = None
total_timesteps_global = 100_000
ga_instance = None

# stablebaselines logger
tmp_path = ""
new_logger = None


def log_generation(ga_inst):
    ga_log = logging.getLogger("ga_log")
    ga_log.info("GEN;{}".format(ga_inst.generations_completed))


def fitness_function(solution, solution_idx):
    global env_global, alg_global, train_global, total_timesteps_global, new_logger
    ga_log = logging.getLogger("ga_log")
    reward = run_env(env_global, alg_global, train_global, total_timesteps=total_timesteps_global,
                  replay=True, solution=solution, logger=new_logger)
    print("Reward:", reward)
    ga_log.info("{};{}".format(solution, reward))
    return reward


def run_env(env=None, alg=None, train=True, total_timesteps=100_000, replay=True, logger=None, max_reward=500,
         solution=None, seed=1):
    if env is None:
        print("No environment!")
        return

    # Train
    save_dir = os.path.join("models", alg)
    # print(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    if train:
        info = []
        for z in zip(params[alg]['parameters'], solution):
            info.append(z)
        print(info, "total timesteps:", total_timesteps)

        if alg == "PPO":
            model = PPO('MlpPolicy',
                        env,
                        learning_rate=solution[0],
                        batch_size=solution[1],
                        gamma=solution[2],
                        n_epochs=solution[3],
                        gae_lambda=solution[4],
                        target_kl=solution[5],
                        seed=seed,
                        verbose=1)
        elif alg == "TRPO":
            model = TRPO('MlpPolicy',
                         env,
                         verbose=0,
                         learning_rate=solution[0],
                         batch_size=solution[1],
                         gamma=solution[2],
                         n_critic_updates=solution[3],
                         gae_lambda=solution[4],
                         target_kl=solution[5],
                         seed=seed)
        elif alg == "DQN":
            model = DQN('MlpPolicy',
                        env,
                        verbose=0,
                        learning_rate=solution[0],
                        batch_size=solution[1],
                        gamma=solution[2],
                        target_update_interval=solution[3],
                        learning_starts=solution[4],
                        tau=solution[5],
                        seed=seed
                        )
        elif alg == "DDPG":
            model = DDPG('MlpPolicy',
                         env,
                         learning_rate=solution[0],
                         batch_size=solution[1],
                         gamma=solution[2],
                         learning_starts=solution[3],
                         tau=solution[4],
                         seed=seed,
                         verbose=1)
        elif alg == "SAC":
            model = SAC('MlpPolicy',
                        env,
                        learning_rate=solution[0],
                        batch_size=solution[1],
                        gamma=solution[2],
                        target_update_interval=solution[3],
                        learning_starts=solution[4],
                        tau=solution[5],
                        seed=seed,
                        verbose=1)
        elif alg == "TD3":
            model = TD3('MlpPolicy',
                        env,
                        learning_rate=solution[0],
                        batch_size=solution[1],
                        gamma=solution[2],
                        policy_delay=solution[3],
                        learning_starts=solution[4],
                        tau=solution[5],
                        seed=1,
                        verbose=1)
        elif alg is None:
            print("No algoritm defined!")
            return
        if logger is not None:
            model.set_logger(logger)
        model.learn(total_timesteps=total_timesteps)

        # Save model
        model.save(os.path.join(save_dir, alg + ".zip"))
        if not replay:
            del model
        # Replay
        if replay:
            # print('Training is finished, press `Y` for replay...')
            # env.wait_keyboard()

            obs = env.reset()
            done = False
            reward_sum = 0
            while not done:
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                # print(obs, reward, done, info)
                # if done:
                #     obs = env.reset()
                reward_sum += reward
                if reward_sum >= max_reward:
                    break
            return reward_sum
        else:
            return 0

    else:
        # load model
        if alg == "PPO":
            model = PPO.load(os.path.join(save_dir, alg + ".zip"))
        elif alg == "TRPO":
            model = TRPO.load(os.path.join(save_dir, alg + ".zip"))
        elif alg == "DQN":
            model = DQN.load(os.path.join(save_dir, alg + ".zip"))
        elif alg == "DDPG":
            model = DDPG.load(os.path.join(save_dir, alg + ".zip"))
        elif alg == "SAC":
            model = SAC.load(os.path.join(save_dir, alg + ".zip"))
        elif alg == "TD3":
            model = TD3.load(os.path.join(save_dir, alg + ".zip"))
        else:
            print("No algoritm defined, or save does not exists!")
            return
        obs = env.reset()
        done = False
        reward_sum = 0
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            # print(obs, reward, done, info)
            # if done:
            #     obs = env.reset()
            reward_sum += reward
            if reward_sum >= max_reward:
                break
        return reward_sum


if __name__ == '__main__':
    # Initialize the environment

    alg_list = ["TRPO", "PPO", "DQN", "DDPG", "SAC", "TD3"]  # full list
    alg_to_run = "TRPO"
    # alg_to_run = None

    # with the given solution vector
    # sol = [256, 0.878810722275697, 4000, 72000, 0.749925080419856]  # DQN best
    # sol = [64, 0.42606948276819134, 50, 0.265483348858523, 0.03482744338983934]  # PPO best
    # sol = [128, 0.5319043091499475, 5200, 5400, 0.6528674693902464]  # SAC best
    sol = None

    train = False  # if True and alg_to_run is not None, then trains only with that algorithm
    train_global = train
    discrete = False
    env_global = OpenAIGymEnvironment(max_episode_steps=2048, discrete=discrete)
    check_env(env_global)

    if train:
        if sol is None:
            if alg_to_run is not None:
                alg_list = [alg_to_run]
            for alg in alg_list:
                try:
                    alg_global = alg

                    # Init ga logger
                    handler = WatchedFileHandler(os.path.join("logs", alg, "ga_log.log"))
                    formatter = logging.Formatter("%(asctime)s.%(msecs)03d;%(levelname)s;%(message)s",
                                                  "%Y-%m-%d %H:%M:%S")
                    handler.setFormatter(formatter)
                    ga_log = logging.getLogger("ga_log")
                    ga_log.setLevel(os.environ.get("LOGLEVEL", "DEBUG"))
                    ga_log.addHandler(handler)

                    # configure training logger
                    tmp_path = os.path.join("logs", alg, "")
                    # set up logger
                    new_logger = configure(tmp_path, ["csv", "tensorboard"])

                    # construct gene space
                    gene_space = []
                    gene_types = []
                    for idx, ra in enumerate(params[alg]['ranges']):
                        gene_space.append(ra)
                    gene_types = params[alg]['types']

                    discrete = params[alg]['discrete']
                    total_timesteps_global = params[alg]['total_timesteps']
                    # GA
                    print(len(gene_space), gene_space)
                    ga_instance = pygad.GA(num_generations=5,
                                           num_parents_mating=2,
                                           sol_per_pop=25,
                                           num_genes=len(gene_space),
                                           fitness_func=fitness_function,
                                           # init_range_low=5,
                                           # init_range_high=15,
                                           mutation_type="random",
                                           gene_type=gene_types,
                                           gene_space=gene_space,
                                           on_generation=log_generation)

                    env_global.set_action_space(discrete)
                    ga_instance.run()
                    solution, solution_fitness, solution_idx = ga_instance.best_solution()

                    print("Parameters of the best solution : {solution}".format(solution=solution))
                    print(
                        "Fitness value of the best solution = {solution_fitness}".format(
                            solution_fitness=solution_fitness))
                    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

                    # ValueError: ctypes objects containing pointers cannot be pickled
                    # filename = 'genetic'
                    # save_dir = os.path.join("models", alg, filename)
                    # ga_instance.save(filename=filename)

                    # Run the training again with the best solution, but with longer timeframe
                    total_timesteps_global += 50_000
                    reward = fitness_function(solution, solution_idx)

                except AssertionError as e:
                    print(alg, e)
        else:

            alg_global = alg_to_run
            discrete = params[alg_to_run]['discrete']
            env_global.set_action_space(discrete)
            total_timesteps_global = params[alg_to_run]['total_timesteps']
            # configure logger
            tmp_path = os.path.join("logs", alg_to_run, "")
            # set up logger
            new_logger = configure(tmp_path, ["csv", "tensorboard"])

            # Run the training again with a given solution
            total_timesteps_global += 100_000
            reward = fitness_function(sol, 0)
    else:
        try:
            discrete = params[alg_to_run]['discrete']

            env_global.set_action_space(discrete)
            run_env(env_global, alg_to_run, train=False)
        except AssertionError as e:
            print(alg_to_run, e)
