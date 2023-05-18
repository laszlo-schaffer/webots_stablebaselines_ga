import numpy as np
import os
import gym
from controller import Supervisor


# ranges must have the same length as the number of parameters in the dict.
params = {
    'TRPO': {
        'parameters': ['learning_rate', 'batch_size', 'gamma', 'n_critic_updates', 'gae_lambda', 'target_kl'],
        'ranges': [{'low': 0.0001, 'high': 0.1},
                   [64, 128, 256, 512],
                   {'low': 0.0, 'high': 1.0},
                   np.arange(10, 110, 10).tolist(),
                   {'low': 0.0, 'high': 1.0},
                   {'low': 0.0, 'high': 1.0}],
        'types': [float, int, float, int, float, float],
        "discrete": True, "total_timesteps": 10_000},
    'DQN': {
        'parameters': ['learning_rate', 'batch_size', 'gamma', 'target_update_interval', 'learning_starts', 'tau'],
        'ranges': [{'low': 0.0001, 'high': 0.1},
                   [64, 128, 256, 512],
                   {'low': 0.0, 'high': 1.0},
                   np.arange(1_000, 100_000 + 1_000, 1_000).tolist(),
                   np.arange(1_000, 100_000 + 1_000, 1_000).tolist(),
                   {'low': 0.0, 'high': 1.0}],
        'types': [float, int, float, int, int, float],
        'discrete': True, "total_timesteps": 100_000},
    'PPO': {
        'parameters': ['learning_rate', 'batch_size', 'gamma', 'n_epochs', 'gae_lambda', 'target_kl'],
        'ranges': [{'low': 0.0001, 'high': 0.1},
                   [64, 128, 256, 512],
                   {'low': 0.0, 'high': 1.0},
                   np.arange(10, 110, 10).tolist(),
                   {'low': 0.0, 'high': 1.0},
                   {'low': 0.0, 'high': 1.0}],
        'types': [float, int, float, int, float, float],
        'discrete': False, "total_timesteps": 10_000},
    'SAC': {
        'parameters': ['learning_rate', 'batch_size', 'gamma', 'target_update_interval', 'learning_starts', 'tau'],
        'ranges': [{'low': 0.0001, 'high': 0.1},
                   [64, 128, 256, 512],
                   {'low': 0.0, 'high': 1.0},
                   np.arange(1_000, 10_000 + 1_000, 1_000).tolist(),
                   np.arange(1_000, 10_000 + 1_000, 1_000).tolist(),
                   {'low': 0.0, 'high': 1.0}],
        'types': [float, int, float, int, int, float],
        'discrete': False, "total_timesteps": 10_000},
    'TD3': {
        'parameters': ['learning_rate', 'batch_size', 'gamma', 'learning_starts', 'tau'],
        'ranges': [{'low': 0.0001, 'high': 0.1},
                   [64, 128, 256, 512],
                   {'low': 0.0, 'high': 1.0},
                   np.arange(1_000, 10_000 + 1_000, 1_000).tolist(),
                   {'low': 0.0, 'high': 1.0}],
        'types': [float, int, float, int, float],
        'discrete': False, "total_timesteps": 10_000},
    'DDPG': {
        'parameters': ['learning_rate', 'batch_size', 'gamma', 'learning_starts', 'tau'],
        # There is no policy delay, or target_update
        'ranges': [{'low': 0.0001, 'high': 0.1},
                   [64, 128, 256, 512],
                   {'low': 0.0, 'high': 1.0},
                   np.arange(1_000, 10_000 + 1_000, 1_000).tolist(),
                   {'low': 0.0, 'high': 1.0}],
        'types': [float, int, float, int, float],
        'discrete': False, "total_timesteps": 10_000}
}


def to_discrete(action):
    if action[0] == action[1]:
        if action[0] == 0.0:
            action = 0
        elif action[0] == 12.0:
            action = 1
    elif action[0] > action[1]:
        action = 2
    elif action[0] < action[1]:
        action = 3
    return np.array([action])


def to_continuous(action, lower=0.0, upper=12.0):
    if action == 0:
        action = [lower, lower]
    elif action == 1:
        action = [upper, upper]
    elif action == 2:
        action = [upper, lower]
    elif action == 3:
        action = [lower, upper]
    return np.array(action)


def get_relative_orientation(robot_pos, robot_orientation, goal_pos):
    phi_robot = robot_orientation + np.pi  # 0-360
    phi_togoal = np.arctan2(robot_pos[1] - goal_pos[1], robot_pos[0] - goal_pos[0])  # θ = atan2(y2 - y1, x2 - x1)
    if phi_togoal < 0:
        phi_togoal += np.pi * 2
    ori_diff = phi_togoal - phi_robot
    if ori_diff > np.pi:
        ori_diff -= 2 * np.pi
    if ori_diff < 0:
        if ori_diff < -np.pi:
            ori_diff += 2 * np.pi
    return ori_diff


arena_size = 100


class PioneerPath(Supervisor, gym.Env):
    def __init__(self, max_episode_steps=500, discrete=True):
        super().__init__()
        self.discrete = discrete
        self.steps = 0
        self.__timestep = int(self.getBasicTimeStep())

        # self.dist_decay = 0.1
        self.goal = self.getFromDef("goal")
        self.goal_pos = self.goal.getPosition()[:2]

        self.robot = self.getSelf()  # Grab the robot reference from the supervisor to access various robot methods
        self.imu = self.getDevice("imu")
        self.imu.enable(self.__timestep)
        self.init_pos = self.robot.getPosition()[:2]
        self.current_pos = self.init_pos

        self.wheel_velocity = np.zeros((2,))
        self.velocity = 0.0
        self.sensor_list = [0, 1, 2, 3, 4, 5, 6, 7]
        self.number_of_sensors = len(self.sensor_list)

        self.observation_space = gym.spaces.Box(
            low=np.concatenate((np.array([0.0]), np.array([-np.sqrt(2) * arena_size]),
                                np.array([-np.pi / 2]))),
            high=np.concatenate((np.array([12.0]), np.array([np.sqrt(2) * arena_size]),
                                 np.array([np.pi / 2]))),
            dtype=np.float32)
        if self.discrete:
            self.action_space = gym.spaces.Discrete(4)
        else:
            self.action_space = gym.spaces.Box(low=np.array([0., 0.]),
                                               high=np.array([12., 12.]),
                                               dtype=np.float32)

        self.init_dist = np.sqrt(
            np.square(self.goal_pos[0] - self.init_pos[0]) + np.square(self.goal_pos[1] - self.init_pos[1]))
        self.last_dist_error = None
        # print("Robot init. coords:", self.init_pos, ", goal coordinates:", self.goal_pos)
        self.pos_sensors = []
        self.pos_sensors.append(self.getDevice('left wheel sensor'))
        self.pos_sensors.append(self.getDevice('right wheel sensor'))

        for ps in self.pos_sensors:
            ps.enable(self.__timestep)

        self.sensors = []
        sc = 0
        for s in self.sensor_list:
            self.sensors.append(self.getDevice('so' + str(s)))
            self.sensors[sc].enable(self.__timestep)
            sc += 1

        self.number_of_states = self.observation_space.shape[0]
        # print("Number of states:", self.number_of_states)
        self.state = np.zeros((self.number_of_states,))
        self.last_state = np.zeros((self.number_of_states,))

        self.wheels = []
        for wheelName in ['left wheel', 'right wheel']:
            wheel = self.getDevice(wheelName)  # Get the wheel handle
            wheel.setPosition(float('inf'))  # Set starting position
            wheel.setVelocity(0.0)  # Zero out starting velocity
            self.wheels.append(wheel)

        self.stepsPerEpisode = max_episode_steps  # Max number of steps per episode
        self.episodeScore = 0  # Score accumulated during an episode
        self.episodeScoreList = []  # A list to save all the episode scores, used to check if task is solved
        self.trainsteps = 0

        self.spec = gym.envs.registration.EnvSpec(id='PioneerEnv-v0', max_episode_steps=max_episode_steps)

    def set_action_space(self, discrete):
        self.discrete = discrete
        if discrete:
            self.action_space = gym.spaces.Discrete(4)
        else:
            self.action_space = gym.spaces.Box(low=np.array([0., 0.]),
                                               high=np.array([12., 12.]),
                                               dtype=np.float32)

    def reset(self):
        # Reset the simulation
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.__timestep)

        self.init_dist = np.sqrt(
            np.square(self.goal_pos[0] - self.init_pos[0]) + np.square(self.goal_pos[1] - self.init_pos[1]))
        self.last_dist_error = None
        # print("Robot init. coords:", self.init_pos, ", goal coordinates:", self.goal_pos)
        self.pos_sensors = []
        self.pos_sensors.append(self.getDevice('left wheel sensor'))
        self.pos_sensors.append(self.getDevice('right wheel sensor'))

        for ps in self.pos_sensors:
            ps.enable(self.__timestep)

        self.sensors = []
        sc = 0
        for s in self.sensor_list:
            self.sensors.append(self.getDevice('so' + str(s)))
            self.sensors[sc].enable(self.__timestep)
            sc += 1

        self.number_of_states = self.observation_space.shape[0]
        # print("Number of states:", self.number_of_states)
        self.state = np.zeros((self.number_of_states,))
        self.last_state = np.zeros((self.number_of_states,))

        self.wheels = []
        for wheelName in ['left wheel', 'right wheel']:
            wheel = self.getDevice(wheelName)  # Get the wheel handle
            wheel.setPosition(float('inf'))  # Set starting position
            wheel.setVelocity(0.0)  # Zero out starting velocity
            self.wheels.append(wheel)

        self.episodeScore = 0  # Score accumulated during an episode
        self.episodeScoreList = []  # A list to save all the episode scores, used to check if task is solved
        self.trainsteps = 0

        # Internals
        super().step(self.__timestep)

        # Open AI Gym generic
        return np.zeros(self.number_of_states).astype(np.float32)

    def step(self, action):
        # Execute the action
        if np.isnan(action).any():
            if self.discrete:
                action = np.array([0, 0, 0, 0])
            else:
                action = np.array([0.0, 0.0])
        # print(action)
        if self.discrete:
            action = to_continuous(action)
        for i in range(len(self.wheels)):
            self.wheels[i].setPosition(float('inf'))
            self.wheels[i].setVelocity(action[i])
        super().step(self.__timestep)

        # Observation
        self.last_state = self.state.copy()
        self.goal_pos = self.goal.getPosition()
        self.current_pos = self.robot.getPosition()
        sc = 0  # state_counter
        for s in range(len(self.wheel_velocity)):
            self.wheel_velocity[s] = self.wheels[s].getVelocity()
        self.velocity = ((self.wheel_velocity[0] + self.wheel_velocity[1]) / 2)
        self.state[sc] = self.velocity
        sc += 1
        distance = np.sqrt(
            np.square(self.goal_pos[0] - self.current_pos[0]) + np.square(self.goal_pos[1] - self.current_pos[1]))
        self.state[sc] = distance
        sc += 1
        ori_diff = get_relative_orientation(self.current_pos, self.imu.getRollPitchYaw()[2], self.goal_pos)
        self.state[sc] = ori_diff
        sc += 1
        if self.trainsteps == 1:
            self.last_state = self.state.copy()

        # Done
        beta = 0.9
        if self.trainsteps < 1:
            self.velocity_avg = self.velocity
        self.velocity_avg = beta * self.velocity_avg + (1 - beta) * self.velocity
        if self.trainsteps >= self.stepsPerEpisode - 5:
            done = True
        else:
            done = bool(
                distance >= 1 and self.velocity_avg < 0.1 and self.trainsteps >= 5  # or
            )

        # Reward
        reward_local = 0
        ori_diff = get_relative_orientation(self.current_pos, self.imu.getRollPitchYaw()[2], self.goal_pos)
        if np.abs(ori_diff) <= np.deg2rad(20):
            reward_local += np.abs(ori_diff) / (np.deg2rad(20) * 2)  # max 0.5 point
        else:
            reward_local -= np.abs(ori_diff) / np.pi  # max 1 point
        distance_ratio = (1 - (distance / self.init_dist)) * 0.5  # max 0.5

        if self.velocity == 0 and distance > 1:
            reward_local += -1.0
        elif self.velocity == 0 and distance < 1:
            reward_local += 1.0
        elif self.velocity > 0 and distance < 1:
            reward_local += 0.5
        elif self.velocity > 0 and distance > self.init_dist# ekv.: distance_ratio < 0:
            reward_local += -0.5
        #else:  # it is not required, because in this case velocity is negative, that is not possible in this env.
        #    reward_local += -distance_ratio
        reward_local = 0 if done else reward_local
        self.trainsteps += 1

        return self.state.astype(np.float32), reward_local, done, {}
