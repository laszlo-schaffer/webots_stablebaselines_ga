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
        'parameters': ['learning_rate', 'batch_size', 'gamma', 'policy_delay', 'learning_starts', 'tau'],
        'ranges': [{'low': 0.0001, 'high': 0.1},
                   [64, 128, 256, 512],
                   {'low': 0.0, 'high': 1.0},
                   {'low': 1, 'high': 100},
                   np.arange(1_000, 10_000 + 1_000, 1_000).tolist(),
                   {'low': 0.0, 'high': 1.0}],
        'types': [float, int, float, int, int, float],
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


def to_discrete(action, low, high):
    if action < (low + high) / 2:
        action = low
    else:
        action = high
    return action


class OpenAIGymEnvironment(Supervisor, gym.Env):
    def __init__(self, max_episode_steps=2048, discrete=True):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.discrete = discrete
        self.steps = 0
        # Open AI Gym generic
        self.theta_threshold_radians = 0.2
        self.x_threshold = 0.3
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max
            ],
            dtype=np.float32
        )
        if self.discrete:
            self.action_space = gym.spaces.Discrete(2)
        else:
            self.action_space = gym.spaces.Box(low=np.array([0.0]), high=np.array([1.0]))

        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.state = None
        self.spec = gym.envs.registration.EnvSpec(id='WebotsEnv-v0', max_episode_steps=max_episode_steps)

        # Environment specific
        self.__timestep = int(self.getBasicTimeStep())
        self.__wheels = []
        self.__pendulum_sensor = None

        # Tools
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.__timestep)

    def set_action_space(self, discrete):
        self.discrete = discrete
        if discrete:
            self.action_space = gym.spaces.Discrete(2)
        else:
            self.action_space = gym.spaces.Box(low=np.array([0.0]), high=np.array([1.0]))

    def wait_keyboard(self):
        while self.keyboard.getKey() != ord('Y'):
            super().step(self.__timestep)

    def reset(self):
        # Reset the simulation
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.__timestep)
        self.steps = 0
        # Motors
        self.__wheels = []
        for name in ['back left wheel', 'back right wheel', 'front left wheel', 'front right wheel']:
            wheel = self.getDevice(name)
            wheel.setPosition(float('inf'))
            wheel.setVelocity(0)
            self.__wheels.append(wheel)

        # Sensors
        self.__pendulum_sensor = self.getDevice('position sensor')
        self.__pendulum_sensor.enable(self.__timestep)

        # Internals
        super().step(self.__timestep)

        # Open AI Gym generic
        return np.array([0, 0, 0, 0]).astype(np.float32)

    def step(self, action):
        # Execute the action
        for wheel in self.__wheels:
            if not self.discrete:
                action = to_discrete(action, low=self.action_space.low, high=self.action_space.high)
            wheel.setVelocity(1.3 if action == 1 else -1.3)
        super().step(self.__timestep)
        self.steps += 1

        # Observation
        robot = self.getSelf()
        endpoint = self.getFromDef("POLE_ENDPOINT")
        self.state = np.array([robot.getPosition()[0], robot.getVelocity()[0],
                               self.__pendulum_sensor.getValue(), endpoint.getVelocity()[4]])

        # Done
        if self.steps >= self.max_episode_steps:
            done = True
        else:
            done = bool(
                self.state[0] < -self.x_threshold or
                self.state[0] > self.x_threshold or
                self.state[2] < -self.theta_threshold_radians or
                self.state[2] > self.theta_threshold_radians
            )

        # Reward
        reward = 0 if done else 1

        return self.state.astype(np.float32), reward, done, {}
