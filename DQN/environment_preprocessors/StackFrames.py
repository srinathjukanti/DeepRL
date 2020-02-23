import gym
import numpy as np
import collections

class StackFrame(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super(StackFrame, self).__init__(env)
        self.stack = collections.deque(maxlen=repeat)
        self.observation_space = gym.spaces.Box(
            env.observation_space.low.repeat(repeat, axis=0),
            env.observation_space.high.repeat(repeat, axis=0),
            dtype=np.float32
        )

    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for _ in range(self.repeat):
            self.stack.append(observation)
        
        return np.array(self.stack).reshape(self.observation_space.low.shape)

    def observation(self, observation):
        self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)