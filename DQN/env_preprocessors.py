import gym
import numpy as np
import cv2
import collections

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super(PreprocessFrame, self).__init__(env)
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                     shape=self.shape, dtype=np.float32)

    def observation(self, observation):
        new_frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(new_frame, self.shape[1:],
                                interpolation=cv2.INTER_AREA)
        new_observation = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        new_observation /= 255.0

        return new_observation

class RepeatActionAndMaxFrame(gym.Wrapper):
    def __init__(self, env, action_repeat=4, clip_rewards=False, no_ops=30):
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.env = env
        self.action_repeat = action_repeat
        self.shape = env.observation_space.low.shape
        self.frame_buffer = np.zeros_like((2, self.shape))
        self.clip_rewards = clip_rewards
        self.no_ops = no_ops
    def step(self, action):
        t_reward = 0.0
        done = False
        for i in range(self.action_repeat):
            observation, reward, done, info = self.env.step(action)
            if self.clip_rewards:
                reward = np.clip(np.array([reward]), -1, 1)[0]
            t_reward += reward
            self.frame_buffer[i%2] = observation
            if done:
                break
        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return max_frame, t_reward, done, info

    def reset(self):
        observation = self.env.reset()
        no_ops = np.random.randint(self.no_ops)+1 if self.no_ops > 0 else 0
        for i in range(no_ops):
            _, _, done, _ = self.step(0)
            if done:
                self.reset()
        self.frame_buffer = np.zeros_like((2, self.shape))
        self.frame_buffer[0] = observation

        return observation

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