from env_preprocessors import PreprocessFrame, RepeatActionAndMaxFrame, StackFrames
import gym

def make_env(env_name, resize_shape=(84,84,1), repeat=4, 
            clip_rewards=False, no_ops=0):
    env = gym.make(env_name)
    env = RepeatActionAndMaxFrame(env, repeat, clip_rewards, no_ops)
    env = PreprocessFrame(env, resize_shape)
    env = StackFrames(env, repeat)

    return env
