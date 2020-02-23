from env_preprocessors import PreprocessFrame, RepeatActionAndMaxFrame, StackFrames
import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor
gymlogger.set_level(40) #error only
import torch
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import math
import glob
import io
import base64
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from IPython.core.debugger import set_trace
from IPython.display import HTML
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display

def make_env(env_name, resize_shape=(84,84,1), repeat=4, 
            clip_rewards=False, no_ops=0):
    env = gym.make(env_name)
    env = RepeatActionAndMaxFrame(env, repeat, clip_rewards, no_ops)
    env = PreprocessFrame(env, resize_shape)
    env = StackFrames(env, repeat)

    return env

def show_video():
  mp4list = glob.glob('video/*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[0]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
  else: 
    print("Could not find video")
    

def wrap_env(env):
  env = Monitor(env, './video', force=True)
  return env