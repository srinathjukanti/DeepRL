import numpy as np
import torch as T
from policy_network import ReinforcePolicyNet
from torch.distributions import Categorical

class ReinforceAgent():
    def __init__(self, gamma, lr, n_actions, input_dims, algo, env_name, checkpoint_dir):
        self.gamma = gamma
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.algo = algo
        self.env_name = env_name
        self.checkpoint_dir = checkpoint_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.log_probs = []
        self.rewards = []

        self.net = ReinforcePolicyNet(self.lr, self.n_actions, 
                                  name=self.env_name+'_'+self.algo+'_net',
                                  input_dims=self.input_dims,
                                  checkpoint_dir=self.checkpoint_dir)
        self.eps = np.finfo(np.float32).eps.item()

    def choose_action(self, observation):
        observation = T.from_numpy(observation).float().to(self.net.device)
        predictions = self.net(observation)
        distribution = Categorical(predictions)
        action = distribution.sample()
        self.log_probs.append(distribution.log_prob(action))
        return action.item()

    def learn(self):
        self.q_net.optimizer.zero_grad()
        G = 0
        returns = []
        losses = []
        for r in self.rewards[::-1]: #Iterate in reverse for easier computation of returns
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = T.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for G, log_prob in zip(returns, self.log_probs):
            losses.append(-(G * log_prob))

        loss = T.cat(losses).sum()
        loss.backward()
        self.net.optim.step()

        del returns[:]
        del self.log_probs[:]

    def save_models(self):
        self.q_net.save_checkpoint()
        self.target_net.save_checkpoint()

    def load_models(self):
        self.q_net.load_checkpoint()
        self.target_net.load_checkpoint()
