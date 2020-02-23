import numpy as np
import torch as T
from deep_q_network import DeepQNetwork
from replay_memory import ReplayMemory

class DDQNAgent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 memory_size, batch_size, algo, env_name, epsilon_min=0.01,
                 epsilon_decay=5e-7, replace_target_count=1000, checkpoint_dir='models/'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.algo = algo
        self.env_name = env_name
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.replace_target_count = replace_target_count
        self.checkpoint_dir = checkpoint_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayMemory(memory_size, input_dims, n_actions)

        self.q_net = DeepQNetwork(self.lr, self.n_actions, 
                                  name=self.env_name+'_'+self.algo+'_q_net',
                                  input_dims=self.input_dims,
                                  checkpoint_dir=self.checkpoint_dir)
        self.target_net = DeepQNetwork(self.lr, self.n_actions, 
                                  name=self.env_name+'_'+self.algo+'_target_net',
                                  input_dims=self.input_dims,
                                  checkpoint_dir=self.checkpoint_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            observation = T.tensor([observation], dtype=T.float).to(self.q_net.device)
            actions = self.q_net(observation)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.remember(state, action, reward, next_state, done)

    def sample_memory(self):
        states, actions, rewards, next_states, dones = \
                                self.memory.sample_buffer(self.batch_size)
        
        states = T.tensor(states).to(self.q_net.device)
        actions = T.tensor(actions).to(self.q_net.device)
        rewards = T.tensor(rewards).to(self.q_net.device)
        next_states = T.tensor(next_states).to(self.q_net.device)
        dones = T.tensor(dones).to(self.q_net.device)

        return states, actions, rewards, next_states, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_count == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_decay \
                            if self.epsilon > self.epsilon_min else self.epsilon_min
            
    def learn(self):
        if self.memory.memory_counter < self.batch_size:
            return
        
        self.q_net.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, next_states, dones = self.sample_memory()

        q_prediction = self.q_net(states) # (batch_size, *n_actions)
        target_predictions = self.target_net(next_states) # (batch_size, *n_actions)
        target_predictions[dones] = 0.0
        
        indices = np.arange(self.batch_size)
        q_value = q_prediction[indices, actions]

        t_actions = T.argmax(self.q_net(next_states))
        target_value = rewards + self.gamma * target_predictions[indices, t_actions]

        loss = self.q_net.loss(q_value, target_value).to(self.q_net.device)
        loss.backward()
        self.q_net.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
            
    def save_models(self):
        self.q_net.save_checkpoint()
        self.target_net.save_checkpoint()

    def load_models(self):
        self.q_net.load_checkpoint()
        self.target_net.load_checkpoint()
