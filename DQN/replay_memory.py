import numpy as np

class ReplayMemory():
    def __init__(self, max_size, input_shape, n_actions):
        self.memory_size = max_size
        self.memory_counter = 0
        self.state_memory = np.zeros((self.memory_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = self.state_memory.copy()
        self.action_memory = np.zeros(self.memory_size, dtype=np.int64)
        self.dones_memory = np.zeros(self.memory_size, dtype=np.uint8)
        self.rewards_memory = np.zeros(self.memory_size, dtype=np.float32)

    def remember(self, state, action, reward, next_state, done):
        index = self.memory_counter % self.memory_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.next_state_memory[index] = next_state
        self.dones_memory[index] = done
        self.rewards_memory[index] = reward
        self.memory_counter += 1

    def sample_buffer(self, batch_size):
        sample_space = min(self.memory_counter, self.memory_size)
        batch = np.random.choice(sample_space, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.rewards_memory[batch]
        next_states = self.next_state_memory[batch]
        dones = self.dones_memory[batch]

        return states, actions, rewards, next_states, dones


