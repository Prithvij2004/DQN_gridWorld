import numpy as np
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense

class ReplayBuffer():
    def __init__(self, max_size, input_dims):
        self.mem_size = max_size
        self.mem_counter = 0

        self.state_buff = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.next_state_buff = np.zeros((self.mem_size, input_dims), dtype=np.float32)

        self.action_buff = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_buff = np.zeros(self.mem_size, dtype=np.float32)
        self.terminated_buff = np.zeros(self.mem_size, dtype=np.int32)

    def store_to_memory(self, state, action, reward, next_State, done):
        index = self.mem_counter % self.mem_size

        self.state_buff[index] = state
        self.action_buff[index] = action
        self.reward_buff[index] = reward
        self.next_state_buff[index] = next_State
        self.terminated_buff[index] = done

        self.mem_counter += 1
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)

        batch_memory = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_buff[batch_memory]
        actions = self.action_buff[batch_memory]
        rewards = self.reward_buff[batch_memory]
        next_states = self.next_state_buff[batch_memory]
        terminated = self.terminated_buff[batch_memory]

        return states, actions, rewards, next_states, terminated
    
def build_network(lr, input_dims, output_dims):
    model = Sequential([
        keras.layers.Dense(10, activation='relu', input_shape=(input_dims,)),
        # keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(output_dims, activation="linear")
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mean_squared_error')
    print(model.summary())
    
    return model