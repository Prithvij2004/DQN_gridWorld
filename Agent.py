import numpy as np
import operator
from tensorflow import keras
from Replay_Buffer import ReplayBuffer, build_network

class Agent:
    def __init__(self, lr, gamma, output_dims, batch_size, input_dims, mem_size=1000000, fname='dqn_model.keras'):
        
        self.learning_rate = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.model_name = fname
        self.action_size = output_dims
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_network = build_network(self.learning_rate, input_dims, output_dims)
        self.target_network = build_network(self.learning_rate, input_dims, output_dims)
        self.weights_update()

    def store_to_memory(self, state, action, reward, next_state, done):
        self.memory.store_to_memory(state, action, reward, next_state, done)
    
    def choose_action(self, state, epsilon):
        rand = np.random.random()
        if rand < epsilon:
            action = np.random.randint(0, self.action_size)
        else:
            action_values = self.q_network.predict(state, verbose=0)
            # print(actions)
            action = np.argmax(action_values)
        
        return action
    

    def train(self):
        if self.memory.mem_counter < self.batch_size:
            return
        
        states, actions, rewards, next_states, done = self.memory.sample_buffer(self.batch_size)
        
        q_state = self.q_network.predict(states, verbose=0)
        q_next_state = self.target_network.predict(next_states, verbose=0)

        target_value_of_q_network = np.copy(q_state)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        target_value_of_q_network[batch_index, actions] = rewards + self.gamma*np.max(q_next_state, axis=1)*(1-done)

        loss = self.q_network.train_on_batch(states, target_value_of_q_network)
        return loss

    
    def save_model(self):
        self.q_network.save(self.model_name)

    def load_model(self):
        self.q_network = keras.models.load_model(self.model_name)
    
    def weights_update(self):
        self.target_network.set_weights(self.q_network.get_weights())

