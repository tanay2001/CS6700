from config import *
import time
import random
import numpy as np
"""

DO not modify the structure of "class Agent".
Implement the functions of this class.
Look at the file run.py to understand how evaluations are done. 

There are two phases of evaluation:
- Training Phase
The methods "registered_reset_train" and "compute_action_train" are invoked here. 
Complete these functions to train your agent and save the state.

- Test Phase
The methods "registered_reset_test" and "compute_action_test" are invoked here. 
The final scoring is based on your agent's performance in this phase. 
Use the state saved in train phase here. 

"""
def softmax(x): 

  return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))

def get_state(ls):
    try:
        i = ls.index('')
    except:
        i = 16  #all full, no ""
    if i!=0:
        val = ls[i-1]   # val =[1,0]
    else:
        val=1
    index = i if val ==1 else 16+i
    return index

class Agent:
    def __init__(self, env):
        self.env_name = env
        self.config = config[self.env_name]
        self.old_state = None
        self.action = None
        np.random.seed(0)
        random.seed(0)

        if self.env_name =='taxi':
            self.alpha = 0.8
            self.epsilon = 0.9
            self.gamma = 0.96
            self.q_table = [[0 for i in range(6)] for j in range(500)]
            self.action_size = 6

        elif self.env_name =='kbca':
            self.alpha = 0.2
            self.epsilon = 1
            self.gamma = 0.95
            self.decay = 0.999999
            self.q_table = np.zeros([33,2])
            self.action_size = 2
            self.epsilon_bound = 0.3


        elif self.env_name=='kbcb':

            self.alpha = 0.9
            self.epsilon = 1
            self.gamma = 0.7
            self.decay = 0.99999
            self.q_table = np.zeros([33,2])
            self.action_size = 2

        elif self.env_name =='kbcc':

            self.alpha = 0.5
            self.epsilon = 1
            self.gamma = 0.99
            self.decay = 0.9999
            self.q_table = np.zeros([33,3])
            self.action_size = 3

        elif self.env_name =='acrobot':

            self.alpha=0.06
            self.weights = np.random.normal(0, 1 / np.sqrt(6), (3, 6))


    def register_reset_train(self, obs):
        """
        Use this function in the train phase
        This function is called at the beginning of an episode. 
        PARAMETERS  : 
            - obs - raw 'observation' from environment
        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        """
        if self.env_name == 'taxi':
            action = np.random.randint(0, self.action_size)

        elif self.env_name =='kbca':
            obs = get_state(obs)
            action = 1

        elif self.env_name =='kbcb':
            obs = get_state(obs)
            action = 1       

        elif self.env_name =='acrobot':

            self.gradient = np.zeros((3,6))
            self.action_probabilities = softmax(np.matmul(self.weights, obs.reshape(6, 1)))
            action = np.random.choice(3, p = self.action_probabilities.reshape(3)) 
        
        elif self.env_name == 'kbcc':
            obs = get_state(obs)
            action = 2

        self.old_state = obs
        self.action = action

        return action

    def compute_action_train(self, obs, reward, done, info):
        """
        Use this function in the train phase
        This function is called at all subsequent steps of an episode until done=True
        PARAMETERS  : 
            - observation - raw 'observation' from environment
            - reward - 'reward' obtained from previous action
            - done - True/False indicating the end of an episode
            - info -  'info' obtained from environment

        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        """

        if self.env_name =='taxi':
            next_state = obs
            old_state = self.old_state
            action = self.action

            old_value = self.q_table[old_state][action] 
            Qmax = max(self.q_table[next_state])

            # Update q-value for current state.
            self.q_table[old_state][action] = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * Qmax)
            if random.random() < self.epsilon:
                action = np.random.randint(0, self.action_size) 
            else:
                action = np.argmax(self.q_table[obs])


        elif self.env_name =='kbca':
            next_state = get_state(obs)
            obs = next_state
            old_state = self.old_state
            action = self.action

            old_value = self.q_table[old_state][action] 
            Qmax = np.max(self.q_table[next_state])

            # Update q-value for current state.
            self.q_table[old_state][action] = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * Qmax)
            
            if np.random.random() < self.epsilon:
                action = 1
            else:
                action = np.argmax(self.q_table[obs])

            self.epsilon = self.epsilon* self.decay
            #self.epsilon = max(self.epsilon, self.epsilon_bound)


        elif self.env_name=='kbcb':
            next_state = get_state(obs)
            obs = next_state
            old_state = self.old_state
            action = self.action

            old_value = self.q_table[old_state][action] 
            Qmax = np.max(self.q_table[next_state])

            # Update q-value for current state.
            self.q_table[old_state][action] = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * Qmax)
            
            if random.random() < self.epsilon:
                action = 1
            else:
                action = np.argmax(self.q_table[obs])

            self.epsilon = self.epsilon*self.decay

        elif self.env_name=='acrobot':

            if done:
                self.weights = self.weights - self.alpha * self.gradient
                action = self.action
            else:
                # choose best action so make its probab max
                self.action_probabilities = -self.action_probabilities
                self.action_probabilities[self.action] += 1
                # Linear Layer
                self.gradient += np.matmul( (self.action_probabilities * reward).reshape(3, 1), obs.reshape(1, 6))
                self.action_probabilities = softmax(np.matmul(self.weights, obs.reshape(6, 1)))
                self.action = np.random.choice(3, p=self.action_probabilities.reshape(3))
                action = self.action

        elif self.env_name == 'kbcc':
            next_state = get_state(obs)
            obs = next_state
            old_state = self.old_state
            action = self.action

            old_value = self.q_table[old_state][action] 
            Qmax = np.max(self.q_table[next_state])

            # Update q-value for current state.
            self.q_table[old_state][action] = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * Qmax)

            #action = np.random.choice([0,1,2], p = [np.exp(-x/self.tau) / np.sum([ np.exp(-i/self.tau) for i in self.q_table[obs]]) for x in self.q_table[obs]])
            
            if random.random() < self.epsilon:
                action = np.random.choice([1,2], p = [3/5, 2/5])   # [2/5, 3/5]
                if action ==1:
                    action = np.random.choice([0,1], p = [1/6, 5/6])
            else:
                action = np.argmax(self.q_table[obs])

            self.epsilon = self.epsilon* self.decay
        self.old_state = obs
        self.action = action

        return action
    
    def register_reset_test(self, obs):
        """
        Use this function in the test phase
        This function is called at the beginning of an episode. 
        PARAMETERS  : 
            - obs - raw 'observation' from environment
        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        """
        if self.env_name =='acrobot':
            action_probabilities = softmax(np.matmul(self.weights, obs))
            return np.random.choice(3, p=action_probabilities)
        elif self.env_name =='taxi':
            return np.argmax(self.q_table[obs])
        elif self.env_name =='kbca':
            return np.argmax(self.q_table[get_state(obs)])
        elif self.env_name=='kbcb':
            return np.argmax(self.q_table[get_state(obs)])
        elif self.env_name =='kbcc':
            return np.argmax(self.q_table[get_state(obs)])

        return 1

    def compute_action_test(self, obs, reward, done, info):
        """
        Use this function in the test phase
        This function is called at all subsequent steps of an episode until done=True
        PARAMETERS  : 
            - observation - raw 'observation' from environment
            - reward - 'reward' obtained from previous action
            - done - True/False indicating the end of an episode
            - info -  'info' obtained from environment

        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        """

        if self.env_name == 'acrobot':
            action_probabilities = softmax(np.matmul(self.weights, obs))
            return np.random.choice(3, p=action_probabilities)
        elif self.env_name == 'taxi':
            return np.argmax(self.q_table[obs])
        elif self.env_name == 'kbca':
            return np.argmax(self.q_table[get_state(obs)])
        elif self.env_name == 'kbcb':
            return np.argmax(self.q_table[get_state(obs)])
        elif self.env_name == 'kbcc':
            return np.argmax(self.q_table[get_state(obs)])

        return 1


