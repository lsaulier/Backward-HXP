import numpy as np
import json
import time
import os

class Agent:

    def __init__(self, name, env, lr=.2, exp_rate=1.0, decay_gamma=0.95):
        #  Version
        self.name = name
        #  Parameters for updating Q
        self.lr = lr
        self.exp_rate = exp_rate
        self.decay_gamma = decay_gamma
        #  Environment
        self.env = env
        #  Actions
        self.actions = self.env.action_space.n
        #  States
        self.states = self.env.observation_space.n
        #  Q table
        self.Q = np.zeros((self.states, self.actions))

    #  Train agent
    #  Input: number of training episodes (int)
    #  Output: None
    def train(self, nE):
        start_time = time.time()
        #  Set list of exploratory rates
        expRate_schedule = self.expRate_schedule(nE)
        print("Starting Agent's training")
        #  Training loop
        for episode in range(nE):

            #  Display the training progression
            if episode % 10000 == 0:
                print("Episodes : " + str(episode))

            #  Reset environment
            state, _ = self.env.reset()

            #  Flag to stop the episode
            done = False
            while not done:
                #  Choose an action
                self.exp_rate = expRate_schedule[episode]
                action = self.chooseAction(state)
                #  Execute it
                new_state, reward, done, _, _ = self.env.step(action)
                #  Update Q value
                self.updateQ(state, action, new_state, reward)

                state = new_state

        print("End of Agent's training !")
        # Give learning duration
        final_time_s = time.time() - start_time
        time_hour, time_minute, time_s = (final_time_s // 60) // 60, (final_time_s // 60) % 60, \
                                         final_time_s % 60
        print("Training process achieved in  : \n {} hour(s) \n {} minute(s) \n {} second(s)".format(
            time_hour, time_minute, time_s))

        return

    #  Choose an action to perform from a state
    #  Input: current state (int)
    #  Output: action (int)
    def chooseAction(self, state):
        #  Exploratory move
        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.randint(0, self.actions)
            return action
        #  Greedy move
        else:
            action = np.argmax(self.Q[state])
            return action

    #  Update a value in the Q table
    #  Input: state (int), action (int), reached state (int), obtained reward (float)
    #  Output:  None
    def updateQ(self, state, action, new_state, reward):

        # Updating rule
        self.Q[state, action] = self.Q[state, action] + self.lr * \
                                (reward + self.decay_gamma * np.max(self.Q[new_state, :]) - self.Q[state, action])

        return

    #  Predict an action from a given state
    #  Input: state (int)
    #  Output: state (int)
    def predict(self, observation):
        return np.argmax(self.Q[observation])

    #  Build a list of values of exploratory rate which decrease over episodes
    #  Input: number of episodes (int), minimum exploratory rate (float)
    #  Output: list of exploratory rate (np.array)
    def expRate_schedule(self, nE, exp_rate_min=0.05):
        x = np.arange(nE) + 1
        exp_rate_decay = exp_rate_min**(1 / nE)
        y = [max((exp_rate_decay**x[i]), exp_rate_min) for i in range(len(x))]
        return y

    #  Save the current Q table in a JSON file
    #  Input: directory path (String)
    #  Output: None
    def save(self, path):
        q_function_list = self.Q.tolist()
        with open(path + os.sep + 'Q_' + self.name, 'w') as fp:
            json.dump(q_function_list, fp)

        return

    #  Load a Q table from a JSON file
    #  Input: directory path (string)
    #  Output: None
    def load(self, path):
        with open(path + os.sep + 'Q_' + self.name, 'r') as fp:
            q_list = json.load(fp)
            self.Q = np.array(q_list)
            print("Q function loaded")

        return
