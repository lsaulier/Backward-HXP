import numpy as np
import json
import time
import os
from collections import defaultdict # https://www.kaggle.com/code/hamishdickson/blackjack-with-reinforcement-learning

#  eA stands for environment agent in the whole script
class Agent:

    def __init__(self, name, env, lr=.2, exp_rate=1.0, decay_gamma=0.95, model_env_eA=None):
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
        #  Q table
        self.Q = defaultdict(lambda: np.zeros(self.actions))
        #  Initialise couple list of model,env of the eA's
        if model_env_eA is not None:
            self.model_env_eA = model_env_eA

    #  Train agent
    #  Input: number of training episodes (int)
    #  Output: None
    def train(self, nE):
        start_time = time.time()
        #  Set list of exploratory rates
        expRate_schedule = self.expRate_schedule(nE-((30/100)*nE))
        print("Starting Agent's training")
        #  Training loop
        for episode in range(nE):

            #  Display the training progression
            if episode % 20000 == 0:
                print("Episodes : {} --- Win ratio over 1000 episodes: {} -- length of Q {}".format(episode, self.evaluate(episodes=1000), len(self.Q)))
                print(self.Q[(21,10,True)])

            #  Reset environment
            state = self.env.reset()

            #  Flag to stop the episode
            done = False
            while not done:
                #  Choose an action
                self.exp_rate = expRate_schedule[episode] if episode < nE-((30/100)*nE) else 0.01
                #print(self.exp_rate)
                action = self.chooseAction(state)
                #  Execute it
                new_state, reward, done, _ = self.env.step(action)
                #print(reward)
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

    #  Train the model
    #  Input: number of training episode (int)
    #  Output: None
    def correlatedTrain(self, nE):
        start_time = time.time()
        #  Set list of exploratory rates
        expRate_schedule = self.expRate_schedule(nE)

        # ------------------------------- AGENT TRAINING -----------------------------------
        print("Starting Agent's training")
        # Training loop
        for episode in range(nE):

            #  Flag to stop the episode
            done = False
            #  Reset environment
            current_state, _ = self.env.reset()
            while not done:
                #  Choose an action
                self.exp_rate = expRate_schedule[episode]
                action = self.chooseAction(current_state)
                #  Execute it
                new_state, reward, done, _, _ = self.env.step(action)
                #  Update Q value
                self.updateQ(current_state, action, new_state, reward)
                #  Useful to keep for updating Q
                current_state = new_state

        print("End of Agent's training !")
        final_time_s = time.time() - start_time
        time_hour, time_minute, time_s = (final_time_s // 60) // 60, (final_time_s // 60) % 60, \
                                         final_time_s % 60
        print("Training process achieved in  : \n {} hour(s) \n {} minute(s) \n {} second(s)".format(
            time_hour, time_minute, time_s))

        # ------------------------------- HOSTILE AGENT TRAINING -----------------------------------
        print("Starting Hostile Agent's training")
        start_time = time.time()
        hostile_model, hostile_env = self.model_env_eA[1][0], self.model_env_eA[1][1]
        # Training loop
        for episode in range(nE):

            #  Flag to stop the episode
            done = False
            #  Reset environment
            current_state, _ = self.env.reset() #hostile_env.reset()
            hostile_env.set_state((current_state[0], current_state[2]))
            while not done:
                #  Choose agent's action
                agent_action = self.predict(current_state)
                #  Hit action (the eA learns only when the agent performs this action)
                if agent_action:
                    #print("Agent action: {}".format(agent_action))
                    # Env agent state
                    eA_state = hostile_env.state
                    #  Choose eA's action
                    hostile_model.exp_rate = expRate_schedule[episode]
                    eA_action = hostile_model.chooseAction(eA_state)
                    #print("eA action: {}".format(eA_action))
                    #  Execute it
                    new_eA_state, reward, done, _, _ = hostile_env.step(eA_action)
                    #  Update agent's state
                    current_state, _, _, _, _ = self.env.step(agent_action, eA_action)
                    #  Update Q value
                    hostile_model.updateQ(eA_state, eA_action, new_eA_state, reward)
                    #print("current agent state: {}".format(current_state))
                #  Stick action: end of an episode
                else:
                    done = True

        print("End of Hostile Agent's training !")
        final_time_s = time.time() - start_time
        time_hour, time_minute, time_s = (final_time_s // 60) // 60, (final_time_s // 60) % 60, \
                                         final_time_s % 60
        print("Training process achieved in  : \n {} hour(s) \n {} minute(s) \n {} second(s)".format(
            time_hour, time_minute, time_s))
        # ------------------------------- FAVORABLE AGENT TRAINING -----------------------------------
        print("Starting Favorable Agent's training")
        start_time = time.time()
        favorable_model, favorable_env = self.model_env_eA[0][0], self.model_env_eA[0][1]
        # Training loop
        for episode in range(nE):

            #  Flag to stop the episode
            done = False
            #  Reset environment
            current_state, _ = self.env.reset() #hostile_env.reset()
            favorable_env.set_state((current_state[0], current_state[2]))
            while not done:
                #  Choose agent's action
                agent_action = self.predict(current_state)
                #  Hit action (the eA learns only when the agent performs this action)
                if agent_action:
                    # Env agent state
                    eA_state = favorable_env.state
                    #  Choose eA's action
                    favorable_model.exp_rate = expRate_schedule[episode]
                    eA_action = favorable_model.chooseAction(eA_state)
                    #  Execute it
                    new_eA_state, reward, done, _, _ = favorable_env.step(eA_action)
                    #  Update agent's state
                    current_state, _, _, _, _ = self.env.step(agent_action, eA_action)
                    #  Update Q value
                    favorable_model.updateQ(eA_state, eA_action, new_eA_state, reward)
                #  Stick action: end of an episode
                else:
                    done = True

        print("End of Favorable Agent's training !")
        final_time_s = time.time() - start_time
        time_hour, time_minute, time_s = (final_time_s // 60) // 60, (final_time_s // 60) % 60, \
                                         final_time_s % 60
        print("Training process achieved in  : \n {} hour(s) \n {} minute(s) \n {} second(s)".format(
            time_hour, time_minute, time_s))

        return

    #  Evaluate how good the policy is
    #  Input: number of episodes (int)
    #  Output: win ratio (float)
    def evaluate(self, episodes=10000):
        wins = 0
        for _ in range(episodes):
            state = self.env.reset()
            #print("state {}".format(state))

            done = False
            while not done:
                action = self.predict(state)

                state, reward, done, _ = self.env.step(action=action)

            if reward > 0:
                wins += 1

        return wins / episodes

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
    #  Input: state (int), action (int), reached state (int), reward obtained (float)
    #  Output:  None
    def updateQ(self, state, action, new_state, reward):
        #print("Update state {} by the next state {}".format(state, new_state))
        # Updating rule
        self.Q[state][action] = self.Q[state][action] + self.lr * \
                                (reward + self.decay_gamma * np.max(self.Q[new_state][:]) - self.Q[state][action])

        return

    #  Predict an action from a given state
    #  Input: state (int)
    #  Output: state (int)
    def predict(self, observation):
        #print(observation)
        #print(self.Q[observation])
        #print(np.argmax(self.Q[observation]))
        return np.argmax(self.Q[observation])

    #  Build a list of values of exploratory rate which decrease over episodes
    #  Input: number of episodes (int), minimum exploratory rate (float)
    #  Output: list of exploratory rate (np.array)
    def expRate_schedule(self, nE, exp_rate_min=0.05):
        x = np.arange(nE) + 1
        exp_rate_decay = exp_rate_min**(1 / nE)
        #print(exp_rate_decay)
        y = [max((exp_rate_decay**x[i]), exp_rate_min) for i in range(len(x))]
        #print(len(y))
        #print(y[250000])
        return y

    #  Save the current Q table in a JSON file
    #  Input: directory path (String)
    #  Output: None
    def save(self, path):
        keys, values = list(self.Q.keys()), list(self.Q.values())
        keys = [str(k) for k in keys]
        values = [list(v) for v in values]
        q_function_list = dict(zip(keys, values))
        with open(path + os.sep + 'Q_' + self.name, 'w') as fp:
            json.dump(q_function_list, fp)

        return

    #  Load a Q table from a JSON file
    #  Input: directory path (string)
    #  Output: None
    def load(self, path):
        absolute_dir_path = os.path.dirname(__file__)
        with open(absolute_dir_path + os.sep + path + os.sep + 'Q_' + self.name, 'r') as fp:
            q_dict = dict(json.load(fp))
            keys = q_dict.keys()
            values = [np.array(v) for v in q_dict.values()]
            tuple_keys = []
            for k in keys:
                #print(k)
                tmp_str_list = k[1:-1].split(', ')
                usable_ace = True if tmp_str_list[2] == 'True' else False
                tuple_keys.append((int(tmp_str_list[0]), int(tmp_str_list[1]), usable_ace))
                #print(tuple_keys)
                #print(tmp_str_list)
            q_dict = dict(zip(tuple_keys, values))
            self.Q = q_dict
            print("Q function loaded")

        return

    #  Compute the state importance
    #  Input: state (int)
    #  Output: state importance (float)
    def stateImportance(self, observation):
        return max(self.Q[observation]) - min(self.Q[observation])

class EnvAgent:

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
        #  Q table
        self.Q = defaultdict(lambda: np.zeros(self.actions))

    #  Choose an action to perform from a state
    #  Input: current state (int)
    #  Output: action (int)
    def chooseAction(self, state):
        #  Exploratory move
        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.randint(1, self.actions + 1)
            return action
        #  Greedy move
        else:
            action = np.argmax(self.Q[state]) + 1  # from 0-9 to 1-10
            return action

    #  Update a value in the Q table
    #  Input: state (int), action (int), reached state (int), reward obtained (float)
    #  Output:  None
    def updateQ(self, state, action, new_state, reward):
        #print("Update state {} by the next state {} doing {} and receive {}".format(state, new_state, action, reward))
        # Updating rule (action-1 is to change range 1-10 to 0-9 to fill the Q-table)
        self.Q[state][action - 1] = self.Q[state][action - 1] + self.lr * \
                                (reward + self.decay_gamma * np.max(self.Q[new_state][:]) - self.Q[state][action - 1])

        return

    #  Predict an action from a given state
    #  Input: state (int)
    #  Output: state (int)
    def predict(self, observation):
        #print(observation)
        #print(self.Q[observation])
        #print(np.argmax(self.Q[observation]))
        return np.argmax(self.Q[observation]) + 1  # from 0-9 to 1-10

    #  Save the current Q table in a JSON file
    #  Input: directory path (String)
    #  Output: None
    def save(self, path):

        #print(self.Q)
        keys, values = list(self.Q.keys()), list(self.Q.values())
        keys = [str(k) for k in keys]
        values = [list(v) for v in values]
        #print("New keys: {}".format(keys))
        #print("New values: {}".format(values))
        q_function_list = dict(zip(keys, values))
        #print(q_function_list)

        with open(path + os.sep + 'Q_' + self.name, 'w') as fp:
            json.dump(q_function_list, fp)

        return

    #  Load a Q table from a JSON file
    #  Input: directory path (string)
    #  Output: None
    def load(self, path):
        with open(path + os.sep + 'Q_' + self.name, 'r') as fp:
            q_dict = dict(json.load(fp))
            keys = q_dict.keys()
            values = [np.array(v) for v in q_dict.values()]
            tuple_keys = []
            for k in keys:
                tmp_str_list = k[1:-1].split(', ')
                usable_ace = True if tmp_str_list[1] in ['True', '1'] else False
                tuple_keys.append((int(tmp_str_list[0]), usable_ace))
                #print(tuple_keys)
                #print(tmp_str_list)
            q_dict = dict(zip(tuple_keys, values))
            self.Q = q_dict
            print("Q function loaded")
            #print(self.Q)

        return