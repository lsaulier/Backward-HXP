import os
from agent import Agent
import gymnasium as gym
import numpy as np
import argparse
from HXp import HXp, HXpMetric, print_frames
import queue

if __name__ == "__main__":

    #  Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-policy', '--policy_name', default="test", help="Part of policy name", type=str, required=False)
    parser.add_argument('-ep', '--nb_episodes', default=1, help="Number of episodes for an agent's policy test", type=int, required=False)
    parser.add_argument('-k', '--length_k', default=8, help="History's length", type=int, required=False)
    parser.add_argument('-n', '--n', default=2, help="Most important action to highlight", type=int, required=False)
    parser.add_argument('-csv', '--csv_filename', default="scores.csv", help="csv file to store scores", type=str, required=False)
    parser.add_argument('-HXp', '--HXp', dest="HXP", action="store_true", help="Compute HXp", required=False)
    parser.add_argument('-no_HXp', '--no_HXp', action="store_false", dest="HXP", help="Do not compute HXp", required=False)
    parser.set_defaults(HXP=True)
    parser.add_argument('-enc', '--encoded_state', dest="encoded_state", action="store_true", help="Provide encoded state in the history", required=False)
    parser.add_argument('-no_enc', '--no_encoded_state', action="store_false", dest="encoded_state", help="Provide encoded state in the history", required=False)
    parser.set_defaults(encoded_state=False)
    parser.add_argument('-pre', '--predicate', default="start", help="Predicate to verify in the history", type=str, required=False)
    parser.add_argument('-pre_info', '--predicate_additional_info', nargs="+", default=-1, help="Specify a location", type=int, required=False)
    parser.add_argument('-spec_his', '--specific_history', nargs="+", default=0, help="Express the specific history", type=str, required=False)
    args = parser.parse_args()

    # Get arguments
    POLICY_NAME = args.policy_name
    NUMBER_EPISODES = args.nb_episodes
    # Only used for HXp
    K = args.length_k
    HXP = args.HXP
    PREDICATE = args.predicate
    PREDICATE_INFO = args.predicate_additional_info
    CSV_FILENAME = args.csv_filename
    temp_history = args.specific_history
    N = args.n
    ENCODED_STATE = args.encoded_state

    # Path to obtain the Q table
    agent_Q_dirpath = "Q-tables"
    # Path to store actions utility in case of HXp
    if HXP:
        utility_dirpath = 'Utility' + os.sep
        if not os.path.exists(utility_dirpath):
            os.mkdir(utility_dirpath)
        utility_csv = utility_dirpath + os.sep + CSV_FILENAME
    else:
        utility_csv = 'scores.csv'

    #  Env initialization
    env = gym.make("Taxi-v3", render_mode='ansi').env

    #  Agent initialization
    agent = Agent(POLICY_NAME, env)

    #  Load Q-table
    agent.load(agent_Q_dirpath)

    #  Fill the specific history list
    if temp_history:
        SPECIFIC_HISTORY = []
        if isinstance(temp_history, list):
            if not ENCODED_STATE:
                cpt = 0
                state = []
                for elm in temp_history:
                    l_elm = list(elm)
                    #  Agent's action
                    if len(l_elm) == 1:
                        SPECIFIC_HISTORY.append(int(elm))
                    #  Agent's state
                    else:
                        tmp_elm = l_elm[:-1]
                        if tmp_elm[0] == '[':
                            tmp_elm.pop(0)
                        state.append(int(tmp_elm[0]))
                        cpt += 1
                        if cpt // 4:
                            SPECIFIC_HISTORY.append(env.encode(*[f for f in state]))
                            state = []
                            cpt = 0
            else:
                for elm in temp_history:
                    SPECIFIC_HISTORY.append(int(elm))
    else:
        SPECIFIC_HISTORY = False

    #  Compute HXp from a specific history
    if HXP and SPECIFIC_HISTORY:
        #  History to test :
        specific_history = queue.Queue(maxsize=K * 2 + 1)
        for sa in SPECIFIC_HISTORY:
            specific_history.put(sa)
        # Compute HXp
        HXpMetric(specific_history, env, agent, utility_csv, PREDICATE, property_info=PREDICATE_INFO, n=N)
    #  Classic test of the agent's policy
    else:
        frames = []
        sum_reward = 0
        steps_list = []
        nb_episode = NUMBER_EPISODES
        # Testing loop
        for episode in range(1, nb_episode + 1):
            obs, _ = env.reset()
            done = False
            score = 0
            steps = 0
            if HXP:
                history = queue.Queue(maxsize=K*2+1)

            while not done:
                steps += 1
                #env.render()
                #  Choose action
                action = agent.predict(obs)
                #  HXp
                if HXP:
                    history.put(obs)
                    # Compute HXp
                    HXp(history, env, agent, n=N, csv_filename=utility_csv)
                    # Update history
                    if history.full():
                        history.get()
                        history.get()
                    history.put(action)
                #  Step
                old_obs = obs
                obs, reward, done, _, _ = env.step(action)
                #  Render or store for an animation
                if nb_episode == 1 and not HXP:
                    frames.append({
                        'frame': env.render(),
                        'state': obs,
                        'action': action,
                        'reward': reward})
                score += reward
                # Store infos
                if done:
                    print("Old obs {} : {} \n with action {} leads to {} ".format(old_obs, list(env.decode(old_obs)), action, list(env.decode(obs))))
                    steps_list.append(steps)

            # Last render and HXp
            #env.render()
            if HXP:
                history.put(obs)
                HXp(history, env, agent, n=N, csv_filename=utility_csv)

            sum_reward += score
            print('----------------------------------------------')
            print('Episode:{} Score: {}'.format(episode, score))

        if nb_episode > 1:
            print('Score average over {} episodes : {}'.format(nb_episode, sum_reward / nb_episode))
            print('----------------------------------------------')
            print('Average of {:.0f} steps to end an episode'.format(np.mean(steps_list)))
            print('----------------------------------------------')
        else:
            print_frames(frames)

    #  Delete agent
    del agent

