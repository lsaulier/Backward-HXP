import csv

from agent import Agent, EnvAgent
import os, sys
import numpy as np
import argparse
import queue
from env import MyBlackjack

# Get access to the HXP file
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from HXP import HXP
from HXP_tools import win, lose
from HXP_tools import transition, terminal, render, preprocess, get_actions, constraint, sample
from HXP_tools import valid_history

# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

#  Create the blackjack transition function
#  Input: Q table (int list list) and environment (Taxi-v3)
#  Output: the transition function (dict)
def P(Q, env):
    keys = Q.keys()
    P = {s: {a: [] for a in range(env.action_space.n)} for s in keys}
    for (player_hand, dealer_card, usable_ace) in keys:
        #  if the player has a higher hand than 21, the game is over, hence there is no transitions
        if player_hand <= 21:
            for a in range(env.action_space.n):
                tr = P[(player_hand, dealer_card, usable_ace)][a]
                #  stick action
                if not a:
                    tr.append((1.0, player_hand, dealer_card, usable_ace))
                #  hit action
                else:
                    for card in set(deck):
                        #print(card)
                        #  special case: ace card
                        if card == 1 and ((player_hand + 11) <= 21):
                            tr.append((deck.count(card)/len(deck), player_hand+11, dealer_card, usable_ace))
                        else:
                            tr.append((deck.count(card) / len(deck), player_hand + card, dealer_card, usable_ace))
    """
    for key in P:
        print("key : {}".format(key))
        for i in range(2):
            print(P[key][i])
    """
    return P


"""
HISTORIQUE A RE TESTER  - 6 9 False 1 11 9 False 1 14 9 False 1 20 9 False 0 20 9 False 
                        - 6 2 False 1 11 2 False 1 16 2 False 1 23 2 False
                        - 11 1 False 1 13 1 False 1 19 1 False 1 26 1 False
                        - 11 1 False 1 13 1 False 1 19 1 False 1 21 1 False 0 21 1 False
"""

if __name__ == "__main__":

    #  Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-policy', '--policy_name', default="player_v2", help="Part of policy name", type=str, required=False)
    parser.add_argument('-ep', '--nb_episodes', default=1, help="Number of episodes for an agent's policy test", type=int, required=False)
    parser.add_argument('-k', '--length_k', default=4, help="History's length", type=int, required=False)
    parser.add_argument('-n', '--n_important_action', default=1, help="Number of important action to display to the user", type=int, required=False)
    parser.add_argument('-csv', '--csv_filename', default="scores.csv", help="csv file to store scores", type=str, required=False)
    parser.add_argument('-HXp', '--HXp', dest="HXP", action="store_true", help="Compute HXp", required=False)
    parser.add_argument('-no_HXp', '--no_HXp', action="store_false", dest="HXP", help="Do not compute HXp", required=False)
    parser.set_defaults(HXP=True)
    parser.add_argument('-probas_HXp', '--probas_HXp', dest="probas_HXP", action="store_true", help="Compute HXp with the proba variant", required=False)
    parser.add_argument('-no_probas_HXp', '--no_probas_HXp', action="store_false", dest="probas_HXP", help="Do not compute HXp", required=False)
    parser.set_defaults(probas_HXP=True)
    parser.add_argument('-th_HXp', '--threshold_HXP', dest="threshold_HXP", action="store_true", help="Compute importance threshold for each predicate and importance type", required=False)
    parser.add_argument('-no_th_HXp', '--no_threshold_HXP', action="store_false", dest="threshold_HXP", help="Do not compute importance thresholds", required=False)
    parser.set_defaults(threshold_HXP=False)
    parser.add_argument('-pre', '--predicate', default="win", help="Predicate to verify in the history", type=str, required=False)
    parser.add_argument('-strat', '--strategy', default="exh", help="Exploration strategy for generating HXp", type=str,
                        required=False)
    parser.add_argument('-strats', '--strategies', default="", help="Exploration strategies for similarity measures",
                        type=str, required=False)
    parser.add_argument('-spec_his', '--specific_history', nargs="+", default=0, help="Express the specific history", type=tuple, required=False)
    parser.add_argument('-thresholds', '--thresholds', nargs="+", default=0, help="Provide importance thresholds", type=tuple, required=False)
    parser.add_argument('-strat_info', '--HXp_strategy_additional_info', default="", help="Additional inforamation for HXp strategy",
                        type=str, required=False)
    parser.add_argument('-select', '--backward_select', default='imp',
                        help="Method to select the important state of a sub-sequence (backward HXP)",
                        type=str, required=False)

    parser.add_argument('-find_histories', '--find_histories', dest="find_histories", action="store_true", help="Find n histories", required=False)
    parser.add_argument('-no_find_histories', '--no_find_histories', action="store_false", dest="find_histories", help="Don't look for n histories", required=False)
    parser.set_defaults(find_histories=False)

    parser.add_argument('-imp_type', '--importance_type', default="action", help="To compute HXp without user queries, choice of the type of importance to search", type=str, required=False)
    parser.add_argument('-pre_info', '--predicate_additional_info', nargs="+", default=-1, help="Specify a location",
                        type=int, required=False)

    parser.add_argument('-fixed_horizon', '--fixed_horizon', dest="fixed_horizon", action="store_true",
                        help="Utility: probability to respect the predicate at horizon k", required=False)
    parser.add_argument('-unfixed_horizon', '--unfixed_horizon', action="store_false", dest="fixed_horizon",
                        help="Utility: probability to respect the predicate at maximal horizon k", required=False)
    parser.set_defaults(fixed_horizon=True)

    args = parser.parse_args()

    # Get arguments
    POLICY_NAME = args.policy_name
    NUMBER_EPISODES = args.nb_episodes
    # Only used for HXp
    K = args.length_k
    N = args.n_important_action
    COMPUTE_HXP = args.HXP
    proba_HXP = args.probas_HXP
    STRATEGY = args.strategy
    PREDICATE = args.predicate
    CSV_FILENAME = args.csv_filename
    IMPORTANCE_TYPE = args.importance_type
    temp_history = args.specific_history
    THRESHOLD = args.threshold_HXP
    FIND_HISTORIES = args.find_histories
    STRATEGIES = args.strategies
    FIXED_HORIZON = args.fixed_horizon
    SELECT = args.backward_select

    """
    HISTORIQUE A RE TESTER  - 6 9 False 1 11 9 False 1 14 9 False 1 20 9 False 0 20 9 False 
                            - 6 2 False 1 11 2 False 1 16 2 False 1 23 2 False
                            - 11 1 False 1 13 1 False 1 18 1 False 1 26 1 False     # 19 instead of 18 (player_v1)
                            - 11 1 False 1 13 1 False 1 18 1 False 1 21 1 False 0 21 1 False
    """

    #  Fill the specific history list
    if temp_history:
        SPECIFIC_HISTORY = []
        tmp = []
        if isinstance(temp_history, list):
            for elm in temp_history:
                if elm not in ['[', ',', ' ', ']']:
                    #print(elm[0])
                    if elm[0] == 'F' or elm[0] == 'T':
                        tmp.append("".join(elm) == 'True')
                    else:
                        tmp.append(int("".join(elm)))
            for i in range(0, len(tmp), 4):
                SPECIFIC_HISTORY.append((tmp[i], tmp[i+1], tmp[i+2]))
                if i+3 != len(tmp):
                    SPECIFIC_HISTORY.append(tmp[i+3])
        print(SPECIFIC_HISTORY)
    else:
        SPECIFIC_HISTORY = False

    # Path to store actions utility in case of HXp
    if COMPUTE_HXP:
        utility_dirpath = 'Utility' + os.sep
        if not os.path.exists(utility_dirpath):
            os.mkdir(utility_dirpath)
        utility_csv = utility_dirpath + os.sep + CSV_FILENAME
    else:
        utility_csv = 'scores.csv'

    # Path to store histories
    if COMPUTE_HXP and FIND_HISTORIES:
        tmp_str = '-histories'
        hist_dirpath = 'Histories' + os.sep + str(NUMBER_EPISODES) + tmp_str

        if not os.path.exists(hist_dirpath):
            os.mkdir(hist_dirpath)
        hist_csv = hist_dirpath + os.sep + CSV_FILENAME
    else:
        hist_csv = 'trash.csv'

    # Paths to store Q tables
    agent_Q_dirpath = "Q-tables" + os.sep + "Agent"
    f_agent_Q_dirpath = "Q-tables" + os.sep + "Favorable"
    h_agent_Q_dirpath = "Q-tables" + os.sep + "Hostile"

    #  Envs initialization
    #env_F = MyBlackjack(behaviour="Favorable")
    #env_H = MyBlackjack(behaviour="Hostile")
    env = MyBlackjack()

    #  Agent initialization
    #agent_F = EnvAgent(POLICY_NAME, env_F)
    #agent_H = EnvAgent(POLICY_NAME, env_H)
    agent = Agent(POLICY_NAME, env)

    #  Load Q-table
    agent.load(agent_Q_dirpath)
    #agent_F.load(f_agent_Q_dirpath)
    #agent_H.load(h_agent_Q_dirpath)

    #eA_agents = [agent_H, agent_F]

    # Initialize HXP class
    if COMPUTE_HXP:
        predicates = {'win': win,
                      'lose': lose}
        functions = [transition, terminal, constraint, sample, preprocess, get_actions, render]
        add_info = {'select': SELECT, 'agent': agent, 'fixed_horizon': FIXED_HORIZON}
        hxp = HXP('BJ', agent, env, predicates, functions, add_info)

    #  Compute HXP from a specific history
    if SPECIFIC_HISTORY and COMPUTE_HXP:
        specific_history = queue.Queue(maxsize=K * 2 + 1)
        #print(specific_history)
        for sa in SPECIFIC_HISTORY:
            specific_history.put(sa)

        if not STRATEGIES:
            hxp.explain('no_user', specific_history, PREDICATE, [STRATEGY], N, IMPORTANCE_TYPE, utility_csv)
        else:
            hxp.explain('compare', specific_history, PREDICATE, STRATEGIES, N, IMPORTANCE_TYPE, utility_csv)
        #HXpMetric(specific_history, env, IMPORTANCE_TYPE, agent, eA_agents, CSV_FILENAME, N, PREDICATE, is_proba_HXp=proba_HXP, thresholds=thresholds, strat_info=HXP_STRATEGY_INFO)

    # Find histories
    elif FIND_HISTORIES and COMPUTE_HXP:
        nb_scenarios = NUMBER_EPISODES
        storage = []
        name, params = hxp.extract(PREDICATE)
        info = {'pred_params': params, 'env': env, 'agent': agent}

        # interaction loop
        while len(storage) != nb_scenarios:
            if not len(storage) % 10:
                print(len(storage), '/', nb_scenarios)
            history = queue.Queue(maxsize=K * 2 + 1)
            obs, _ = env.reset()
            done = False
            history.put(obs)  # initial state

            while not done:
                action = agent.predict(obs)
                if history.full():
                    history.get()
                    history.get()
                history.put(action)
                obs, reward, done, _, _ = env.step(action)
                history.put(obs)

                if valid_history(obs, name, info) > 0.8 and history.full(): # more than 80% of chances to respecting the predicate
                    data = [list(history.queue)]
                    storage.append(data)
                    if len(storage) == nb_scenarios:  # deal with specific_parts predicate (more than 1 history per episode)
                        break

        # Store infos into CSV
        with open(hist_csv, 'a') as f:
            writer = csv.writer(f)
            # First Line
            line = ['History']
            writer.writerow(line)
            # Data
            for data in storage:
                writer.writerow(data)

    # Classic testing loop
    else:
        nb_episode = NUMBER_EPISODES
        sum_reward = 0
        steps_list = []

        for episode in range(1, nb_episode + 1):
            obs, _ = env.reset()
            done = False
            score = 0
            steps = 0
            if COMPUTE_HXP:
                history = queue.Queue(maxsize=K * 2 + 1)

            while not done:
                steps += 1
                # env.render()
                #  Choose action
                action = agent.predict(obs)
                #  HXP
                if COMPUTE_HXP:
                    history.put(obs)
                    # Compute HXp
                    hxp.explain('user', history, approaches=[STRATEGY], n=N, imp_type=IMPORTANCE_TYPE, csv_file=utility_csv)
                    # Update history
                    if history.full():
                        history.get()
                        history.get()
                    history.put(action)
                #  Step
                old_obs = obs
                obs, reward, done, _, _ = env.step(action)
                score += reward

                # Store infos
                if done:
                    print("Old obs {}  with action {} leads to {} ".format(old_obs, action, obs))
                    steps_list.append(steps)

            # Last HXp
            history.put(obs)
            hxp.explain('user', history, approaches=[STRATEGY], n=N, imp_type=IMPORTANCE_TYPE, csv_file=utility_csv)

            if reward:
                sum_reward += 1

            print('----------------------------------------------')
            print('Episode:{} Score: {}'.format(episode, score))

        if nb_episode > 1:
            print('Win ratio over {} episodes : {}'.format(nb_episode, sum_reward / nb_episode))
            print('----------------------------------------------')
            print('Average of {:.0f} steps to end an episode'.format(np.mean(steps_list)))
            print('----------------------------------------------')

    #  Delete agent
    del agent

