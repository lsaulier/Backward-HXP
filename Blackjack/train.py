import argparse
import gym
from agent import Agent, EnvAgent
from env import MyBlackjack
import os


if __name__ == "__main__":

    #  Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-policy', '--policy_name', default="test", help="Common part of policies name", type=str, required=False)
    parser.add_argument('-ep', '--nb_episodes', default=100000, help="Number of training episodes", type=int, required=False)

    args = parser.parse_args()

    # Get arguments
    POLICY_NAME = args.policy_name
    NB_EPISODES = args.nb_episodes

    # Paths to store Q tables
    agent_Q_dirpath = "Q-tables" + os.sep + "Agent"
    f_agent_Q_dirpath = "Q-tables" + os.sep + "Favorable"
    h_agent_Q_dirpath = "Q-tables" + os.sep + "Hostile"

    #  Envs initialization
    env_F = MyBlackjack(behaviour="Favorable")
    env_H = MyBlackjack(behaviour="Hostile")
    env = MyBlackjack() # gym.make("Blackjack-v0")

    #  Agents initialization
    agent_F = EnvAgent(POLICY_NAME, env_F)
    agent_H = EnvAgent(POLICY_NAME, env_H)
    agent = Agent(POLICY_NAME, env, lr=0.3, decay_gamma=0.92, model_env_eA=[[agent_F, env_F], [agent_H, env_H]])

    #  Train
    agent.correlatedTrain(NB_EPISODES)

    #  Save Q table
    agent.save(agent_Q_dirpath)
    agent_F.save(f_agent_Q_dirpath)
    agent_H.save(h_agent_Q_dirpath)

    #  Delete agent
    del agent
    del agent_F
    del agent_H

