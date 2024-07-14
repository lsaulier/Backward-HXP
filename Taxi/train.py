import argparse
import gymnasium as gym
from agent import Agent


if __name__ == "__main__":

    #  Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-policy', '--policy_name', default="test", help="Part of policy name", type=str, required=False)
    parser.add_argument('-ep', '--nb_episodes', default=10000, help="Number of training episodes", type=int, required=False)

    args = parser.parse_args()

    # Get arguments
    POLICY_NAME = args.policy_name
    NB_EPISODES = args.nb_episodes

    #  Env initialization
    env = gym.make("Taxi-v3").env
    #  Agent initialization
    agent = Agent(POLICY_NAME, env)
    #  Train
    agent.train(NB_EPISODES)
    #  Save Q table
    agent.save("Q-tables")
    #  Delete agent
    del agent

