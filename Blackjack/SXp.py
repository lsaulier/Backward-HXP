from copy import deepcopy


#  Infinite deck used to draw cards
#  1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

#  Compute a P scenario
#  Input: environment (MyBlackjack), the current state (int), the number of steps to look forward (int), agent (Agent)
#  and probability starting from obs (float)
#  Output: last state (int)
def P_scenario(env, obs, k, model, proba, contrastive_action):
    bust = True
    #done = False
    #  Simulate the agent's behaviour
    env_copy = deepcopy(env)
    env_copy.reset()  # useful to counter an issue (self.player/self.dealer doesn't exist)
    env_copy.set_state(obs) # only used in the case of HXp
    env_copy.set_hands()

    # Sequence loop
    for i in range(k):
        #  Action choice
        if i == 0 and contrastive_action is not None:  # Special case: first step scenario with a contrastive action
            action = contrastive_action
        else:
            action = model.predict(obs)
        if action:  # Hit action
            #  Step
            card = max(deck, key=deck.count)  # the highest probability is to pick a card with a value of 10
            p = deck.count(card) / len(deck)
            obs, _, done, _, _ = env_copy.step(action, card)
        else:  # Stick action
            p = 1.0
            done = True
            bust = False
        # Specific to HXp proba
        proba *= p
        # Check whether the scenario ends or not
        if done and i != k - 1:
            break
    #print("Last-step P-scenario: state - {} reward - {}".format(obs, reward))
    if done and not bust:
        p, d, ace = obs
        if proba:
            return (-p, d, ace), proba
        else:
            return -p, d, ace
    else:
        if proba:
            return obs, proba
        else:
            return obs

#  Compute a HE/FE-scenario, depending on the environment type
#  Input: environment (MyBlackjack), current state (int), number of steps to look forward (int), agent (Agent) and
#  environment agent (EnvAgent) and probability starting from obs (float)
#  Output: last state(int)
def E_scenario(env, obs, k, model, eA_model, proba, contrastive_action):
    bust = True
    #done = False
    #  Simulate the agent's behaviour
    env_copy = deepcopy(env)
    env_copy.reset()  # useful to counter an issue (self.player/self.dealer doesn't exist)
    env_copy.set_state(obs)  # only used in the case of HXp
    env_copy.set_hands()
    #  Sequence loop
    for i in range(k):
        #  Save obs to get reward
        last_obs = obs
        #  Actions choice
        if i == 0 and contrastive_action is not None:  # Special case: first step scenario with a contrastive action
            action = contrastive_action
        else:
            action = model.predict(obs)
        if action:
            #  Step
            eA_action = eA_model.predict((obs[0], obs[2]))  # eA state
            #  Update agent's state
            obs, _, done, _, _ = env_copy.step(action, eA_action)
            p = deck.count(eA_action) / len(deck)
        else:  # Stick action
            p = 1.0
            done = True
            bust = False
        # Specific to HXp proba
        proba *= p

        if done and i != k - 1:
            break

    if done and not bust:
        p, d, ace = obs
        if proba:
            return (-p, d, ace), proba
        else:
            return -p, d, ace
    else:
        if proba:
            return obs, proba
        else:
            return obs

#  Compute SXps from a specific state. This function only return last-states from each SXp and it's used in the context
#  of SXp.
#  Input: environment (MyBlackjack), current state (int), agent (Agent), number of steps to look forward (int),
#  environment agents (EnvAgent list) and probability starting from obs (float)
#  Output: (no return)
def SXpForHXp(env, obs, model, k, eA_agents, proba=0.0, contrastive_action=None):
    #  Bust terminal state
    if obs[0] > 21:
        if proba:
            return [(obs, proba)]
        else:
            return [obs]
    else:
        #  ------------------------ HE-scenario ----------------------------------------
        last_state_HE = E_scenario(env, obs, k, model, eA_agents[0], proba, contrastive_action)
        #  ------------------------ P-scenario ----------------------------------------
        last_state_P = P_scenario(env, obs, k, model, proba, contrastive_action)
        #  ------------------------ FE-scenario ----------------------------------------
        last_state_FE = E_scenario(env, obs, k, model, eA_agents[1], proba, contrastive_action)

        return last_state_HE, last_state_P, last_state_FE
