#  This Blackjack environment is based on : https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py

from typing import Optional
import gym
from gym import spaces
from itertools import product

#  Infinite deck used to draw cards
#  1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

#  Compare player and dealer's scores and deduce the reward
#  Input: hand's scores (int)
#  Output: reward (float)
def cmp(a, b):
    return float(a > b) - float(a < b)

#  Randomly draw a card
#  Input: random seed (int)
#  Output: card value (int)
def draw_card(np_random):
    return int(np_random.choice(deck))

#  Create a hand of two cards
#  Input: random seed (int)
#  Output: list of card values (int list)
def draw_hand(np_random):
    return [draw_card(np_random), draw_card(np_random)]

#  Does this hand have a usable ace?
#  Input: hand (int list)
#  Output: usable or not ace (bool)
def usable_ace(hand):
    return 1 in hand and sum(hand) + 10 <= 21

#  Return current hand total value
#  Input: hand (int list)
#  Output: total value (int)
def sum_hand(hand):
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)

#  Is this hand a bust?
#  Input: hand (int list)
#  Output: the hand is or not a bust (bool)
def is_bust(hand):
    return sum_hand(hand) > 21

#  What is the score of this hand (0 if bust)
#  Input: hand (int list)
#  Output: hand's score (int)
def score(hand):
    return 0 if is_bust(hand) else sum_hand(hand)

#  Is this hand a natural blackjack?
#  Input: hand (int list)
#  Output: natural blackjack or not (bool)
def is_natural(hand):
    return sorted(hand) == [1, 10]

class MyBlackjack(gym.Env):

    def __init__(self, behaviour=None, natural=False, sab=False):

        # Render mode
        self.state = None
        # Behaviour
        self.behaviour = behaviour
        # State/Action space according to the agent's type
        if behaviour:
            self.action_space = spaces.Discrete(10, start=1)
            # We only consider the player hand values reachable during a game and the usable ace
            self.observation_space = spaces.Tuple((spaces.Discrete(32), spaces.Discrete(2)))
        else:
            self.action_space = spaces.Discrete(2)
            self.observation_space = spaces.Tuple((spaces.Discrete(32), spaces.Discrete(10, start=1), spaces.Discrete(2)))

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural
        # Flag for full agreement with the (Sutton and Barto, 2018) definition. Overrides self.natural
        self.sab = sab

        #  Probability matrix according to the agent's type
        #  Limit of hand value is 21 since the player can't play with a higher value
        if self.behaviour:
            states = list(product(range(4, self.observation_space[0].n - 10), range(self.observation_space[1].n))) # minimal player hand value: 2 + 2
            states = [s for s in states if not(s[0] < 12 and s[1])]  # Minimal hand with a usable ace is 12, so we remove unreachable hands like (6,1,1)
            self.P = {s: {a: [] for a in range(1, self.action_space.n + 1)} for s in states}

        else:
            states = list(product(range(4, self.observation_space[0].n - 10), range(1, self.observation_space[1].n + 1), range(self.observation_space[2].n)))
            states = [s for s in states if not(s[0] < 12 and s[2])]  # Minimal hand with a usable ace is 12, so we remove unreachable hands like (6,1,1)
            self.P = {s: {a: [] for a in range(self.action_space.n)} for s in states}

        #  Choose a reward for EnvAgent, depending on the player hand's value
        #  Input: hand value (int)
        #  Output: reward (float)
        def env_reward(hand_value):
            reward_type = 1 if self.behaviour == "Favorable" else -1
            if hand_value == 21:
                return reward_type * 1
            elif hand_value > 21:
                return reward_type * -1
            #elif hand_value in [19, 20]:
            #    return reward_type * 0.5
            else:
                return 0.0

        #  Update the probability matrix
        #  Input: coordinates (int or tuple), action (int)
        #  Output: new state (int or tuple), reward (float) and end of episode (bool)
        def update_probability_matrix(state, action):
            # EnvAgent case
            if self.behaviour:
                hand, usable_ace = state
            # Agent case
            else:
                hand, dealer_hand, usable_ace = state
            # New state
            if usable_ace:  # Can reduce by 10 the hand_value if the score exceeds 21
                new_hand = hand + action
                if new_hand > 21:
                    new_hand -= 10
                    usable_ace = 0
            else:  # Can't reduce by 10 the hand_value
                new_hand = hand + action
                if action == 1 and new_hand + 10 <= 21:  # Ace case
                    new_hand += 10
                    usable_ace = 1
            # Done
            done = new_hand > 21
            # EnvAgent case
            if self.behaviour:
                new_state = (new_hand, usable_ace)
                # Reward
                reward = env_reward(new_hand)
                return new_state, reward, done
            # Agent case
            else:
                return (new_hand, dealer_hand, usable_ace), None, done  # reward is not used


        #  Fill the probability matrix to get the transition function
        if self.behaviour:
            for player_hand in range(4, self.observation_space[0].n - 10):
                for usable_ace in range(self.observation_space[1].n):
                    for a in range(1, self.action_space.n + 1):
                        # We ensure that the hand is reachable (if usable_ace, then player hand > 11
                        if not(player_hand < 12 and usable_ace):
                            s = (player_hand, usable_ace)
                            #print("state {} and action {}".format(s, a))
                            self.P[s][a].append((1.0, *update_probability_matrix(s, a)))
        else:
            for player_hand in range(4, self.observation_space[0].n - 10):
                for dealer_hand in range(1, self.observation_space[1].n + 1):
                    for usable_ace in range(self.observation_space[2].n):
                        for a in range(self.action_space.n):
                            # We ensure that the hand is reachable (if usable_ace, then player hand > 11
                            if not (player_hand < 12 and usable_ace):
                                s = (player_hand, dealer_hand, usable_ace)
                                if a:  # Hit action
                                    l = self.P[s][a]
                                    for card in set(deck):
                                        l.append((deck.count(card)/len(deck), *update_probability_matrix(s, card)))
                                else:  # Stick action
                                    self.P[s][a].append((1.0, s, None, True)) #  Terminal state (None reward since it
                                                                              #  depends on the dealer's turn)
        #print(self.P)
        """
        print("Transition function of {}".format(self.behaviour))
        if not self.behaviour:
            for s in self.P:
                print(s)
                print(self.P[s])
        """


    def step(self, action, new_card=0):
        #print("Action {} in {} env".format(action, self.behaviour))
        assert self.action_space.contains(action)
        #  eA Training
        if self.behaviour:
            #print(self.P[self.state][action])
            p, s, r, d = self.P[self.state][action][0]  # deterministic transition --> one tuple per list
            self.set_state(s)
            return s, r, d, p, {}
        else:
            # eA Training, we simply simulate a step of the agent
            if new_card:
                #if new_card == 1:
                #print('Old state {}'.format(self.state))
                #print(action)
                #print(self.P[self.state][action])
                p, s, r, d = self.P[self.state][action][new_card - 1]  # last index: from 1-10 to 0-9
                self.set_state(s)
                #if new_card == 1: print("new state: {}".format(self.state))
                return s, r, d, p, {}

            # agent training and testing
            else:
                if action:  # hit: add a card to players hand and return
                    self.player.append(draw_card(self.np_random))
                    #print("hit!")
                    if is_bust(self.player):
                        terminated = True
                        reward = -1.0
                    else:
                        terminated = False
                        reward = 0.0
                else:  # stick: play out the dealers hand, and score
                    terminated = True
                    while sum_hand(self.dealer) < 17:
                        self.dealer.append(draw_card(self.np_random))
                    reward = cmp(score(self.player), score(self.dealer))
                    if self.sab and is_natural(self.player) and not is_natural(self.dealer):
                        # Player automatically wins. Rules consistent with S&B
                        reward = 1.0
                    elif (
                        not self.sab
                        and self.natural
                        and is_natural(self.player)
                        and reward == 1.0
                    ):
                        # Natural gives extra points, but doesn't autowin. Legacy implementation
                        reward = 1.5

                return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player))

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)

        #  Distinct state according to the agent's type
        if self.behaviour:
            self.state = self._get_obs()[0]
        else:
            self.state = self._get_obs()

        return self.state, {}

    def set_state(self, state):
        self.state = state
        return

    #  Set dealer and player's hand
    #  Input: None
    def set_hands(self):
        #  Set dealer's hand
        self.dealer = [self.state[1], draw_card(self.np_random)]
        #  Set player's hand
        hand_value = self.state[0]
        self.player = []

        if self.state[2]:  # Usable ace
            hand_value -= 11
            self.player.append(1)
        while hand_value:
            if hand_value in deck:
                self.player.append(hand_value)
                hand_value -= hand_value
            else:
                highest_card = max(deck)
                self.player.append(highest_card)
                hand_value -= highest_card

        return
