import random
from copy import deepcopy
from statistics import mean

#  Infinite deck used to draw cards
#  1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

### HXP functions ###

#  Get from a state-action couple, the entire/part of the transitions available, i.e. the new states associated with
#  their probabilities
#  Input: agent's state (int (list)), action (int), environment (MyBlackjack), importance score method (str), number of
#  exhaustive/deterministic steps (int), additional information (dictionary), importance type (str)
#  Output: list of transition-probability couples (couple list)
def transition(s, a, env, approx_mode, exh_steps=0, det_tr=0, add_info=None, imp_type=None):
    transitions = [(t[0], t[1]) for t in deepcopy(env.P[s][a])]

    # Look all possible transitions from s
    if approx_mode == 'none' or exh_steps:
        return transitions

    else:
        # Look for the most probable transition
        if approx_mode == 'last':
            # Specific case: equiprobable transitions
            return extract_transitions(1, transitions, approx_mode)
        # Select the 'det' most probable transition(s)
        else:
            return extract_transitions(det_tr, transitions, approx_mode)

#  Check whether the state is terminal or not
#  Input: state (int), environment (MyBlackjack), additional information (dict)
#  Output: (bool)
def terminal(s, env, add_info):
    #if not add_info['agent'].predict(s):
    #    print('terminal: {}'.format(s))
    return s[0] > 21 or not add_info['agent'].predict(s) or not add_info['S_succ'] and not add_info['opp_action']

#  Compute the argmax of an array
#  Input: an array (list)
#  Output: index of the maximum value (int)
def argmax(array):
    array = list(array)
    return array.index(max(array))


#  Extract n most probable transitions
#  Input: number of transition to extract (int), transitions (tuple list), importance score method (str)
#  Output: most probable transition(s) (tuple list)
def extract_transitions(n, transitions, approx_mode):
    most_probable = []

    while n != len(most_probable):
        probas = [t[0] for t in transitions]
        max_pr, idx_max_pr = max(probas), argmax(probas)
        tmp_cpt = probas.count(max_pr)
        # Only one transition is the current most probable one
        if tmp_cpt == 1:
            temp_t = list(transitions[idx_max_pr])
            most_probable.append(temp_t)
            transitions.remove(transitions[idx_max_pr])

        else:
            # There are more transitions than wanted (random pick)
            if tmp_cpt > n - len(most_probable):
                random_tr = random.choice([t for t in transitions if t[0] == max_pr])
                #print('random_tr: {}'.format(random_tr))
                temp_random_tr = list(random_tr)
                most_probable.append(temp_random_tr)
                transitions.remove(random_tr)

            else:
                tmp_list = []
                for t in transitions:
                    if t[0] == max_pr:
                        temp_t = list(t)
                        most_probable.append(temp_t)
                        tmp_list.append(t)
                for t in tmp_list:
                    transitions.remove(t)

    # Probability distribution
    sum_pr = sum([p for p, s in most_probable])
    if sum_pr != 1.0:
        delta = 1.0 - sum_pr
        add_p = delta / len(most_probable)
        for elm in most_probable:
            elm[0] += add_p
    return most_probable

#  Multiple-tasks function to update some data during the HXP process
#  Input: environment (MyBlackjack), agent (Agent), location of the modification in the HXP process (str),
#  state-action list (int (list)-int list), additional information (dictionary)
#  Output: variable
def preprocess(env, agent, location, s_a_list=None, add_info=None):
    # store contrastive action
    if location == 'hxp':
        hist_action = s_a_list[1]
        opp_action = 0 if hist_action else 1
        add_info['opp_action'] = opp_action
    # bool to track which scenarios are generated
    elif location == 'impScore':
        if add_info.get('S_succ') is None or not add_info['S_succ']:
            add_info['S_succ'] = True
        else:
            add_info['S_succ'] = False

    return env, agent

#  Check whether an importance score can be computed or not
#  Input: action (int), importance type (str), additional information (dictionary)
#  Output: (bool)
def constraint(action, imp_type, add_info):
    if imp_type == 'transition':
        return action == 0
    return False

#  Get available actions from a state (available actions are similar no matter the state)
#  Input: state (int list), environment (MyBlackjack)
#  Output: action list (int list)
def get_actions(s, env):
    return [j for j in range(env.action_space.n)]

#  Render the most important action(s) / transition(s)
#  Input: state-action list to display (int (list)-int  list), environment (MyBlackjack), agent (Agent),
#  importance type (str), runtime (float), additional information (dictionary)
#  Output: None
def render(hxp, env, agent, imp_type, runtime, add_info):
    # Render
    for s_a_list, i in hxp:
        print("Timestep {}".format(i))
        if imp_type == 'action':
            print("From state {}, action {}".format(s_a_list[0], s_a_list[1]))
        else:
            print(s_a_list)
            print("Transition: \n s: {} \n a: {} \n s': {}".format(s_a_list[0], s_a_list[1], s_a_list[2]))
    # Runtime
    print("-------------------------------------------")
    print("Explanation achieved in: {} second(s)".format(runtime))
    print("-------------------------------------------")
    return

### Backward HXP functions ###

#  Sample n states from the state space. These states match the set of fixed features of v.
#  This is an exhaustive sampling.
#  Input: environment (MyBlackjack), partial state (int (list)), index of feature to remove from v (int),
#  number of samples to generate (int), additional information (dict)
#  Output: list of states (int (list) list)
def sample(env, v, i, n, add_info=None):
    return

### Predicates ###

#  Check whether the agent wins or not
#  Input: state (int (list)), additional information (dictionary)
#  Output: (bool), probability to win (float)
def win(s, info):
    player_hand, visible_dealer_card, usable_ace = s
    #print('state: {} - proba to win: {}'.format(s, dealerPartialHand(player_hand, visible_dealer_card, 'win')))
    return dealerPartialHand(player_hand, visible_dealer_card, 'win')

#  Check whether the agent loses or not
#  Input: state (int list), additional information (dictionary)
#  Output: (bool)
def lose(s, info):
    player_hand, visible_dealer_card, usable_ace = s
    #print('state: {} - proba to lose: {}'.format(s, dealerPartialHand(player_hand, visible_dealer_card, 'lose')))
    return dealerPartialHand(player_hand, visible_dealer_card, 'lose')

#  Given a partial dealer hand, compute the different probabilities for the player to win/loose according to each
#  possible dealer hand
#  Input: player hand value (int), visible dealer card value (int) and predicate to verify (str)
#  Output: mean of win/loose probability given a state (float)
def dealerPartialHand(player_hand, visible_dealer_card, predicate):
    # Quick output
    if player_hand > 21:
        return 0.0 if predicate == 'win' else 1.0

    reachable_hands = []
    # Specific case, dealer's visible card is an ace
    if visible_dealer_card == 1:
        current_hand = visible_dealer_card + 10
        usable_ace = True
    else:
        current_hand = visible_dealer_card
        usable_ace = False
    # Produce each reachable hand from each possible dealer's cards pair
    for card in set(deck):
        if card == 1:
            if usable_ace:  # Can't have two usable aces
                hand_value = current_hand + card
                usable_ace_tmp = False
            else:
                hand_value = current_hand + card + 10
                usable_ace_tmp = True
        else:
            hand_value = current_hand + card
            usable_ace_tmp = False
        #print("hand value: {}, visible card: {}".format(hand_value, visible_dealer_card))
        #print('Reachable hands from dealers value {} usable ace {}'.format(hand_value, usable_ace))
        reachable_hands.append(dealerFullHand(hand_value, usable_ace or usable_ace_tmp))

    # Count win/loose hands from each reachable hands
    if predicate == 'win':
        win_hands_ratio = []
        for hands in reachable_hands:
            win_hands = len([h for h in hands if player_hand > h or h > 21])  # win condition for the player
            total_hands = len(hands)
            win_hands_ratio.append(win_hands / total_hands)
        # print('player_hand:{}'.format(player_hand))
        #print("partial hand from couple {} --- {} add {}".format(player_hand, visible_dealer_card, win_hands_ratio))
        return mean(win_hands_ratio)
    else:
        loose_hands_ratio = []
        for hands in reachable_hands:
            loose_hands = len([h for h in hands if player_hand < h <= 21])
            total_hands = len(hands)
            loose_hands_ratio.append(loose_hands / total_hands)
        #print("partial hand from couple {} --- {}  add {}".format(player_hand, visible_dealer_card, loose_hands_ratio))
        return mean(loose_hands_ratio)

#  Produce all possible hands for the dealer given a starting pair of cards
#  Input: dealer hand value (int) and usable ace's presence (bool)
#  Output: list of all possible hands (int list)
def dealerFullHand(hand_value, usable_ace):
    draw_hands = [(hand_value, usable_ace)] if hand_value < 17 else []  # draw a card from these hands
    other_hands = [(hand_value, usable_ace)] if hand_value >= 17 else []  # stick
    # print('Starting draw_hands: {} and other_hands:{}'.format(draw_hands, other_hands))
    #  Produce each reachable hand from a card pair
    while len(draw_hands):
        for hand, uace in draw_hands:
            other_hands.extend(nextSumDrawCard(hand, uace))  # concatenate
        draw_hands = [(h, u) for (h, u) in other_hands if h < 17]
        # print('draw_hands: {}'.format(draw_hands))
        other_hands = [(h, u) for (h, u) in other_hands if h >= 17]
        # print('other_hands: {}'.format(other_hands))
    #print("Last draw_hands: {} and other_hands:{}".format(draw_hands, other_hands))
    return [h for (h, u) in other_hands]

#  Provide reachable hands by drawing one card
#  Input: current dealer hand value (int) and usable ace's presence (bool)
#  Output: list of reachable hands by drawing a card (int list)
def nextSumDrawCard(current_hand, usable_ace):
    reachable_hands = []
    #print("reachable hands from {} with usable ace {}".format(current_hand, usable_ace))
    for card in set(deck):
        ua_tmp = usable_ace
        if card == 1:  # Ace
            if usable_ace:
                hand_value = current_hand + card
            else:
                if current_hand + card + 10 <= 21:
                    hand_value = current_hand + card + 10
                    ua_tmp = True
                else:
                    hand_value = current_hand + card

        else:  # Other cards
            hand_value = current_hand + card
        # Specific case: remove usable ace additional value
        if hand_value > 21 and usable_ace:
            hand_value -= 10
            ua_tmp = False
        reachable_hands.append((hand_value, ua_tmp))
    #print(reachable_hands)
    return reachable_hands

### Find histories for a specific predicate ###

#  Verify if the last state from a proposed history respects a predicate
#  Input: state (int (list)), predicate (str), additional information (int)
#  Output: probability to respect or not of the predicate (float)
def valid_history(s, predicate, info):
    if predicate == 'win':
        return win(s, info)
    elif predicate == 'lose':
        return lose(s, info)
