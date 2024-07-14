import csv
from copy import deepcopy
import numpy as np
from IPython.display import clear_output
from time import sleep

#  Main function which manages user queries about action importance
#  Input: the history (int list), environment (Taxi-v3), agent (Agent), number of action ot highlight (int),
#  and CSV file to store scores (String)
#  Output: None
def HXp(history, env, agent=None, n=2, csv_filename=""):
    answer = False
    good_answers = ["yes", "y", "Y"]
    history = list(history.queue)
    print("History : {}".format(history))
    while not answer:

        question = "Do you want an HXp?"
        action_HXp = input(question)

        # Provide an action HXp
        if action_HXp in good_answers:
            env_copy = deepcopy(env)
            property_question = "Which predicate do you want to verify? (start/destination/location) \n"
            property = input(property_question)
            if property == 'start':
                HXp_actions = HXp_action(history, env_copy, agent, n=n, csv_filename=csv_filename)
            elif property == 'destination':
                HXp_actions = HXp_action(history, env_copy, agent, property, n=n, csv_filename=csv_filename)
            elif property == 'location':
                taxi_coords_question = "Specify the location: "
                taxi_coords = input(taxi_coords_question)
                taxi_coords = int(taxi_coords[0])-1, int(taxi_coords[2])-1
                HXp_actions = HXp_action(history, env_copy, agent, property, property_info=taxi_coords, n=n
                                         , csv_filename=csv_filename)

            else:
                print('None property was selected!')
                HXp_actions = 0
            # Render important actions
            # if HXp_actions:
                # render_actionImportance(HXp_actions, env_copy)

        answer = True

    return


#  Similar to the main function without the user interaction. All strategies are tested
#  Input: the history (int list), environment (Taxi-v3), the agent (Agent), CSV file to store scores (String),
#  predicate to verify (String), additional information for the predicate i.e. a specific state (int) and number of
#  action to highlight (int)
#  Output: None
def HXpMetric(history, env, agent, csv_filename, property, property_info=None, n=2):

    history = list(history.queue)
    print("History : {}".format(history))
    env_copy = deepcopy(env)
    env_copy.reset()
    # ------------------------------- Compute Action HXp ------------------------------------
    HXp_actions = HXp_action(history, env_copy, agent, property, property_info=property_info, n=n
                             , csv_filename=csv_filename)
    # Render important actions
    # print("Rendering of important actions using the pi strategy")
    # render_actionImportance(HXp_actions, env_copy)

    return

#  Measure action importance of each action in the history
#  Input: the history (int list), environment (Taxi-v3), agent (Agent), predicate to verify (String), additional
#  information for the predicate i.e. a specific state (int), number of action to highlight (int) and CSV file to
#  store scores (String)
#  Output: list of important actions, states and time-steps associated (int list list)
def HXp_action(H, env, agent=None, property='start', property_info=None, n=2, csv_filename=""):
    HXp_actions = []
    score_list = []
    firstLineCsvTable(property, property_info, csv_filename)  # head of table in CSV file
    fixed_k = ((len(H) - 1) // 2) - 1  # ((len(H) - i - 1) // 2) - 1
    print(fixed_k)
    for i in range(0, len(H)-1, 2):
        s, a = H[i], H[i+1]
        actions = [j for j in range(env.action_space.n)]
        score = actionImportance(s, a, env, actions, fixed_k, agent, property, property_info, csv_filename, H[-1])
        score_list.append([score, s, a, i // 2])

    # Rank the scores to select the n most important actions
    print("Actual score list : {}".format(score_list))
    tmp_list = [elm[0] for elm in score_list]
    print("Scores : {}".format(tmp_list))
    if ((len(H)-1) // 2) <= n:
        n = 1
    top_idx = np.sort(np.argpartition(np.array(tmp_list), -n)[-n:])
    print("Top {} indexes : {}".format(n, top_idx))
    for i in range(n):
        HXp_actions.append(score_list[top_idx[i]][1:])
    print("HXp action list: {}".format(HXp_actions))

    return HXp_actions


#  Define the importance of an action a from a state s
#  Input: starting state and associated action (int), environment (Taxi-v3), available actions from s (int list),
#  trajectories length (int), agent (Agent), predicate to verify (String), additional information for
#  the predicate i.e. a specific state (int) the agent (Agent), CSV file to store scores (String) and the last history
#  state which is used only for the location predicate (int)
#  Output: importance of a from s (bool)
def actionImportance(s, a, env, actions, k, agent=None, property='start',
                     property_info=None, csv_filename="", last_history_state=None):
    # Get two initial sets
    Ac_a, Ac_nota = get_Ac(s, a, actions, env)
    print("Reachable states from state {} by doing action {}: {}".format(s, a, decodeStateList(env, Ac_a)))
    print("Reachable states from other actions : {}".format(decodeStateList(env, Ac_nota)))
    # Get two sets representing the final states
    Sf_a = succ(Ac_a, k, env, agent) if k else Ac_a
    Sf_nota = succ(Ac_nota, k, env, agent) if k else Ac_nota
    # Count in both sets the number of states that respect the property
    return checkPredicate([s, a], Sf_a, Sf_nota, env, property, property_info, csv_filename, last_history_state)


#  Compute action utilities and use them to compute the action importance of a
#  Input: state-action list (int list), final states of scenarios starting from s for each action (int list or int list
#  list), environment (Taxi-v3), predicate to verify (String) additional information for the predicate i.e. a specific
#  state (int), CSV file to store scores (String) strategy to organize CSV file (String), and the last history state
#  which is used only for the location predicate (int)
#  Output: action importance (float)
def checkPredicate(s_a_list, Sf, Sf_not, env, property='start', property_info=None, csv_filename="",
                   last_history_state=None):
    s, a = s_a_list
    decoded_s = list(env.decode(s))
    # Important action if the probability of reaching the pickup state is higher when doing action a.
    # This predicate can be tested only when the taxi is going to pick up the passenger, for the second part of the
    # objective we look at the destination predicate
    if property == 'start':
        Sf = decodeStateList(env, Sf)
        Sf_not = decodeStateList(env, Sf_not)
        print("Sf : {}".format(Sf))
        print("Sf_not : {}".format(Sf_not))
        Sf_pickup = sum([1 for s in Sf if s[2] == 4 or s[2] == s[3]])  # Passenger in taxi or at its destination
        tmp_counter = []
        tmp_len = []
        for sublist in Sf_not:
            tmp_counter.append(sum([1 for s in sublist if s[2] == 4 or s[2] == s[3]]))
            tmp_len.append(len(sublist))
        probas = [tmp_counter[i] / tmp_len[i] for i in range(len(tmp_counter))]
        best = argmax(probas)
        Sf_not_best = tmp_counter[best]
        Sf_not_average = sum(probas) / len(probas)

        return storeInfoCsv(csv_filename, [decoded_s, a], [Sf_pickup, Sf_not_best, Sf_not_average, tmp_counter])
    # Important action if the probability of reaching the drop-off state is higher when doing action a.
    # This predicate can be tested only when the taxi is going to drop-off the passenger.
    elif property == 'destination':

        x_dest, y_dest = to_coordinates(decoded_s[3])
        dropoff_state = env.encode(x_dest, y_dest, decoded_s[3], decoded_s[3])
        print("Sf : {}".format(decodeStateList(env, Sf)))
        print("Sf_not : {}".format(decodeStateList(env, Sf_not)))
        Sf_dropoff = Sf.count(dropoff_state)
        tmp_counter = []
        tmp_len = []
        for sublist in Sf_not:
            tmp_counter.append(sublist.count(dropoff_state))
            tmp_len.append(len(sublist))
        probas = [tmp_counter[i] / tmp_len[i] for i in range(len(tmp_counter))]
        best = argmax(probas)
        Sf_not_best = tmp_counter[best]
        Sf_not_average = sum(probas) / len(probas)

        return storeInfoCsv(csv_filename, [decoded_s, a], [Sf_dropoff, Sf_not_best, Sf_not_average, tmp_counter])

    # Important action if the probability of reaching a specific taxi's location is higher when doing action a.
    # This predicate can be tested in each case, except when location to check isn't the last history's location and
    # there is a change of passenger index in the state to search.
    elif property == 'location':
        taxi_coords = property_info
        print(taxi_coords)
        last_history_state_dcdd = list(env.decode(last_history_state))
        print('last history state: {}'.format(last_history_state_dcdd))
        specific_state = env.encode(taxi_coords[0], taxi_coords[1], last_history_state_dcdd[2], last_history_state_dcdd[3])
        print("Sf : {}".format(decodeStateList(env, Sf)))
        print("Sf_not : {}".format(decodeStateList(env, Sf_not)))
        Sf_spec = Sf.count(specific_state)
        tmp_counter = []
        tmp_len = []
        for sublist in Sf_not:
            tmp_counter.append(sublist.count(specific_state))
            tmp_len.append(len(sublist))
        probas = [tmp_counter[i] / tmp_len[i] for i in range(len(tmp_counter))]
        best = argmax(probas)
        Sf_not_best = tmp_counter[best]
        Sf_not_average = sum(probas) / len(probas)

        return storeInfoCsv(csv_filename, [decoded_s, a], [Sf_spec, Sf_not_best, Sf_not_average, tmp_counter])
    else:
        return


#  Explore trajectories from a set of initial states
#  Input: set of initial states (int list), trajectories length (int), environment (Taxi-v3), the agent (Agent)
#  Output: list of final states, i.e.  the reachable states at horizon k, starting from each state s in S (int list)
def succ(S, k, env, agent=None):
    S_tmp = []
    terminal_states = terminalState(env)
    for _ in range(k):
        # Used for generating a group of list, where each list represents the reachable final states from a specific
        # action, different from the current action
        if all(isinstance(el, list) for el in S):
            for sublist in S:
                S_tmp_sublist = []
                for s in sublist:
                    if s not in terminal_states:
                        action = agent.predict(s)
                        for _, new_s, _, _ in env.P[s][action]:
                            S_tmp_sublist.append(new_s)
                    else:
                        S_tmp_sublist.append(s)
                S_tmp.append(S_tmp_sublist)

            S = S_tmp
            S_tmp = []
        # Used for generating the list of reachable final states from the current history's action
        else:
            for s in S:
                if s not in terminal_states:
                    action = agent.predict(s)
                    for _, new_s, _, _ in env.P[s][action]:
                        S_tmp.append(new_s)
                else:
                    S_tmp.append(s)
            S = S_tmp
            S_tmp = []

    return S


#  Store all info about an action importance in a CSV file.
#  Input: CSV file to store scores (String), state-action (int list), list of info to store in the Csv file
#  (int list list)
#  Output: an action importance (float)
def storeInfoCsv(csv_filename, s_a_list, info_list):

    Sf_goal, Sf_not_best, Sf_not_average, tmp_counter = info_list

    # --------------------------- Display info -------------------------------------
    s, a = s_a_list
    print(
        "By doing action {} from state {}, the predicate is respected in {}% \n"
        "By doing the best contrastive acttion, it is respected in {}% \n"
        "By not doing action {}, the average respect of the predicate is {}% \n".format(a, s, Sf_goal * 100, Sf_not_best * 100, a,
                                                                  Sf_not_average * 100))

    #  ------------------------ Store in CSV ----------------------------------------
    with open(csv_filename, 'a') as f:
        writer = csv.writer(f)
        # Action importance of history's action
        writer.writerow(['{}-{}'.format(s, a), 'Action : {}'.format(a), Sf_goal])
        # Action importance of the best different action
        writer.writerow(['', 'Other actions: Best', Sf_not_best])
        # Average action importance of different actions
        writer.writerow(['', 'Other actions: Average', tmp_counter, Sf_not_average])

        writer.writerow('')  # blank line

    return Sf_goal - Sf_not_average  # avg -


#  Compute the set of reachable states from s by doing a and the set of reachable states from s by doing a different
#  action from a.
#  Input: state and associate action according to the history H (int), list of available actions from s (int list), the
#  environment (Taxi-v3)
#  Output: set of reachable states from s doing a (int list) and set of reachable states from s doing a different
#  action from a (int list)
def get_Ac(s, a, actions, env):
    Ac_a = []
    Ac_nota = []
    Ac_nota_tmp = []
    for action in actions:
        for _, new_s, _, _ in env.P[s][action]:
            if action == a:
                Ac_a.append(new_s)
            else:
                Ac_nota_tmp.append(new_s)
        if action != a:
            Ac_nota.append(Ac_nota_tmp)
            Ac_nota_tmp = []

    return Ac_a, Ac_nota


#  List all terminal states
#  Input: environment (Taxi-v3)
#  Output: list of terminal states (int list)
def terminalState(env):
    terminal_states = []
    for s in [i for i in range(env.observation_space.n)]:
        for a in [i for i in range(env.action_space.n)]:
            _, new_s, _, d = env.P[s][a][0]
            if d:
                terminal_states.append(new_s)

    return terminal_states


#  Transform the passenger index of a decoded state into its location
#  Input: passenger index (int)
#  Output: coordinates (int)
def to_coordinates(passenger_idx):
    if passenger_idx == 0:
        return 0, 0
    elif passenger_idx == 1:
        return 0, 4
    elif passenger_idx == 2:
        return 4, 0
    elif passenger_idx == 3:
        return 4, 3
    else:
        return None, None


#  Render state and associated important action
#  Input: list of important actions (int list list), environment (Taxi-v3)
#  Output: None
def render_actionImportance(HXp_actions, env):
    frames = []
    for s, a, i in HXp_actions:
        env.s = s
        env.lastaction = a
        frames.append({
            'frame': env.render(),
            'state': env.s,
            'action': env.lastaction})
    print_frames(frames, hxp=True)
    print("------ End of action importance explanation ------")
    return


#  Write the head of the Table in the Csv file
#  Input: predicate to verify (String), additional information for the predicate i.e. a specific state (int) and
#  CSV file(String)
#  Output: None
def firstLineCsvTable(property, property_info, csv_filename):
    with open(csv_filename, 'a') as f:
        writer = csv.writer(f)
        if property != 'specific_state':
            writer.writerow(['PI', '', property, 'Proportion'])
        else:
            writer.writerow(['PI', '', 'state : {}'.format(property_info), 'Proportion'])
    return


#  Decode a list of states (from int to int list)
#  Input: environment (Taxi-v3), list of states (int list or int list list)
#  Output: decoded state list (int list list or int list list list)
def decodeStateList(env, state_list):
    decoded_list = []
    if all(isinstance(el, list) for el in state_list):
        for sublist in state_list:
            tmp_list = []
            for s in sublist:
                tmp_list.append(list(env.decode(s)))
            decoded_list.append(tmp_list)
    else:
        for s in state_list:
            decoded_list.append(list(env.decode(s)))
    return decoded_list


#  Compute the argmax of an array
#  Input: an array (list)
#  Output: index of the maximum value (int)
def argmax(array):
    array = list(array)
    return array.index(max(array))

#  cf: https://casey-barr.github.io/open-ai-taxi-problem/
#  Display sequence of render with info in a row
#  Input: Dictionary containing rend, timestep, state, action and reward info (dict)
#  Output: None
def print_frames(frames, hxp=False):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        if not hxp:
            print(f"Reward: {frame['reward']}")
        sleep(.5)
    return
