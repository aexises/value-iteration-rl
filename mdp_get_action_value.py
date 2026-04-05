def get_action_value(mdp, state_values, state, action, gamma):
    """ Вычисляет Q(s,a) согласно формуле выше """
    q_value = 0.0
    for next_state, prob in mdp.get_next_states(state, action).items():
        reward = mdp.get_reward(state, action, next_state)
        q_value += prob * (reward + gamma * state_values[next_state])
    return q_value
