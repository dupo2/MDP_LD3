
# 9 states and 9 actions
states = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', ]
actions = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', ]


transition_probs = {
    's1': {'a1': {'s2': 1.0}, 'a2': {'s1': 1.0}},
    's2': {'a1': {'s1': 1.0}, 'a2': {'s2': 1.0}},
}

rewards = {
    's1': {'a1': {'s2': 5}, 'a2': {'s1': 1}},
    's2': {'a1': {'s1': 3}, 'a2': {'s2': 2}},
}

# Discount factor
gamma = 0.9

# Value iteration to compute V*(s)
def value_iteration(states, actions, transition_probs, rewards, gamma, theta=1e-6):
    V = {s: 0 for s in states}  # Initialize V(s) to 0
    while True:
        delta = 0
        for s in states:
            v = V[s]
            # Compute the new value for V[s] using the Bellman optimality equation
            V[s] = max(
                sum(
                    transition_probs[s][a].get(s_next, 0) *\
                    (rewards[s][a].get(s_next, 0) + gamma * V[s_next])
                    for s_next in states
                )
                for a in actions
            )
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V

# Derive the optimal policy pi*(s) from V*(s)
def get_optimal_policy(states, actions, transition_probs, rewards, gamma, V):
    policy = {}
    for s in states:
        # Choose the action that maximizes the expected return
        policy[s] = max(
            actions,
            key=lambda a: sum(
                transition_probs[s][a].get(s_next, 0) *\
                (rewards[s][a].get(s_next, 0) + gamma * V[s_next])
                for s_next in states
            )
        )


    return policy

V_optimal = value_iteration(states, actions, transition_probs, rewards, gamma)

# Get the optimal policy using the optimal state-value function V_optimal
optimal_policy = get_optimal_policy(states, actions, transition_probs, rewards, gamma, V_optimal)

# Print the results
print("Optimal Value Function:")
for s in states:
    print(f"V*({s}) = {V_optimal[s]:.2f}")

print("\nOptimal Policy:")
for s in states:
    print(f"pi*({s}) = {optimal_policy[s]}")
