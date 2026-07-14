####################################################################################################
#
# THEMESIS, INC. — ACTIVE INFERENCE LAB COURSE
# Building Active Inference in Python
#
#===================================================================================================
# MIT License — Copyright 2026 Themesis, Inc.
# Code created by: Alianna J. Maren; assigned to Themesis, Inc.
#===================================================================================================
#
# LAB: 1.5
# TITLE: The B Matrix — How Does the Agent Move?
# DATE: 2026-07-14
# VERSION: 1.0
#
#===================================================================================================
# WHAT THIS LAB DOES:
#===================================================================================================
#
# Introduces the B matrix — the transition probability model.
# The B matrix encodes: given that the agent is at state x and takes
# action a, what state is it likely to end up in?
#
# The B matrix is our first 3D array: shape N x N x a
#   N = number of states (5 locations)
#   a = number of actions (1 for now: Move)
#
# Each "slice" B[:,:,a] is an N x N transition matrix for one action.
# Each COLUMN of that slice sums to 1.0 — same rule as the A matrix.
#
# In this lab we use a single action (Move) with a = 1.
# The agent moves to a randomly chosen connected neighbor.
# Transition probabilities follow graph topology:
#   only connected neighbors can be reached in one step.
#
# HOW TO USE THIS FILE:
#   Run the whole file with F5 or the Run button.
#   Plots appear in the vertical thumbnail strip on the right.
#   Click any thumbnail to view full-size.
#   Click "Remove all plots" before re-running.
#
# By the end of this lab you will have:
#   (1) Built the B matrix as a 5x5x1 numpy array
#   (2) Confirmed that each column of B[:,:,0] sums to 1.0
#   (3) Run the agent through N steps and visualized each position
#   (4) Seen that the agent's sojourn is different every run
#   (5) Connected this to the Markov chain concept
#
#===================================================================================================
# KEY INSIGHT — THE AGENT MOVES BUT DOES NOT CHOOSE:
#===================================================================================================
#
# In Lab 1.5 the agent moves through its world — but randomly.
# It has no preference for Goal over Start.
# It has no memory of where it wants to go.
# It simply takes one step at a time, weighted by graph connectivity.
#
# This is a random walk — a Markov chain on the graph.
# Purposeful movement — choosing actions to reach Goal — requires
# inference and policy selection. That is Week 2.
#
#===================================================================================================
# QUESTIONS: themesisinc1@gmail.com
# GITHUB: https://github.com/Themesis1/Themesis-BuildingActiveInf
#===================================================================================================

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Clear previous plots
plt.close('all')

#===================================================================================================
# SECTION 1: BUILD THE WORLD (from Labs 1.2, 1.3, 1.4)
#===================================================================================================

nodes = ['Start', 'Room A', 'Room B', 'Room C', 'Goal']
num_nodes = len(nodes)

edges = [
    ('Start', 'Room A'),
    ('Room A', 'Room B'),
    ('Room A', 'Room C'),
    ('Room B', 'Goal'),
    ('Room C', 'Goal'),
]

G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

pos = {
    'Start':  (0, 1),
    'Room A': (1, 1),
    'Room B': (2, 1.5),
    'Room C': (2, 0.5),
    'Goal':   (3, 1),
}

# Build adjacency structure for reference
neighbors = {i: [] for i in range(num_nodes)}
for edge in edges:
    i = nodes.index(edge[0])
    j = nodes.index(edge[1])
    neighbors[i].append(j)
    neighbors[j].append(i)

print("World built successfully.")
print("\nNeighbors of each node:")
for i, node in enumerate(nodes):
    print(f"  {node}: {[nodes[j] for j in neighbors[i]]}")

#===================================================================================================
# SECTION 2: THE AGENT CLASS (from Labs 1.3 and 1.4)
#===================================================================================================
# The Agent class carries forward from Lab 1.4.
# We add one new method: act(B) — which samples the next position
# from the B matrix given the agent's current position.

class Agent:
    def __init__(self, num_nodes):
        self.D = np.zeros(num_nodes)
        self.D[0] = 1.0              # certain prior at Start
        self.position = 0
        self.history = []

    def sample_position(self):
        # Weighted random draw from prior D
        self.position = np.random.choice(len(self.D), p=self.D)
        self.history.append(self.position)
        return self.position

    def observe(self, A):
        # Weighted random draw from A column for current position
        observation = np.random.choice(A.shape[0], p=A[:, self.position])
        return observation

    def act(self, B, action=0):
        # Sample next position from B matrix.
        # B[:, self.position, action] is the probability distribution
        # over next states given current position and chosen action.
        # This is a weighted random draw — the agent moves to one
        # connected neighbor, weighted by transition probabilities.
        next_position = np.random.choice(
            B.shape[0],
            p=B[:, self.position, action]
        )
        self.position = next_position
        self.history.append(self.position)
        return next_position

    def describe(self):
        print(f"\nAgent State:")
        print(f"  position (actual): {self.position} ({nodes[self.position]})")
        print(f"  history: {[nodes[i] for i in self.history]}")

#===================================================================================================
# SECTION 3: BUILD THE B MATRIX
#===================================================================================================
# The B matrix is a 3D numpy array: shape N x N x a
#   N = 5 states, a = 1 action (Move)
#
# B[:, :, 0] is the transition matrix for the Move action.
# Each COLUMN gives the probability distribution over next states
# given the current state. Columns must sum to 1.0.
#
# Transition probabilities follow graph topology:
#   - Equal probability of moving to any connected neighbor
#   - Zero probability of moving to non-connected nodes
#   - No self-loops (agent always moves when taking Move action)
#
# Neighbors and equal probabilities:
#   Start  (1 neighbor):  Room A = 1.0
#   Room A (3 neighbors): Start = 0.33, Room B = 0.33, Room C = 0.33
#   Room B (2 neighbors): Room A = 0.5, Goal = 0.5
#   Room C (2 neighbors): Room A = 0.5, Goal = 0.5
#   Goal   (2 neighbors): Room B = 0.5, Room C = 0.5

# Build B matrix programmatically from graph structure
B = np.zeros((num_nodes, num_nodes, 1))  # shape: N x N x a (a=1)

for i in range(num_nodes):
    n = neighbors[i]
    if len(n) > 0:
        prob = 1.0 / len(n)      # equal probability for each neighbor
        for j in n:
            B[j, i, 0] = prob    # B[next_state, current_state, action]

print("\nB matrix — Move action (B[:,:,0]):")
print("Columns = current state, Rows = next state")
print("\n       Start  RoomA  RoomB  RoomC  Goal")
for i, node in enumerate(nodes):
    row = "  ".join(f"{B[i,j,0]:.2f}" for j in range(num_nodes))
    print(f"{node:8s} {row}")

#===================================================================================================
# SECTION 4: VERIFY COLUMNS SUM TO 1.0
#===================================================================================================

print("\n" + "="*60)
print("VERIFICATION: Column sums of B[:,:,0]")
print("="*60)
print("\nColumn sums (should all be 1.0):")
for i, node in enumerate(nodes):
    col_sum = B[:, i, 0].sum()
    status = "✓" if abs(col_sum - 1.0) < 1e-10 else "✗ ERROR"
    print(f"  Column {i} ({node:8s}): sum = {col_sum:.2f} {status}")

print("\nNote: unlike the A matrix, rows of B do NOT represent")
print("probability distributions either. The B matrix is read")
print("column by column: each column is one current state,")
print("each entry is the probability of moving to that next state.")

#===================================================================================================
# SECTION 5: VISUALIZATION FUNCTION
#===================================================================================================

def visualize_agent_path(G, pos, nodes, history, step, title="Agent Position"):
    # Color coding:
    # Red:       current position
    # Light red: previously visited nodes (in history)
    # Pale teal: not yet visited

    visited = set(history[:-1])   # all except current
    current = history[-1]

    node_colors = []
    for i in range(len(nodes)):
        if i == current:
            node_colors.append('#e84040')      # red: agent is HERE now
        elif i in visited:
            node_colors.append('#f4a0a0')      # light red: visited before
        else:
            node_colors.append('#e8f4f8')      # pale teal: not yet visited

    # Draw history path as arrows
    path_str = " → ".join(nodes[i] for i in history)
    subtitle = f"Step {step}  |  Path so far: {path_str}"

    plt.figure(figsize=(8, 4))
    plt.subplots_adjust(top=0.88, bottom=0.15)
    nx.draw(
        G, pos,
        labels={node: node for node in G.nodes()},
        node_color=node_colors,
        node_size=2800,
        font_size=11,
        font_weight='bold',
        font_color='#111111',
        edge_color='#2a6070',
        width=2,
        with_labels=True,
        edgecolors='#2a6070',
        linewidths=2.0,
    )
    plt.title(title, fontsize=13, fontweight='bold', color='#1a4a5e', pad=15)
    plt.suptitle(subtitle, fontsize=10, color='#222222', y=0.02)
    plt.show()

#===================================================================================================
# SECTION 6: RUN THE AGENT — N STEPS FROM START
#===================================================================================================
# Place the agent at Start (certain prior) and let it move
# for N steps. Visualize each step.
# Each run produces a different sojourn through the world.

N_STEPS = 5   # number of steps to take

print("\n" + "="*60)
print(f"RUNNING THE AGENT: {N_STEPS} steps from Start")
print("="*60)

# Initialize agent at Start
agent = Agent(num_nodes)
agent.sample_position()   # always Start (D[0]=1.0)
print(f"\nStep 0: Agent starts at {nodes[agent.position]}")
visualize_agent_path(G, pos, nodes, agent.history, step=0,
    title="Step 0: Agent at Start")

# Take N steps
for step in range(1, N_STEPS + 1):
    prev = nodes[agent.position]
    next_pos = agent.act(B, action=0)
    print(f"Step {step}: {prev} → {nodes[next_pos]}")
    visualize_agent_path(G, pos, nodes, agent.history, step=step,
        title=f"Step {step}: Agent moves to {nodes[next_pos]}")

# Final summary
print(f"\nFull sojourn: {' → '.join(nodes[i] for i in agent.history)}")
if agent.position == nodes.index('Goal'):
    print("Agent reached Goal!")
else:
    print(f"Agent ended at {nodes[agent.position]} — did not reach Goal this run.")
print("\nRun again (F5) to see a different sojourn.")

#===================================================================================================
# SECTION 7: MULTIPLE RUNS — WHERE DOES THE AGENT END UP?
#===================================================================================================
# Run the agent 20 times and record where it ends up after N steps.
# This gives a sense of the probability distribution over end states.

print("\n" + "="*60)
print(f"20 RUNS: Where does the agent end up after {N_STEPS} steps?")
print("="*60)

end_counts = {node: 0 for node in nodes}

for run in range(20):
    agent_run = Agent(num_nodes)
    agent_run.sample_position()
    for _ in range(N_STEPS):
        agent_run.act(B, action=0)
    end_counts[nodes[agent_run.position]] += 1

print("\nEnd state distribution after 20 runs:")
for node in nodes:
    count = end_counts[node]
    bar = "#" * count
    reached_goal = " ← GOAL!" if node == 'Goal' else ""
    print(f"  {node:8s}: {bar} ({count}/20){reached_goal}")

print("\nNotice: Room A tends to appear most often — it is the hub,")
print("reachable from three directions. Without preferences,")
print("the agent gravitates toward the most connected node.")
print("\nThis is a Markov chain random walk — the agent has no")
print("preference for Goal. It moves randomly, weighted only")
print("by graph connectivity.")

#===================================================================================================
# SECTION 8: CONNECTION TO MARKOV CHAINS
#===================================================================================================
# The B matrix (Move slice) IS a stochastic state transition matrix —
# the classic object from Markov chain theory.
#
# Key Markov property: the next state depends ONLY on the current state,
# not on the history of how we got here.
# (Our agent's history list tracks the path, but the transition
# probabilities don't depend on it.)
#
# After many steps, the agent converges to a STATIONARY DISTRIBUTION —
# the probability of being at each node in the long run.
# For our graph, this is proportional to node degree (number of connections):
#   Start:  degree 1 → ~10% of visits
#   Room A: degree 3 → ~30% of visits
#   Room B: degree 2 → ~20% of visits
#   Room C: degree 2 → ~20% of visits
#   Goal:   degree 2 → ~20% of visits

print("\n" + "="*60)
print("MARKOV CHAIN: Stationary distribution")
print("="*60)

# Compute stationary distribution via matrix powers
T = B[:, :, 0]   # transition matrix (the Move slice)
state = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # start at Start

print("\nDistribution over states as agent takes more steps:")
print(f"  Step  0: {np.round(state, 3)}")
for n in [1, 2, 5, 10, 20, 50, 100]:
    state_n = np.linalg.matrix_power(T, n) @ np.array([1.0,0,0,0,0])
    print(f"  Step {n:3d}: {np.round(state_n, 3)}")

print("\nAfter many steps, converges to stationary distribution:")
print("  [0.1, 0.3, 0.2, 0.2, 0.2]")
print("  (proportional to node degrees: [1, 3, 2, 2, 2])")
print("\nThis is the MAXIMUM ENTROPY state of the system —")
print("probability spread as broadly as topology allows.")
print("Active inference adds PREFERENCES to escape this state")
print("and move purposefully toward a goal. That is Week 2.")

#===================================================================================================
# END OF LAB 1.5
#===================================================================================================
print("\n" + "="*60)
print("Lab 1.5 complete.")
print("")
print("You have built the B matrix — the transition model.")
print("You have seen the agent move through its world.")
print("You have connected this to Markov chain theory.")
print("")
print("Thinking ahead to Lab 1.6:")
print("D matrix + A matrix + B matrix = complete generative model.")
print("Lab 1.6 runs a full forward pass — the agent starts,")
print("observes, moves, and observes again.")
print("Week 1 closes. The generative model is complete.")
print("="*60)
