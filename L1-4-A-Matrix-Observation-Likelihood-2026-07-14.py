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
# LAB: 1.4
# TITLE: The A Matrix — What Can the Agent Observe?
# DATE: 2026-07-14
# VERSION: 1.0
#
#===================================================================================================
# WHAT THIS LAB DOES:
#===================================================================================================
#
# Introduces the A matrix — the observation likelihood model. The A matrix
# encodes what the agent can observe from each location: given that the agent
# is at state x, what observation y is it likely to receive?
#
# The A matrix is our first true 2D matrix: shape O x N, where O is the
# number of possible observations and N is the number of states.
# In our five-location world, O = N = 5.
#
# Key insight: observation noise follows graph topology.
# An agent can be confused about nearby (connected) locations.
# An agent cannot be confused about distant (unconnected) locations.
#
# HOW TO USE THIS FILE:
#   Run the whole file with F5 or the Run button.
#   Plots appear in the vertical thumbnail strip on the right side of
#   the Plots pane. Click any thumbnail to view it full-size.
#   Click "Remove all plots" before re-running to clear previous results.
#
# By the end of this lab you will have:
#   (1) Built the A matrix as a 5x5 numpy array
#   (2) Confirmed that each COLUMN sums to 1.0 (each column is a
#       probability distribution over observations given one state)
#   (3) Confirmed that ROWS do not necessarily sum to 1.0
#   (4) Sampled an observation from the agent's current position
#   (5) Visualized the agent's position (red) and observation
#       uncertainty (pink) simultaneously in the graph
#
#===================================================================================================
# KEY INSIGHT — COLUMNS SUM TO 1, ROWS DO NOT:
#===================================================================================================
#
# Each COLUMN of the A matrix is a complete probability distribution:
#   "Given that I am at state X, what is the probability of each observation?"
#   → Must sum to 1.0
#
# Each ROW of the A matrix is NOT a probability distribution:
#   "Given that I received observation Y, how likely is each state?"
#   → Does NOT need to sum to 1.0
#
# This distinction is fundamental. Confusing rows and columns is one of
# the most common errors in understanding observation models.
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
# SECTION 1: BUILD THE WORLD (from Labs 1.2 and 1.3)
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

print("World built successfully.")

#===================================================================================================
# SECTION 2: THE AGENT CLASS (from Lab 1.3, unchanged)
#===================================================================================================
# The Agent class carries forward from Lab 1.3.
# We add one new method: observe() — which samples an observation
# from the A matrix given the agent's current position.

class Agent:
    def __init__(self, num_nodes):
        # Default: certain prior at node 0 (Start)
        self.D = np.zeros(num_nodes)
        self.D[0] = 1.0
        self.position = 0
        self.history = []

    def sample_position(self):
        # Makes a single WEIGHTED random draw from the range 0 to (num_nodes - 1).
        # np.random.choice arguments:
        #   - len(self.D) : the number of choices (5 nodes, indexed 0-4)
        #   - p=self.D    : the probability weights for each choice
        #                   (this is what makes it WEIGHTED, not purely random)
        # Each node's probability of being selected equals its value in self.D.
        # Example: if D = [0.0, 1.0, 0.0, 0.0, 0.0], node 1 is ALWAYS selected.
        # Example: if D = [0.2, 0.2, 0.2, 0.2, 0.2], each node is equally likely.
        # Example: if D = [0.6, 0.1, 0.1, 0.1, 0.1], node 0 selected ~60% of the time.
        self.position = np.random.choice(len(self.D), p=self.D)
        self.history.append(self.position)
        return self.position

    def observe(self, A):
        # Sample an observation from the A matrix column
        # corresponding to the agent's current position.
        # A[:, self.position] is the probability distribution
        # over observations given the agent's true position.
        observation = np.random.choice(A.shape[0], p=A[:, self.position])
        return observation

    def describe(self):
        print(f"\nAgent State:")
        print(f"  D (prior belief): {self.D}")
        print(f"  position (actual): {self.position} ({nodes[self.position]})")
        print(f"  history: {[nodes[i] for i in self.history]}")

#===================================================================================================
# SECTION 3: BUILD THE A MATRIX
#===================================================================================================
# The A matrix is a 5x5 numpy array: shape O x N
#   O = number of possible observations = 5
#   N = number of states (locations) = 5
#
# Each COLUMN represents one state — it is a probability distribution
# over observations given that state. Each column MUST sum to 1.0.
#
# Noise follows graph topology:
#   - High probability on the diagonal (agent correctly identifies location)
#   - Small noise only on directly connected neighbors
#   - Zero probability for non-connected locations
#
# Columns: Start, Room A, Room B, Room C, Goal
# Rows:    observation of Start, Room A, Room B, Room C, Goal

A = np.array([
    # Start  RoomA  RoomB  RoomC  Goal
    [0.90,  0.10,  0.00,  0.00,  0.00],  # observe Start
    [0.10,  0.80,  0.10,  0.10,  0.00],  # observe Room A
    [0.00,  0.05,  0.80,  0.00,  0.10],  # observe Room B
    [0.00,  0.05,  0.00,  0.80,  0.10],  # observe Room C
    [0.00,  0.00,  0.10,  0.10,  0.80],  # observe Goal
])

print("\nA matrix (O x N = 5 x 5):")
print(f"Shape: {A.shape}")
print("\nColumn labels: Start, Room A, Room B, Room C, Goal")
print("Row labels:    Obs:Start, Obs:RoomA, Obs:RoomB, Obs:RoomC, Obs:Goal")
print("\nA =")
print(A)

#===================================================================================================
# SECTION 4: VERIFY COLUMNS SUM TO 1.0, ROWS DO NOT
#===================================================================================================

print("\n" + "="*60)
print("VERIFICATION: Column and Row Sums")
print("="*60)

print("\nColumn sums (should all be 1.0):")
for i, node in enumerate(nodes):
    col_sum = A[:, i].sum()
    status = "✓" if abs(col_sum - 1.0) < 1e-10 else "✗ ERROR"
    print(f"  Column {i} ({node:8s}): sum = {col_sum:.2f} {status}")

print("\nRow sums (do NOT need to be 1.0):")
obs_labels = ['Obs:Start', 'Obs:RoomA', 'Obs:RoomB', 'Obs:RoomC', 'Obs:Goal']
for i, label in enumerate(obs_labels):
    row_sum = A[i, :].sum()
    print(f"  Row {i} ({label:12s}): sum = {row_sum:.2f}")

print("\nNotice: columns sum to 1.0 (they are probability distributions).")
print("Rows do NOT sum to 1.0 — and that is correct.")
print("Each column answers: 'Given I am at state X, what do I observe?'")
print("Each row answers:    'Given I observe Y, how likely is each state?'")
print("Only the first question is a probability distribution.")

#===================================================================================================
# SECTION 5: SAMPLE AN OBSERVATION
#===================================================================================================
# Place the agent at a specific location and sample an observation.
# The observation is drawn from A[:, agent.position] —
# the column of A corresponding to the agent's true position.

print("\n" + "="*60)
print("SAMPLING OBSERVATIONS")
print("="*60)

# Place agent at Room A (position 1) with certain prior
agent = Agent(num_nodes)
agent.D = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
agent.sample_position()
agent.describe()

# Sample observation
obs = agent.observe(A)
print(f"\n  Observation received: {obs} ({nodes[obs]})")
print(f"  Agent is actually at: {agent.position} ({nodes[agent.position]})")

if obs == agent.position:
    print(f"  → Correct observation: agent observed its true location.")
else:
    print(f"  → Noisy observation: agent observed {nodes[obs]} but is at {nodes[agent.position]}.")
    print(f"     This is observation noise in action.")

#===================================================================================================
# SECTION 6: VISUALIZATION — RED AND PINK NODES
#===================================================================================================
# Red node:  where the agent actually IS (ground truth — position)
# Pink nodes: locations that could be confused with the actual position
#             based on the A matrix (observation uncertainty)
# Pale teal:  locations ruled out completely (zero observation probability)

def visualize_agent_with_observation(G, pos, nodes, agent, A, obs, title="Agent and Observation"):
    node_colors = []
    for i in range(len(nodes)):
        if i == agent.position:
            node_colors.append('#e84040')      # red: agent is HERE (ground truth)
        elif A[obs, i] > 0 and i != agent.position:
            node_colors.append('#f4a0a0')      # pink: could be confused with actual position
        else:
            node_colors.append('#e8f4f8')      # pale teal: ruled out

    # Build subtitle
    pos_str = nodes[agent.position]
    obs_str = nodes[obs]
    correct = "✓ correct" if obs == agent.position else "✗ noisy"
    subtitle = (f"True position: {pos_str}  |  "
                f"Observation received: {obs_str} ({correct})")

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

# Visualize the agent at Room A with its observation
visualize_agent_with_observation(G, pos, nodes, agent, A, obs,
    title="Lab 1.4: Agent Position (red) and Observation Uncertainty (pink)")

#===================================================================================================
# SECTION 7: FOUR EXPERIMENTS — OBSERVATION FROM EACH LOCATION
#===================================================================================================
# Place the agent at each of the five locations in turn.
# Sample one observation from each.
# Visualize: where is the agent (red) vs. what could be confused (pink)?

print("\n" + "="*60)
print("FIVE EXPERIMENTS: Agent at each location")
print("="*60)

for i, node_name in enumerate(nodes):
    # Place agent with certain prior at location i
    agent_exp = Agent(num_nodes)
    agent_exp.D = np.zeros(num_nodes)
    agent_exp.D[i] = 1.0
    agent_exp.sample_position()

    # Sample observation
    obs_exp = agent_exp.observe(A)

    print(f"\nExperiment {i+1}: Agent at {node_name}")
    print(f"  True position: {nodes[agent_exp.position]}")
    print(f"  Observation:   {nodes[obs_exp]}")
    if obs_exp == agent_exp.position:
        print(f"  → Correct observation")
    else:
        print(f"  → Noisy observation (confused {nodes[agent_exp.position]} with {nodes[obs_exp]})")

    visualize_agent_with_observation(G, pos, nodes, agent_exp, A, obs_exp,
        title=f"Experiment {i+1}: Agent at {node_name}")

#===================================================================================================
# SECTION 8: THE BAYESIAN UPDATE — WHAT DOES THE OBSERVATION TELL US?
#===================================================================================================
# The A matrix encodes p(y|x) — the likelihood.
# The D matrix encodes p(x) — the prior.
# Together they give us the posterior p(x|y) via Bayes' theorem:
#
#   p(x|y) ∝ p(y|x) · p(x)
#
# (We omit the normalizing term p(y) here and normalize manually.)
# This is the beginning of inference — Week 2 of the lab course.

print("\n" + "="*60)
print("PREVIEW: The Bayesian Update")
print("="*60)

# Place agent with uniform prior — complete uncertainty
# We use a uniform prior here deliberately.
# When all states are equally likely before the observation,
# the posterior is shaped entirely by the likelihood — the A matrix.
# This is the clearest way to see what the A matrix is doing.
# In later labs, when the prior is non-uniform, both D and A
# will shape the posterior together. Here, only A matters.
agent_bayes = Agent(num_nodes)
agent_bayes.D = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
agent_bayes.sample_position()
obs_bayes = agent_bayes.observe(A)

print(f"\nAgent's prior D: {agent_bayes.D}")
print(f"True position:   {nodes[agent_bayes.position]}")
print(f"Observation:     {nodes[obs_bayes]}")

# Compute unnormalized posterior: p(y|x) * p(x) for each state
likelihood = A[obs_bayes, :]           # p(y|x) for each x, given observation obs_bayes
prior = agent_bayes.D                  # p(x)
unnormalized = likelihood * prior      # p(y|x) * p(x)
posterior = unnormalized / unnormalized.sum()
# .sum() adds all elements of the array together.
# Dividing by this normalizes the result so the posterior sums to 1.0
# — making it a valid probability distribution.
# This is mathematically equivalent to dividing by p(y),
# the model evidence term in Bayes' theorem.

print(f"\nLikelihood p(y={nodes[obs_bayes]}|x): {np.round(likelihood, 3)}")
print(f"Prior p(x):                          {np.round(prior, 3)}")
print(f"Posterior p(x|y) (normalized):       {np.round(posterior, 3)}")
print(f"\nMost likely state after observation: {nodes[np.argmax(posterior)]}")
print(f"True position:                       {nodes[agent_bayes.position]}")

print("\n" + "="*60)
print("Lab 1.4 complete.")
print("")
print("You have built the A matrix and seen observation noise in action.")
print("You have confirmed that columns sum to 1.0 and rows do not.")
print("You have visualized the gap between true position (red)")
print("and observation uncertainty (pink).")
print("")
print("In the preview above, you computed your first Bayesian update:")
print("  p(x|y) ∝ p(y|x) · p(x)")
print("That computation — inference — is the heart of AIF.")
print("Week 2 of this lab course is devoted entirely to it.")
print("="*60)
