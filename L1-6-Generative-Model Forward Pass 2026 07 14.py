# -*- coding: utf-8 -*-


####################################################################################################
#
# THEMESIS, INC. -- ACTIVE INFERENCE LAB COURSE
# Building Active Inference in Python
#
#===================================================================================================
# MIT License -- Copyright 2026 Themesis, Inc.
# Code created by: Alianna J. Maren; assigned to Themesis, Inc.
#===================================================================================================
#
# LAB: 1.6
# TITLE: The Generative Model -- Putting It All Together
# DATE: 2026-07-14
# VERSION: 1.0
#
#===================================================================================================
# WHAT THIS LAB DOES:
#===================================================================================================
#
# Lab 1.6 is the closing lab of Week 1.
# It brings together the D, A, and B matrices in a complete
# forward pass through the generative model.
#
# The forward pass:
#   (1) Sample starting position from D (prior belief)
#   (2) Sample observation from A (what does the agent sense?)
#   (3) Take one step via B (where does the agent move?)
#   (4) Sample new observation from A (what does it sense now?)
#   (5) Visualize each step: position (red) and uncertainty (pink)
#
# This is NOT inference. The agent is not updating its beliefs.
# It is running the generative model FORWARD --
# generating states, observations, and transitions.
#
# Inference -- using observations to UPDATE beliefs -- is Week 2.
#
# HOW TO USE THIS FILE:
#   Run with F5 or the Run button.
#   Plots appear in the vertical thumbnail strip on the right.
#   Click any thumbnail to view full-size.
#   Click "Remove all plots" before re-running.
#   Each run produces a different forward pass.
#
# By the end of this lab you will have:
#   (1) Run a complete forward pass through D, A, and B
#   (2) Seen the generative model working as a unified whole
#   (3) Understood why Week 2 (inference) is the necessary next step
#   (4) Connected the generative model to all of generative AI
#   (5) Understood why variational methods exist (the MCMC argument)
#
#===================================================================================================
# KEY INSIGHT -- THE LOOP IS CLOSED:
#===================================================================================================
#
# D matrix: where might the agent be? (prior belief)
# A matrix: what can the agent observe from where it is?
# B matrix: where does the agent go when it moves?
#
# Together: D -> observe via A -> move via B -> observe via A again
#
# The agent generates states and observations.
# It does NOT yet infer its location from observations.
# That gap -- between generating and inferring -- is what Week 2 closes.
#
#===================================================================================================
# QUESTIONS: themesisinc1@gmail.com
# GITHUB: https://github.com/Themesis1/Themesis-BuildingActiveInf
#===================================================================================================

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

plt.close('all')

#===================================================================================================
# SECTION 1: BUILD THE COMPLETE WORLD (carried forward from Labs 1.2-1.5)
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

neighbors = {i: [] for i in range(num_nodes)}
for edge in edges:
    i = nodes.index(edge[0])
    j = nodes.index(edge[1])
    neighbors[i].append(j)
    neighbors[j].append(i)

print("World built successfully.")

#===================================================================================================
# SECTION 2: BUILD ALL THREE MATRICES
#===================================================================================================
# For the first time, all three matrices appear together in one file.
# This is the complete generative model.

# -- D matrix: prior belief (certain at Start) --
D = np.zeros(num_nodes)
D[0] = 1.0
print(f"\nD matrix (prior):  {D}")
print(f"  Shape: {D.shape}  (N,)  -- one probability per state")

# -- A matrix: observation likelihood --
# Columns sum to 1.0. Noise follows graph topology only.
A = np.array([
    [0.90,  0.10,  0.00,  0.00,  0.00],  # observe Start
    [0.10,  0.80,  0.10,  0.10,  0.00],  # observe Room A
    [0.00,  0.05,  0.80,  0.00,  0.10],  # observe Room B
    [0.00,  0.05,  0.00,  0.80,  0.10],  # observe Room C
    [0.00,  0.00,  0.10,  0.10,  0.80],  # observe Goal
])
print(f"\nA matrix shape:    {A.shape}  (O x N)")
print(f"  Columns sum to 1.0: {np.allclose(A.sum(axis=0), 1.0)}")

# -- B matrix: transition model --
# Equal probability of moving to any connected neighbor.
B = np.zeros((num_nodes, num_nodes, 1))
for i in range(num_nodes):
    n = neighbors[i]
    prob = 1.0 / len(n)
    for j in n:
        B[j, i, 0] = prob
print(f"\nB matrix shape:    {B.shape} (N x N x a)")
print(f"  Columns sum to 1.0: {np.allclose(B[:,:,0].sum(axis=0), 1.0)}")

print("\nAll three matrices built successfully.")
print("The generative model is complete.")

#===================================================================================================
# SECTION 3: THE AGENT CLASS (complete -- all methods from Labs 1.3-1.5)
#===================================================================================================

class Agent:
    def __init__(self, D):
        # Initialize with prior belief D
        self.D = D.copy()
        self.position = 0
        self.history = []       # track true positions (simulator's view)
        self.observations = []  # track observations received (agent's view)

    def sample_position(self):
        # Weighted random draw from prior D
        # The agent "wakes up" somewhere in the world
        self.position = np.random.choice(len(self.D), p=self.D)
        self.history.append(self.position)
        return self.position

    def observe(self, A):
        # Weighted random draw from A[:, position]
        # The agent senses its environment through a noisy channel
        obs = np.random.choice(A.shape[0], p=A[:, self.position])
        self.observations.append(obs)
        return obs

    def act(self, B, action=0):
        # Weighted random draw from B[:, position, action]
        # The agent moves to a connected neighbor
        next_pos = np.random.choice(B.shape[0], p=B[:, self.position, action])
        self.position = next_pos
        self.history.append(self.position)
        return next_pos

    def describe(self):
        print(f"  True positions:  {[nodes[i] for i in self.history]}")
        print(f"  Observations:    {[nodes[i] for i in self.observations]}")

#===================================================================================================
# SECTION 4: VISUALIZATION
#===================================================================================================

def visualize_step(G, pos, nodes, agent, A, obs, title):
    # Red:      where the agent actually IS (ground truth)
    # Pink:     where it might think it is (observation uncertainty)
    # Pale teal: ruled out completely
    node_colors = []
    for i in range(len(nodes)):
        if i == agent.position:
            node_colors.append('#e84040')
        elif A[obs, i] > 0 and i != agent.position:
            node_colors.append('#f4a0a0')
        else:
            node_colors.append('#e8f4f8')

    correct = "correct" if obs == agent.position else "noisy"
    subtitle = (f"True position: {nodes[agent.position]}  |  "
                f"Observation: {nodes[obs]} ({correct})")

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
# SECTION 5: THE FORWARD PASS -- D -> A -> B -> A
#===================================================================================================

print("\n" + "="*60)
print("THE FORWARD PASS")
print("D -> observe -> move -> observe")
print("="*60)

agent = Agent(D)

# Step 1: Sample starting position from D
print("\nStep 1: Sample starting position from D (prior belief)")
agent.sample_position()
print(f"  Agent wakes up at: {nodes[agent.position]}")

# Step 2: First observation via A
print("\nStep 2: First observation via A (noisy sensing)")
obs1 = agent.observe(A)
correct1 = "correct" if obs1 == agent.position else "noisy"
print(f"  True position:  {nodes[agent.position]}")
print(f"  Observation:    {nodes[obs1]} ({correct1})")
visualize_step(G, pos, nodes, agent, A, obs1,
    title="Step 1: Agent wakes up -- first observation")

# Step 3: Move via B
print("\nStep 3: Agent moves via B (transition model)")
prev = nodes[agent.position]
agent.act(B, action=0)
print(f"  Moved: {prev} -> {nodes[agent.position]}")

# Step 4: Second observation via A
print("\nStep 4: Second observation via A (noisy sensing)")
obs2 = agent.observe(A)
correct2 = "correct" if obs2 == agent.position else "noisy"
print(f"  True position:  {nodes[agent.position]}")
print(f"  Observation:    {nodes[obs2]} ({correct2})")
visualize_step(G, pos, nodes, agent, A, obs2,
    title="Step 2: Agent moves -- second observation")

# Summary
print("\n" + "="*60)
print("FORWARD PASS COMPLETE")
print("="*60)
agent.describe()

#===================================================================================================
# SECTION 6: WHAT THE AGENT KNOWS vs. WHAT WE KNOW
#===================================================================================================

print("\n" + "="*60)
print("WHAT THE AGENT KNOWS vs. WHAT WE KNOW")
print("="*60)
print("\nWE (simulators) know everything:")
print(f"  True position step 1: {nodes[agent.history[0]]}")
print(f"  True position step 2: {nodes[agent.history[1]]}")
print(f"  Observation step 1:   {nodes[obs1]}")
print(f"  Observation step 2:   {nodes[obs2]}")
print("\nTHE AGENT knows only its observations:")
print(f"  Observation step 1:   {nodes[obs1]}")
print(f"  Observation step 2:   {nodes[obs2]}")
print("\nThe agent does NOT know its true position.")
print("It must INFER position from its observations.")
print("That inference requires Bayes' theorem:")
print("  p(x|y) = p(y|x) * p(x) / p(y)")
print("\nInference is Week 2.")

#===================================================================================================
# SECTION 7: WHY ACTIVE INFERENCE? THE MCMC ARGUMENT
#===================================================================================================

print("\n" + "="*60)
print("WHY ACTIVE INFERENCE? THE MCMC ARGUMENT")
print("="*60)
print()
print("In principle, we could estimate where the agent is by running")
print("thousands of forward passes and counting outcomes.")
print("This is the Monte Carlo Markov Chain (MCMC) approach.")
print()
print("For our 5-node world, MCMC works fine.")
print()
print("But real-world systems have millions or billions of states.")
print("Running MCMC to convergence becomes computationally intractable.")
print()
print("This is precisely why active inference uses VARIATIONAL methods.")
print("Rather than sampling the posterior exhaustively, variational")
print("inference APPROXIMATES it efficiently using the variational")
print("free energy (VFE) -- the same F = H - S from T3,")
print("now applied to belief updating.")
print()
print("The entire mathematical machinery of Week 2 exists because")
print("MCMC is too slow. Variational inference is the solution.")
print("Active inference is variational inference with action.")
print()
print("That is why we are here.")

#===================================================================================================
# SECTION 8: WEEK 1 COMPLETE -- WHAT YOU HAVE BUILT
#===================================================================================================

print()
print("="*60)
print("WEEK 1 COMPLETE")
print("="*60)
print()
print("You have built a complete generative model from scratch:")
print()
print("  Lab 1.1: Python environment confirmed")
print("  Lab 1.2: World model -- graph with 5 locations, 5 connections")
print("  Lab 1.3: D matrix -- prior belief over starting states")
print("  Lab 1.4: A matrix -- observation likelihood, noisy sensing")
print("  Lab 1.5: B matrix -- transition probabilities, movement model")
print("  Lab 1.6: Forward pass -- D, A, B working as a unified whole")
print()
print("Every line of code connects to a mathematical operation.")
print("Every matrix was built from first principles.")
print("Nothing was hidden in a black box.")
print()
print("This is what pymdp does internally.")
print("You now know what is inside.")
print()
print("Week 2 asks the harder question:")
print("Given what the agent observes, what state is it probably in?")
print("And once it knows that -- what should it do next?")
print()
print("That is inference. That is action. That is active inference.")