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
# LAB: 1.3
# TITLE: The D Matrix — Where Does the Agent Begin?
# DATE: 2026-07-09
# VERSION: 1.2
#
#===================================================================================================
# WHAT THIS LAB DOES:
#===================================================================================================
#
# Introduces the Agent as a Python object and the D matrix as the
# agent's prior belief about its starting location. The D matrix is
# a probability distribution over states — not a certainty, but a
# best guess before any observation is made.
#
# HOW TO USE THIS FILE:
#   Run each cell one at a time using Ctrl+Enter (or the Run Cell button).
#   Study the plot before moving to the next cell.
#   You can re-run any cell independently.
#   All cells share the same kernel — variables carry forward.
#
# By the end of this lab you will:
#   (1) Define the Agent as a Python class with three attributes:
#       D (prior belief), position (actual location), history (memory)
#   (2) Run four experiments showing different prior distributions
#   (3) See the agent's position visualized as a red node in the graph
#   (4) Experience what uniform uncertainty looks like across six runs
#
#===================================================================================================
# KEY INSIGHT — THE MODEL IS NOT THE WORLD:
#===================================================================================================
#
# D is the agent's BELIEF about where it might be.
# position is where the agent ACTUALLY IS.
# These are never the same thing.
# That gap between belief and reality is what active inference
# is designed to close.
#
#===================================================================================================
# QUESTIONS: themesisinc1@gmail.com
# GITHUB: https://github.com/Themesis1/Themesis-BuildingActiveInf
#===================================================================================================

# %% ==============================================================================================
# CELL 1: IMPORTS AND SETUP
# Run this cell first. It sets up the world and the Agent class.
# All other cells depend on this one.
# =================================================================================================

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Clear all previous plots before starting
plt.close('all')

# --- Build the world (same as Lab 1.2) ---
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

# --- Define the Agent class ---
# D        = prior belief distribution (numpy array)
# position = where the agent actually is (integer index)
# history  = list of visited nodes (memory)
#
# NOTE: D and position are two separate things.
# D is the model. position is the world. They are never the same attribute.

class Agent:
    def __init__(self, num_nodes):
        # Default: certain prior at node 0 (Start)
        self.D = np.zeros(num_nodes)
        self.D[0] = 1.0
        self.position = 0
        self.history = []

    def sample_position(self):
        # Sample actual position from prior distribution D
        self.position = np.random.choice(len(self.D), p=self.D)
        self.history.append(self.position)
        return self.position

    def describe(self):
        print(f"\nAgent State:")
        print(f"  D (prior belief): {self.D}")
        print(f"  position (actual): {self.position} ({nodes[self.position]})")
        print(f"  history: {[nodes[i] for i in self.history]}")

# --- Visualization function ---
def visualize_agent(G, pos, nodes, agent, title="Agent Position"):
    node_colors = [
        '#e84040' if i == agent.position
        else '#e8f4f8'
        for i in range(len(nodes))
    ]
    prior_str = "[" + ", ".join(f"{v:.1f}" for v in agent.D) + "]"
    subtitle = f"Prior D = {prior_str}   |   Agent at: {nodes[agent.position]}"

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
    plt.suptitle(subtitle, fontsize=11, color='#222222', y=0.02)
    plt.show()

print("Cell 1 complete: world built, Agent class defined, visualization function ready.")
print("Run Cell 2 when ready.")

# %% ==============================================================================================
# CELL 2: EXPERIMENT A — Certain Prior at Start
# The agent knows with certainty it starts at Start.
# D = [1.0, 0.0, 0.0, 0.0, 0.0]
# =================================================================================================

print("\nExperiment A: Certain prior — agent KNOWS it starts at Start")
agent_A = Agent(num_nodes)
agent_A.D = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
agent_A.sample_position()
agent_A.describe()
visualize_agent(G, pos, nodes, agent_A,
    title="Experiment A: Certain Prior — Agent at Start")

print("\nStudy the graph: Start node is RED.")
print("Run Cell 3 when ready.")

# %% ==============================================================================================
# CELL 3: EXPERIMENT B — Certain Prior at Goal
# The agent knows with certainty it starts at Goal.
# D = [0.0, 0.0, 0.0, 0.0, 1.0]
# Watch the red node jump to the other end of the world.
# =================================================================================================

print("\nExperiment B: Certain prior — agent KNOWS it starts at Goal")
agent_B = Agent(num_nodes)
agent_B.D = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
agent_B.sample_position()
agent_B.describe()
visualize_agent(G, pos, nodes, agent_B,
    title="Experiment B: Certain Prior — Agent at Goal")

print("\nStudy the graph: Goal node is RED.")
print("The D matrix placed the agent at the opposite end of the world.")
print("Run Cell 4 when ready.")

# %% ==============================================================================================
# CELL 4: EXPERIMENT C — Biased Prior
# The agent thinks it probably starts at Start, but isn't certain.
# D = [0.6, 0.1, 0.1, 0.1, 0.1]
# Run this cell a few times — sometimes the agent lands elsewhere!
# =================================================================================================

print("\nExperiment C: Biased prior — agent THINKS it probably starts at Start")
agent_C = Agent(num_nodes)
agent_C.D = np.array([0.6, 0.1, 0.1, 0.1, 0.1])
agent_C.sample_position()
agent_C.describe()
visualize_agent(G, pos, nodes, agent_C,
    title="Experiment C: Biased Prior (Start = 0.6)")

print("\nStudy the graph: which node is RED?")
print("Re-run from Experiment C a few times — the agent doesn't always land at Start.")

# =================================================================================================
# EXPERIMENT D: Uniform Prior — complete uncertainty
# D = [0.2, 0.2, 0.2, 0.2, 0.2]
# Run three times and see where the agent lands each time.
# =================================================================================================

print("\n" + "="*60)
print("Experiment D: Uniform Prior — three runs")
print("D = [0.2, 0.2, 0.2, 0.2, 0.2]")
print("="*60)

for i in range(3):
    agent_D = Agent(num_nodes)
    agent_D.D = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    agent_D.sample_position()
    print(f"\nRun {i+1}: Agent started at {nodes[agent_D.position]}")
    visualize_agent(G, pos, nodes, agent_D,
        title=f"Experiment D — Run {i+1} of 3: Uniform Prior")

# =================================================================================================
# END OF LAB 1.3
# =================================================================================================

print("\n" + "="*60)
print("Lab 1.3 complete.")
print("")
print("You have seen four kinds of prior:")
print("  Experiment A: certain at Start    D = [1.0, 0.0, 0.0, 0.0, 0.0]")
print("  Experiment B: certain at Goal     D = [0.0, 0.0, 0.0, 0.0, 1.0]")
print("  Experiment C: biased toward Start D = [0.6, 0.1, 0.1, 0.1, 0.1]")
print("  Experiment D: uniform uncertainty D = [0.2, 0.2, 0.2, 0.2, 0.2]")
print("")
print("D is the agent's BELIEF about where it might be.")
print("position is where the agent ACTUALLY IS.")
print("These are never the same thing.")
print("")
print("Thinking ahead to Lab 1.4:")
print("The agent knows where it MIGHT be (D matrix).")
print("But what can it OBSERVE from where it is?")
print("That is the A matrix.")
print("="*60)
