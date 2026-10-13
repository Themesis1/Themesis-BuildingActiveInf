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
# DATE: 2026-10-12
# VERSION: 2.0   (the starting distribution and the agent's belief are now
#                 two separate objects: TRUE_START_D and D)
#
#===================================================================================================
# WHAT THIS LAB DOES:
#===================================================================================================
#
# Introduces the Agent as a Python object, and two distributions over the
# five rooms that are easy to mix up:
#
#   TRUE_START_D -- the WORLD's rule for where the agent really starts.
#                   The simulator uses it once, to place the agent. The
#                   agent never sees it.
#   D            -- the agent's BELIEF about where it is (its prior). A
#                   best guess, not a certainty. Until Week 2 nothing in
#                   the code uses it; it simply sits on the agent.
#
# HOW TO USE THIS FILE:
#   Run each cell one at a time using Ctrl+Enter (or the Run Cell button).
#   Study the plot before moving to the next cell.
#   You can re-run any cell independently.
#   All cells share the same kernel — variables carry forward.
#
# By the end of this lab you will:
#   (1) Define the Agent as a Python class with three attributes:
#       D (belief), position (actual location), history (memory)
#   (2) Run four experiments showing different STARTING distributions
#       (TRUE_START_D), and see where the world places the agent
#   (3) See the agent's position visualized as a red node in the graph
#   (4) Run a fifth experiment where the agent's belief is confident and
#       WRONG, and see that belief and position are separate things
#
#===================================================================================================
# KEY INSIGHT — THE MODEL IS NOT THE WORLD:
#===================================================================================================
#
# D is the agent's BELIEF about where it might be.
# position is where the agent ACTUALLY IS.
# TRUE_START_D is the rule the world used to put it there.
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
# D        = the agent's BELIEF about where it is (numpy array)
# position = where the agent actually is (integer index)
# history  = list of visited nodes (memory)
#
# NOTE: D and position are two separate things.
# D is the model. position is the world. They are never the same attribute.
#
# TRUE_START_D (the world's starting rule) is NOT stored on the agent.
# It belongs to the world, and is passed INTO sample_position(), the one
# method that needs it.

class Agent:
    def __init__(self, num_nodes, initial_belief=None):
        # By default the agent starts genuinely unsure: uniform belief,
        # one fifth for each room. Pass initial_belief to give it a
        # different guess (Experiment E does this).
        if initial_belief is None:
            self.D = np.ones(num_nodes) / num_nodes
        else:
            self.D = initial_belief.copy()
        self.position = 0
        self.history = []

    def sample_position(self, TRUE_START_D):
        # The WORLD places the agent: a weighted random draw from the
        # starting distribution. (It does not use the agent's belief D.)
        self.position = np.random.choice(len(TRUE_START_D), p=TRUE_START_D)
        self.history.append(self.position)
        return self.position

    def describe(self):
        print(f"\nAgent State:")
        print(f"  D (agent's belief): {self.D}")
        print(f"  position (actual):  {self.position} ({nodes[self.position]})")
        print(f"  history: {[nodes[i] for i in self.history]}")

# --- Visualization function ---
def visualize_agent(G, pos, nodes, agent, start_rule, title="Agent Position"):
    node_colors = [
        '#e84040' if i == agent.position
        else '#e8f4f8'
        for i in range(len(nodes))
    ]
    rule_str = "[" + ", ".join(f"{v:.1f}" for v in start_rule) + "]"
    belief_str = "[" + ", ".join(f"{v:.1f}" for v in agent.D) + "]"
    subtitle = (f"World's starting rule TRUE_START_D = {rule_str}\n"
                f"Agent's belief D = {belief_str}   |   "
                f"Agent at: {nodes[agent.position]}")

    plt.figure(figsize=(8, 4))
    plt.subplots_adjust(top=0.88, bottom=0.22)
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

print("Cell 1 complete: world built, Agent class defined, visualization function ready.")
print("Run Cell 2 when ready.")

# %% ==============================================================================================
# CELL 2: EXPERIMENT A — Certain Start
# The world always places the agent at Start.
# TRUE_START_D = [1.0, 0.0, 0.0, 0.0, 0.0]
# (The agent's own belief D is left at its default: uniform.)
# =================================================================================================

print("\nExperiment A: the world ALWAYS starts the agent at Start")
TRUE_START_D = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
agent_A = Agent(num_nodes)
agent_A.sample_position(TRUE_START_D)
agent_A.describe()
visualize_agent(G, pos, nodes, agent_A, TRUE_START_D,
    title="Experiment A: Certain Start — Agent at Start")

print("\nStudy the graph: Start node is RED.")
print("Notice that the agent's belief D is still uniform. The agent does not")
print("know where the world placed it.")
print("Run Cell 3 when ready.")

# %% ==============================================================================================
# CELL 3: EXPERIMENT B — Certain Start at Goal
# The world always places the agent at Goal.
# TRUE_START_D = [0.0, 0.0, 0.0, 0.0, 1.0]
# Watch the red node jump to the other end of the world.
# =================================================================================================

print("\nExperiment B: the world ALWAYS starts the agent at Goal")
TRUE_START_D = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
agent_B = Agent(num_nodes)
agent_B.sample_position(TRUE_START_D)
agent_B.describe()
visualize_agent(G, pos, nodes, agent_B, TRUE_START_D,
    title="Experiment B: Certain Start — Agent at Goal")

print("\nStudy the graph: Goal node is RED.")
print("The starting distribution placed the agent at the opposite end of the world.")
print("Run Cell 4 when ready.")

# %% ==============================================================================================
# CELL 4: EXPERIMENT C — Biased Start
# The world usually starts the agent at Start, but not always.
# TRUE_START_D = [0.6, 0.1, 0.1, 0.1, 0.1]
# Run this cell a few times — sometimes the agent lands elsewhere!
# =================================================================================================

print("\nExperiment C: the world USUALLY starts the agent at Start")
TRUE_START_D = np.array([0.6, 0.1, 0.1, 0.1, 0.1])
agent_C = Agent(num_nodes)
agent_C.sample_position(TRUE_START_D)
agent_C.describe()
visualize_agent(G, pos, nodes, agent_C, TRUE_START_D,
    title="Experiment C: Biased Start (Start = 0.6)")

print("\nStudy the graph: which node is RED?")
print("Re-run from Experiment C a few times — the agent doesn't always land at Start.")

# =================================================================================================
# EXPERIMENT D: Uniform Start — the world places the agent anywhere
# TRUE_START_D = [0.2, 0.2, 0.2, 0.2, 0.2]
# Run three times and see where the agent lands each time.
# =================================================================================================

print("\n" + "="*60)
print("Experiment D: Uniform Start — three runs")
print("TRUE_START_D = [0.2, 0.2, 0.2, 0.2, 0.2]")
print("="*60)

TRUE_START_D = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
for i in range(3):
    agent_D = Agent(num_nodes)
    agent_D.sample_position(TRUE_START_D)
    print(f"\nRun {i+1}: Agent started at {nodes[agent_D.position]}")
    visualize_agent(G, pos, nodes, agent_D, TRUE_START_D,
        title=f"Experiment D — Run {i+1} of 3: Uniform Start")

# =================================================================================================
# EXPERIMENT E: A Confident, WRONG Belief
# The world places the agent at Goal. The agent, though, is sure it is at Start.
# (The agent is never told where the world placed it. Its belief is its own.)
# TRUE_START_D = [0.0, 0.0, 0.0, 0.0, 1.0]     (the world: Goal)
# D            = [1.0, 0.0, 0.0, 0.0, 0.0]     (the agent's belief: Start)
# This is the gap between belief and reality, made visible.
# =================================================================================================

print("\n" + "="*60)
print("Experiment E: the agent's belief is confident and WRONG")
print("="*60)

TRUE_START_D = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
agent_E = Agent(num_nodes, initial_belief=np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
agent_E.sample_position(TRUE_START_D)
agent_E.describe()
visualize_agent(G, pos, nodes, agent_E, TRUE_START_D,
    title="Experiment E: Belief says Start, but the Agent is at Goal")

print("\nStudy the graph: the RED node is where the agent really is (Goal).")
print("The agent's belief D says Start. Nothing forces the two to agree.")

# =================================================================================================
# END OF LAB 1.3
# =================================================================================================

print("\n" + "="*60)
print("Lab 1.3 complete.")
print("")
print("You have seen four kinds of starting distribution (TRUE_START_D):")
print("  Experiment A: certain at Start    [1.0, 0.0, 0.0, 0.0, 0.0]")
print("  Experiment B: certain at Goal     [0.0, 0.0, 0.0, 0.0, 1.0]")
print("  Experiment C: biased toward Start [0.6, 0.1, 0.1, 0.1, 0.1]")
print("  Experiment D: uniform             [0.2, 0.2, 0.2, 0.2, 0.2]")
print("")
print("And one case where the agent's belief D disagrees with its position (E).")
print("")
print("D is the agent's BELIEF about where it might be.")
print("position is where the agent ACTUALLY IS.")
print("TRUE_START_D is the world's rule for putting it there.")
print("These are never the same thing.")
print("")
print("Thinking ahead to Lab 1.4:")
print("The agent has a belief about where it MIGHT be (D).")
print("But what can it OBSERVE from where it is?")
print("That is the A matrix.")
print("=" * 60)
