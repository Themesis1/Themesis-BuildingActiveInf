# Themesis, Inc. -- Building Active Inference in Python
# Lab 2.2 -- Predict AND Update, Over Multiple Steps
# v1.0 (2026-10-10)
#
# In Lab 2.1 the agent stood still and saw one observation. In this lab the
# agent moves, and everything it believes has to keep up with that.
#
# In Lab 2.1 the position was held fixed, so one step (UPDATE) was enough.
# Here the true position CHANGES between observations, and that single fact
# forces a second step in front of the first: PREDICT.
# (See Lab 2.2 Appendix B for why PREDICT is a sum over
# previous states.)
#
# KEY INSIGHT: when the hidden state changes between observations, yesterday's
# posterior is NOT today's prior. It has to be pushed forward through B first.
#
#     PREDICT:  p(x_t)   = B @ p(x_{t-1})
#     UPDATE:   p(x_t|y) = p(y|x_t) * p(x_t) / p(y)       (A, as in Lab 2.1)
#
# Reading the PREDICT line in English:
#   x_{t-1} is the room I was in one step ago; x_t is the room I am in now.
#   They are the SAME hidden state x, seen at two moments. My belief about
#   where I am now, BEFORE the new observation comes in, is last step's
#   belief pushed through B.
#
# --------------------------------------------------------------------
# TWO DIRECTIONS (the idea at the heart of Bayesian thinking)
# --------------------------------------------------------------------
# GENERATING (Labs 1.3-1.6): start from the hidden state x
#   (the agent's TRUE position, self.position), produce an observation y
#   (the variable obs).    x  --A-->  y     observe() does this. The world
#   works this way. Nothing about it is uncertain to the simulator.
#
# INFERRING (Lab 2.1 onward): start from the observation y (obs), work BACK
#   to a belief about the hidden state x (the vector self.D, p(x)).
#       y  -->  p(x|y)     bayes_update() does this. The agent has to work
#   this way, because it never sees x.
#
# Same matrix A in both directions. Generating reads a COLUMN of A (fixed x,
# varying y). Inferring reads a ROW of A (fixed y, varying x). Bayes'
# theorem is the rule for turning the first direction into the second.
#
# NEW IN THIS LAB: a third direction -- FORWARD THROUGH TIME. The hidden
# state x changes from one moment to the next (via B), so the belief has to
# be carried forward before the next observation can be used.
#
# SYMBOL KEY (code name for each Bayesian symbol)
#   x      the hidden state: the true room            self.position
#   y      the observation: a noisy reading           obs   (all of them are
#                                                     kept in self.observations)
#   p(x)   the belief about x                         self.D
#   p(y|x) the observation model                      A
#   p(x_t | x_{t-1})  the transition model            B
# --------------------------------------------------------------------
#
# --------------------------------------------------------------------
# TWO KINDS OF MULTIPLICATION (see Lab 2.2 Appendix A)
# --------------------------------------------------------------------
#   @   matrix-vector product, used in PREDICT. Each output entry is a SUM
#       of products across ALL five rooms, so belief gets mixed and spreads.
#   *   elementwise product, used in UPDATE. Entry i only meets entry i. No
#       mixing: each room's belief is reweighted by how well that room
#       explains the observation.
# --------------------------------------------------------------------
#
# Architecture note: TRUE_WORLD_D, A and B are fixed, standalone objects,
# passed INTO the methods that need them. The agent itself carries only
# what changes: self.position, self.D, and its two history lists.

import numpy as np

np.random.seed(7)   # remove this line to get a different run every time

# --- The five-node world (unchanged from Labs 1.2-2.1) ---
nodes = ['Start', 'Room A', 'Room B', 'Room C', 'Goal']
num_nodes = len(nodes)

edges = [
    ('Start', 'Room A'),
    ('Room A', 'Room B'),
    ('Room A', 'Room C'),
    ('Room B', 'Goal'),
    ('Room C', 'Goal'),
]
neighbors = {i: [] for i in range(num_nodes)}
for edge in edges:
    i = nodes.index(edge[0])
    j = nodes.index(edge[1])
    neighbors[i].append(j)
    neighbors[j].append(i)

# --- TRUE_WORLD_D: fixed ground truth, used only to sample the real start ---
# In English: the simulator's rule for where the agent REALLY starts. The
# agent never sees it.
TRUE_WORLD_D = np.array([1.0, 0.0, 0.0, 0.0, 0.0])   # certain at Start

# --- A matrix (unchanged from Lab 1.4 / Lab 2.1) ---
# In English: if I am really in room x, how likely is each reading y?
# This is p(y|x). Each COLUMN is one true room, and sums to 1.
A = np.array([
    [0.90, 0.10, 0.00, 0.00, 0.00],
    [0.10, 0.80, 0.10, 0.10, 0.00],
    [0.00, 0.05, 0.80, 0.00, 0.10],
    [0.00, 0.05, 0.00, 0.80, 0.10],
    [0.00, 0.00, 0.10, 0.10, 0.80],
])

# --- B matrix (unchanged from Lab 1.5) ---
# In English: if I am in room x now, how likely am I to be in each room
# next? This is p(x_next | x_now). B[next, current, action]; each COLUMN
# (one current room) sums to 1.
# One action only here: "move to a randomly chosen neighbor."
B = np.zeros((num_nodes, num_nodes, 1))
for i in range(num_nodes):
    n = neighbors[i]
    if len(n) > 0:
        prob = 1.0 / len(n)
        for j in n:
            B[j, i, 0] = prob


def entropy(p):
    """
    Shannon entropy of a belief, in nats (natural log, as in statistical mechanics):

        H = -sum_j p_j ln p_j        (with 0 * ln 0 taken as 0)

    Reading it: how spread out the belief is across the five rooms. H = 0
    means all the weight is on one room; the uniform belief over five rooms
    has the maximum, ln 5 = 1.61 nats. (Divide by ln 2 to get bits.)
    """
    p = p[p > 0]
    return max(0.0, -(p * np.log(p)).sum())   # max() avoids printing -0.00


class Agent:
    """
    The agent carries only what CHANGES:

    self.position     : x, the TRUE hidden state. The simulator knows it;
                        the agent's belief code never reads it.
    self.D            : the agent's BELIEF, p(x). Starts uniform. Now changed
                        by TWO methods per time step: predict() then
                        bayes_update().
    self.history      : the true positions, in order (the path).
    self.observations : the readings the agent got, in order.

    Fixed objects (TRUE_WORLD_D, A, B) are NOT stored here; they are passed
    into the methods that use them.
    """
    def __init__(self, num_states, initial_belief=None):
        # Belief starts genuinely uncertain: uniform over all states.
        if initial_belief is None:
            self.D = np.ones(num_states) / num_states
        else:
            self.D = initial_belief.copy()
        self.position = 0
        self.history = []
        self.observations = []
        self.broken = False

    # ================================================================
    # WORLD SIDE: the simulator does these. (The agent has no say, and
    # never sees self.position directly.)
    # ================================================================
    # Pattern in the next three methods: RECORD the result in an attribute
    # (so we can review the path later), and RETURN it (so the main loop can
    # use it right now).

    def sample_position(self, TRUE_WORLD_D):
        """Draw the TRUE starting position from TRUE_WORLD_D (not belief)."""
        self.position = np.random.choice(len(TRUE_WORLD_D), p=TRUE_WORLD_D)
        self.history.append(self.position)
        return self.position

    def act(self, B, action=0):
        """
        The TRUE position changes: it is drawn from B's column for the
        current room. (Only one action in this lab: move to a random neighbor.)
        The agent's BELIEF has not caught up yet -- self.D is now out of date,
        which is exactly why PREDICT is needed.
        """
        self.position = np.random.choice(B.shape[0], p=B[:, self.position, action])
        self.history.append(self.position)
        return self.position

    def observe(self, A):
        """Draw a noisy observation y from A, based on the TRUE position."""
        obs = np.random.choice(A.shape[0], p=A[:, self.position])
        self.observations.append(obs)
        return obs

    # ================================================================
    # AGENT SIDE: the agent does these, using only its belief and what it
    # was told (B, A, obs). Neither method reads self.position.
    # ================================================================

    # ---------------- NEW IN THIS LAB ----------------
    def predict(self, B, action=0):
        """
        PREDICT: push the current belief self.D forward through B.

            p(x_t) = sum over x_{t-1} of  B[x_t, x_{t-1}] * p(x_{t-1})

        That sum-of-products is a matrix-vector product (@). The result is
        the PRIOR for the coming update. Notice: no observation is used here.
        Movement alone spreads belief out, so entropy usually goes UP.
        """
        self.D = B[:, :, action] @ self.D
        return self.D

    # ---------------- REUSED FROM LAB 2.1 ----------------
    def bayes_update(self, A, obs):
        """
        UPDATE: p(x|y) = p(y|x) p(x) / p(y). Identical to Lab 2.1.
        Uses the elementwise product (*), then normalizes.
        Returns the new belief AND the evidence p(y); only the belief is
        stored (in self.D).
        """
        unnormalized = A[obs, :] * self.D
        evidence = unnormalized.sum()
        if evidence == 0:
            # The belief said this observation was IMPOSSIBLE. Bayes cannot
            # divide by zero, so we flag it and leave the belief as it was.
            # (A belief that ruled out the truth is a broken belief.)
            self.broken = True
            return self.D, 0.0
        self.D = unnormalized / evidence
        return self.D, evidence


def show(label, p, broken=False):
    if broken:
        print(f"    {label:<24}BROKEN: observation was impossible under this belief")
        return
    print(f"    {label:<24}{np.round(p, 3)}   H = {abs(entropy(p)):.2f} nats")


if __name__ == "__main__":
    NUM_STEPS = 6

    # One simulated world; two agents watching it through the SAME observations.
    #   smart: PREDICT then UPDATE (correct)
    #   naive: UPDATE only, ignoring that the world moved (the Lab 2.1 habit)
    smart = Agent(num_nodes)
    naive = Agent(num_nodes)

    smart.sample_position(TRUE_WORLD_D)
    naive.position = smart.position          # same true world for both

    print("Lab 2.2 -- Predict AND Update")
    print(f"Nodes: {nodes}\n")

    # Step 0: the agent has not moved yet, so a plain update (exactly Lab 2.1).
    obs = smart.observe(A)
    smart.bayes_update(A, obs)
    naive.bayes_update(A, obs)
    print(f"t=0  true: {nodes[smart.position]:<7} observed: {nodes[obs]}")
    show("smart posterior:", smart.D)
    show("naive posterior:", naive.D)

    for t in range(1, NUM_STEPS):
        true_pos = smart.act(B)              # the TRUE position changes
        naive.position = true_pos            # (neither belief has caught up)

        prior = smart.predict(B).copy()      # PREDICT (smart only)
        obs = smart.observe(A)               # one observation, shared
        smart.bayes_update(A, obs)           # UPDATE
        if not naive.broken:
            naive.bayes_update(A, obs)       # UPDATE only

        print(f"\nt={t}  true: {nodes[true_pos]:<7} observed: {nodes[obs]}")
        show("smart after PREDICT:", prior)
        show("smart after UPDATE:", smart.D)
        show("naive (update only):", naive.D, naive.broken)

    print("\nChecks")
    print(f"  smart belief sums to {smart.D.sum():.6f}")
    print(f"  naive belief sums to {naive.D.sum():.6f}   (broken: {naive.broken})")
    print(f"  true path:  {[nodes[i] for i in smart.history]}")
    print(f"  most likely (smart): {nodes[int(np.argmax(smart.D))]}   "
          f"most likely (naive): {nodes[int(np.argmax(naive.D))]}")
