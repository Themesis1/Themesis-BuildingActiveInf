# Themesis, Inc. -- Building Active Inference in Python
# Lab 2.1 -- Bayesian Inference: The Update
# DRAFT -- extends the Lab 1.6 Agent class. Position is held fixed in this
# lab; no movement, so no PREDICT step is needed (see Appendix F). This is
# the first time D is asked to do a genuinely new job -- see the note below,
# right where D is introduced.

import numpy as np

# --- The five-node world (unchanged from Labs 1.2-1.6) ---
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

# --- TRUE_WORLD_D (this is Lab 1.3's D, carried forward, renamed) ---
# Fixed. Never updated, here or anywhere else in this lab. Used only to
# sample where the agent actually, truly starts -- a fact the SIMULATOR
# knows and the AGENT never does. This is the "ground truth" prior.
#
# Labs 1.2-1.6 only ever needed ONE distribution named D, because the only
# thing worth tracking was the true position -- there was no belief yet to
# confuse it with. This lab is the first one that needs a SECOND, separate
# distribution: the agent's own belief about where it is. Rather than
# invent a new name for that belief, we keep calling it "D" (Lab 1.6's own
# Bayes-theorem table already mapped D to p(x) -- that mapping doesn't
# change, it just starts moving). The FIXED, ground-truth version gets the
# new name instead: TRUE_WORLD_D.
#
#   TRUE_WORLD_D -- fixed, ground truth, used only to sample the real start
#   D            -- belief, p(x); starts uncertain, gets replaced by a real
#                   posterior once an observation comes in (see Agent.D
#                   below -- it lives on each agent, since belief is
#                   something each agent holds for itself)
TRUE_WORLD_D = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # certain at Start

# --- A matrix (unchanged from Lab 1.4) ---
# Rows = observation, columns = true state (see Appendix B).
A = np.array([
    [0.90, 0.10, 0.00, 0.00, 0.00],
    [0.10, 0.80, 0.10, 0.10, 0.00],
    [0.00, 0.05, 0.80, 0.00, 0.10],
    [0.00, 0.05, 0.00, 0.80, 0.10],
    [0.00, 0.00, 0.10, 0.10, 0.80],
])

# --- B matrix is not used in this lab -- position is held fixed. ---
# (No movement -> no PREDICT step. See Week2_Lab_Arc_Outline_v2.md, and
# Appendix F for the full predict-vs-update discussion.)

# --------------------------------------------------------------------
# FROM MATRICES TO PROBABILITIES
# --------------------------------------------------------------------
# Four probability terms matter in this lab. Two of them we already HAVE,
# by construction. Two of them we CALCULATE. The Lab 2.1 Tutorial works
# through all four of these in real depth, including exactly why each one
# takes the shape it does -- this is the short version, for orientation.
#
# p(x)   -- belief over the hidden state. WE HAVE THIS: it's D, the whole
#           vector, directly. Whatever numbers are currently in D, that
#           IS p(x) right now -- nothing to compute. Length 5. Sums to 1.
#
# p(y|x) -- likelihood: given a true state, how likely is each possible
#           observation. WE HAVE THIS TOO: it's sitting inside A. A itself
#           is genuinely a MATRIX here -- a full set of five distributions,
#           one per possible true state x, sitting side by side as A's
#           five COLUMNS. Each column, on its own, is a valid distribution
#           over y: length 5, sums to 1 (that's how observe() uses it).
#
#           bayes_update() below needs the opposite slice -- a ROW of A,
#           fixed y, varying x. A row is still length 5, matching D's
#           size, but it does NOT sum to 1 in general (only columns are
#           guaranteed to). Try it: A's "Room A" row sums to 1.10; its
#           "Room B" row sums to 0.95. That's expected, not an error --
#           a row mixes one piece from each of five DIFFERENT column
#           distributions, so there's no reason those particular numbers
#           should add to anything in particular. The Lab 2.1 Tutorial
#           works through this in full; the short version is: this is
#           exactly why the division by evidence below is doing real
#           work, not just tidying up.
#
# p(y)   -- evidence: the overall probability of the one observation that
#           actually happened. WE CALCULATE THIS, from D and A together --
#           it's not stored anywhere on its own. A single scalar for the
#           one observation we actually got (the full p(y) vector, across
#           all five possible observations, would sum to 1 -- but that's
#           not what gets computed below; only the one entry that matters
#           gets calculated).
#
# p(x|y) -- posterior: the updated belief. WE CALCULATE THIS -- it's the
#           actual output of bayes_update(), and it becomes the new D.
#           Length 5. Sums to 1.
#
# The pattern going forward: p(x|y) computed this round becomes p(x) --
# the prior -- for the next one. That loop is what lets belief accumulate
# evidence over multiple observations, though this lab only takes the
# loop's first single step.
# --------------------------------------------------------------------


class Agent:
    """
    True position and belief are tracked as two genuinely separate
    variables -- a distinction this lab needs for the first time.
    Labs 1.2-1.6 never had to make it: the only thing worth tracking was
    the true position, so one distribution (D) was enough. Now that the
    agent also needs its own belief about that position, D and
    TRUE_WORLD_D have to be two different things, kept carefully apart.

    self.TRUE_WORLD_D : fixed. Copied in at construction, never touched
                        again. Used only by sample_position(), to decide
                        where the agent truly starts.
    self.D            : the agent's BELIEF -- p(x). Starts uncertain
                        (uniform, by default) and is REPLACED by a new
                        posterior every time bayes_update() runs. This is
                        the first lab where D is not fixed.
    """
    def __init__(self, TRUE_WORLD_D, initial_belief=None):
        self.TRUE_WORLD_D = TRUE_WORLD_D.copy()   # fixed, ground truth
        # Belief starts genuinely uncertain by default -- the whole premise
        # of a POMDP is that the agent does NOT know its true position, even
        # when we (the designers) do. Pass initial_belief explicitly only
        # for controlled demos/tests.
        if initial_belief is None:
            self.D = np.ones(len(TRUE_WORLD_D)) / len(TRUE_WORLD_D)  # uniform
        else:
            self.D = initial_belief.copy()
        self.position = 0
        self.history = []
        self.observations = []

    def sample_position(self):
        """Draw the agent's TRUE position from TRUE_WORLD_D (not belief)."""
        self.position = np.random.choice(len(self.TRUE_WORLD_D), p=self.TRUE_WORLD_D)
        self.history.append(self.position)
        return self.position

    def observe(self, A):
        """Draw a noisy observation from A, based on the TRUE position."""
        obs = np.random.choice(A.shape[0], p=A[:, self.position])
        self.observations.append(obs)
        return obs

    def bayes_update(self, A, obs):
        """
        THE NEW CONTENT OF THIS LAB. Turn one observation into a posterior
        belief, via Bayes' theorem:

            p(x|y) = p(y|x) * p(x) / p(y)

        p(y|x) : the ROW of A matching the observation we actually got --
                  not a column. See the "FROM MATRICES TO PROBABILITIES"
                  note above: this row does not sum to 1, which is exactly
                  why normalizing at the end of this method is necessary.
        p(x)   : self.D, the CURRENT belief (the prior going into this step).
        p(y)   : the evidence -- a single scalar, computed by summing the
                  elementwise (Hadamard) product of the row and the prior.
                  This is implicit vectorization: no for-loop appears here,
                  but numpy still touches every one of the 5 elements to
                  multiply them pairwise, and again to sum them -- the
                  looping just happens inside numpy, not in visible Python.
                  This is genuinely different from the explicit for-loops
                  in Appendix H, not just a hidden version of them.
                  Normalization first becomes necessary here -- Lab 1.6
                  explicitly did not need it; this lab does.
        """
        likelihood_row = A[obs, :]                       # p(y|x), one row
        unnormalized = likelihood_row * self.D            # elementwise product
        evidence = unnormalized.sum()                     # p(y) -- a scalar
        self.D = unnormalized / evidence                  # the posterior, p(x|y)
        return self.D, evidence

    def describe(self):
        print(f"  True positions:  {[nodes[i] for i in self.history]}")
        print(f"  Observations:    {[nodes[i] for i in self.observations]}")


# --- Minimal demo: one observation, one update ---
if __name__ == "__main__":
    agent = Agent(TRUE_WORLD_D)

    agent.sample_position()
    obs = agent.observe(A)

    print(f"True position:        {nodes[agent.position]}")
    print(f"Observation received: {nodes[obs]}")
    print()
    print(f"Prior belief D     (before update): {np.round(agent.D, 3)}")

    posterior, evidence = agent.bayes_update(A, obs)

    print(f"Posterior belief D (after update):  {np.round(posterior, 3)}")
    print(f"Evidence p(y) for this observation: {evidence:.3f}")
    print(f"Posterior sums to:                  {posterior.sum():.6f}")
