# Themesis, Inc. -- Building Active Inference in Python
# Lab 2.3 -- Variational Free Energy: Inference as Minimizing a Number
# v1.0 (2026-10-13)
#
# In Labs 2.1 and 2.2 the agent found its belief (where it is in the five-room
# world) by calculation: PREDICT, then UPDATE with Bayes' rule. That worked
# because this world is tiny: we could sum over all five rooms exactly.
#
# In this lab the agent finds the SAME belief a different way: it holds a
# candidate belief, q(x), scores it with one number (the variational free
# energy, F), and adjusts q(x) to make F smaller. When F stops falling, q(x)
# IS the posterior.
#
# Notice where q(x) lives. x is the HIDDEN (latent) variable: which room the
# agent is in, something it can never observe directly. q(x) is a belief
# about x, with one weight for each of the five rooms. All of the agent's
# work happens in this hidden space; it never sees x, only the reading y
# that x produces.
#
# KEY INSIGHT:
#   A reminder from Lab 2.1. The PRIOR, p(x), is the agent's belief about the
#   five rooms BEFORE the reading (in the code, self.D after PREDICT). The
#   POSTERIOR, p(x|y), is its belief AFTER the reading y, found with the
#   A matrix. Bayes' rule gave us the posterior as one definite answer.
#   Minimizing F is a second way to arrive at a belief. In this tiny world q
#   can be ANY belief over the five rooms, so the belief with the lowest F
#   turns out to be that same posterior. Same answer, found by searching
#   instead of by calculating.
#
#   Why bother, if Bayes already works? In a bigger world we cannot sum over
#   every state, so q is restricted to a simpler form (for example, a product
#   of independent pieces). Minimizing F then gives the BEST APPROXIMATION to
#   the posterior that form allows. That approximation is the whole reason
#   active inference uses free energy. We practice it here, in a world small
#   enough to check against the exact answer.
#
#     F(q) = sum_x q(x) [ ln q(x) - ln p(x) - ln p(y|x) ]
#
# WHERE EACH PIECE COMES FROM (left to right):
#   x       which room: the HIDDEN (latent) variable, the thing the model is
#           about and the agent never observes directly. Inside the sum,
#           x takes each room's value in turn: Start, A, B, C, Goal. It
#           does two jobs at once: it NAMES a room,
#           and it is that room's POSITION in every array (q[x], prior[x]).
#           "sum_x" means: consider each of the five rooms the agent's model
#           allows, as a "what if I were here?" The agent's real room is not
#           singled out anywhere in this equation. (The designer chose the
#           list of five; the agent does not discover it.)
#   q(x)    the belief being TESTED: five weights, one per room, adding to 1.
#           Like p(x) and p(x|y), it is a list (numpy array) of five numbers,
#           the same shape as they are, so we can compare them directly.
#           Think of the true-room vector [0, 0, 1, 0, 0] ("in Room B, for
#           sure"). q is a SOFT version of that: weights instead of one 1.
#   p(x)    the PRIOR: the belief the agent held BEFORE the reading
#           (after PREDICT).
#   y       the reading (observation) the agent just received.
#   p(y|x)  how likely this reading (y) would be IF the agent were in room x
#           (read from the A matrix).
#
#   Together p(x) and p(y|x) are a two-step GENERATIVE STORY: first pick a
#   room (p(x)), then generate a reading from that room (p(y|x)). F measures
#   how well q matches that story once the reading has arrived.
#
# Reading the equation in English:
#   The model scores each room as p(x) * p(y|x): how plausible the room was
#   beforehand, times how well it explains the reading.
#   F asks, room by room: does q give this room MORE weight than that score
#   supports, or LESS? Overshoots push F up; undershoots pull it down.
#   Each room counts in proportion to q's weight on it. Add up the five.
#   But q's weights must add to 1, so q cannot undershoot every room: weight
#   taken from one room has to go to another.
#   F is lowest when q's weights match the model's scores. That match is the
#   Bayes posterior.
#
# THREE WAYS TO WRITE THE SAME F  (algebra, not new ideas; the tutorial
# derives each one). Labels follow Namjoshi (2026, Chpt. 4, Eqns. 4.27-4.29).
# H(q) is the entropy of q:  H(q) = -sum_x q(x) ln q(x)  (never negative).
#
#   E-form (4.29):  F = E_q[ -ln p(x,y) ]  -  H(q)
#                       "average energy"      "entropy"
#       In T3 this is F = H - S: the average-energy term plays the role of
#       the enthalpy (T3's H), and entropy is T3's S. Here we write the
#       entropy as H(q), so in this course the letter H always means
#       entropy. (The E in E_q[...] is the expectation under q, not energy.)
#
#   A word on "KL" before the next two forms. KL[ q || p ] is the KL
#   divergence, a single number measuring how far belief q has drifted from
#   belief p:  KL[q || p] = sum_x q(x) ln( q(x) / p(x) ). The pair of
#   parallel lines || is just notation that separates the two beliefs being
#   compared; it is not an operation. Order matters (KL[q||p] differs from
#   KL[p||q]). KL is 0 when q = p, and never negative.
#   See also: Maren (2024), "Minding Your P's and Q's: Notational Variations
#   Expressing the Kullback-Leibler Divergence," Themesis TN2024-001:
#   https://themesis.com/Downloads/Kullback-Leibler-Notational-Variations-Themesis-upload-2024-08-11.pdf
#
#   All three forms (E, C and D) are algebraic rearrangements of one
#   source equation, the definition of F above. (Friston 2013, "Life as We
#   Know It," Eqns. 2.7 and 2.8, first presents the key active inference
#   equations. J. R. Soc. Interface 10(86): 20130475.)
#   https://royalsocietypublishing.org/rsif/article/10/86/20130475/35150/Life-as-we-know-itLife-as-we-know-it
#
#   C-form (4.28):  F = KL[ q(x) || p(x) ]  -  E_q[ ln p(y|x) ]
#                       "complexity"            "accuracy"
#   D-form (4.27):  F = KL[ q(x) || p(x|y) ]  +  ( -ln p(y) )
#                       "divergence"             "surprisal"
#
# Because KL is never negative, the D-form shows F can never go below the
# surprisal -ln p(y): that floor is Namjoshi's Eqn. 4.26, F >= -ln p(y).
# (Namjoshi writes integrals because his states are continuous; our five
# rooms are discrete, so we sum.)
#
# SYMBOL KEY (code name for each symbol)
#   x         each room the model allows (the index we sum over);
#             NOT self.position (the true room)
#   y         the observation                       obs
#   p(x)      the PRIOR: belief before seeing y     prior (= self.D after PREDICT)
#   q(x)      the CANDIDATE belief being scored     q
#   p(y|x)    the observation model                 A  (row A[obs, :])
#   p(x|y)    the exact posterior (Bayes)           what bayes_update() returns
#   F         variational free energy               free_energy(q, prior, A, obs)
#   H(q)      entropy of the belief q               entropy(q)
#   p(x,y)    the JOINT: the model's score for each room, p(x) p(y|x).
#             Five numbers; they sum to p(y), not to 1   joint
#   p(y)      the evidence: the probability of the one
#             reading that arrived, so a single number  evidence
#
# Architecture note (unchanged): TRUE_START_D, A and B are fixed, standalone
# objects, passed INTO the methods that need them. The agent carries only what
# changes: self.position, self.D, and its history lists.

import numpy as np

np.random.seed(7)   # remove this line to get a different run every time

# --- The five-node world (unchanged from Labs 1.2-2.2) ---
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

TRUE_START_D = np.array([1.0, 0.0, 0.0, 0.0, 0.0])   # certain at Start

A = np.array([
    [0.90, 0.10, 0.00, 0.00, 0.00],
    [0.10, 0.80, 0.10, 0.10, 0.00],
    [0.00, 0.05, 0.80, 0.00, 0.10],
    [0.00, 0.05, 0.00, 0.80, 0.10],
    [0.00, 0.00, 0.10, 0.10, 0.80],
])

B = np.zeros((num_nodes, num_nodes, 1))
for i in range(num_nodes):
    n = neighbors[i]
    if len(n) > 0:
        prob = 1.0 / len(n)
        for j in n:
            B[j, i, 0] = prob


def entropy(p):
    """Shannon entropy in nats (as in Lab 2.2)."""
    p = p[p > 0]
    return max(0.0, -(p * np.log(p)).sum())


# ====================================================================
# NEW IN THIS LAB: scoring a belief
# ====================================================================
# free_energy() computes F straight from the definition (the source equation
# in the header), as one sum over the rooms. It is the working function: the
# minimizers below call it over and over. The three forms are regroupings of
# this same sum (see free_energy_forms next).
def free_energy(q, prior, A, obs):
    """
    Variational free energy F of a candidate belief q, in nats.

        F = sum_x q(x) [ ln q(x) - ln p(x) - ln p(y|x) ]

    Lower is better. If q puts any weight on a room that the model says is
    IMPOSSIBLE (prior = 0, or that room can never produce this observation),
    F is infinite: you cannot believe in what your own model rules out.
    Rooms with q = 0 contribute nothing (0 * ln 0 is taken as 0).
    """
    likelihood_row = A[obs, :]                 # p(y|x): one row of A
    joint = prior * likelihood_row             # p(x,y): 5 scores, sum = p(y)
    if np.any((q > 0) & (joint == 0)):
        return np.inf
    has_weight = q > 0                         # rooms where q puts any weight
    # for each room: q * (ln q - ln score); then add the rooms up
    return float((q[has_weight]
                  * (np.log(q[has_weight]) - np.log(joint[has_weight]))).sum())


# free_energy_forms() computes the SAME F three ways (E-, C- and D-form),
# each a regrouping of the sum in free_energy(). It exists for teaching: it
# shows that every regrouping gives the same number. The minimizers do not
# use it.
def free_energy_forms(q, prior, A, obs):
    """
    The same F computed three ways (Namjoshi's E-, C- and D-forms).
    Returns (E_form, C_form, D_form); they must agree.
    Defined only where q is allowed (q = 0 wherever prior * likelihood = 0).
    """
    likelihood_row = A[obs, :]
    joint = prior * likelihood_row
    evidence = joint.sum()                      # p(y), one number
    has_weight = q > 0                         # rooms where q puts any weight
    # E-form: average energy minus entropy
    average_energy = float((-q[has_weight]
                            * np.log(joint[has_weight])).sum())
    E_form = average_energy - entropy(q)
    # C-form: complexity minus accuracy
    complexity = float((q[has_weight] * (np.log(q[has_weight])
                                         - np.log(prior[has_weight]))).sum())
    accuracy = float((q[has_weight]
                      * np.log(likelihood_row[has_weight])).sum())
    C_form = complexity - accuracy
    # D-form: divergence from the exact posterior plus surprisal
    posterior = joint / evidence
    divergence = float((q[has_weight]
                        * (np.log(q[has_weight])
                           - np.log(posterior[has_weight]))).sum())
    D_form = divergence + (-np.log(evidence))
    return E_form, C_form, D_form


def free_energy_parts(q, prior, A, obs):
    """
    The same F, split the second way:  F = complexity - accuracy.
      complexity = KL[q || p(x)]      how far q has moved from the prior
      accuracy   = E_q[ln p(y|x)]     how well q's rooms explain the reading
    Returns (F, complexity, accuracy). Defined only where q is allowed.

    (The same C-form quantities appear in free_energy_forms(). Repeated on
    purpose, so that each function reads on its own.)
    """
    likelihood_row = A[obs, :]
    has_weight = q > 0                         # rooms where q puts any weight
    complexity = float((q[has_weight] * (np.log(q[has_weight])
                                         - np.log(prior[has_weight]))).sum())
    accuracy = float((q[has_weight]
                      * np.log(likelihood_row[has_weight])).sum())
    return complexity - accuracy, complexity, accuracy


# ====================================================================
# TWO ROADS DOWN THE BOWL. Both find the q that minimizes F; both reach
# the same place. First pass? Read the GENTLE road (next to this one) first
# and skip this one; come back when you want the gradient itself.
# ====================================================================
def minimize_free_energy(prior, A, obs, steps=5000, lr=2.0):
    """
    MAIN ROAD: find q(x) by gradient descent on F. It never divides by p(y).

    q is kept as a proper belief by writing it as a softmax of free numbers s:
        q = exp(s) / sum(exp(s))
    Rooms the model rules out (prior * likelihood = 0) are held at q = 0.
    Start from a flat q over the allowed rooms and nudge s downhill on F.
    (The starting q is only a guess, like the flat D in Lab 1.3. The search
    pulls it toward the answer.)
    Returns q and the list of F values, one per step (so we can watch F fall).
    """
    joint = prior * A[obs, :]
    allowed = joint > 0
    s = np.zeros(len(prior))
    s[~allowed] = -np.inf                       # exp(-inf) = 0: these rooms get q = 0
    history = []
    for _ in range(steps):
        q = np.exp(s - s[allowed].max()); q = q / q.sum()
        history.append(free_energy(q, prior, A, obs))
        g = np.zeros(len(prior))
        g[allowed] = np.log(q[allowed]) - np.log(joint[allowed])   # dF/dq (up to a constant)
        grad_s = q * (g - (q * g).sum())                            # chain rule through softmax
        s[allowed] -= lr * grad_s[allowed]
    q = np.exp(s - s[allowed].max()); q = q / q.sum()
    history.append(free_energy(q, prior, A, obs))
    return q, history


def minimize_free_energy_damped(prior, A, obs, step=0.5, steps=30):
    """
    GENTLE ROAD: the same minimization, with no softmax and no gradient.

    Each step moves q PART of the way toward prior * likelihood (the
    unnormalized posterior) and renormalizes. In log form:

        ln q_new = (1 - step) * ln q_old  +  step * ln( p(x) p(y|x) )

    step = 1 jumps straight to the Bayes answer in one move. A smaller step
    (0.5 here) walks there: F falls on every step and never rises.
    Rooms the model rules out are held at q = 0, exactly as in the main road.
    Returns q and the list of F values, one per step.
    """
    joint = prior * A[obs, :]
    allowed = joint > 0
    q = np.where(allowed, 1.0, 0.0)
    q = q / q.sum()                              # start flat over the allowed rooms
    # (As in the main road, this starting q is only a guess, like the flat D
    # in Lab 1.3. The steps below pull it toward the answer.)
    history = [free_energy(q, prior, A, obs)]
    for _ in range(steps):
        new = np.zeros(len(prior))
        new[allowed] = np.exp((1 - step) * np.log(q[allowed]) + step * np.log(joint[allowed]))
        q = new / new.sum()
        history.append(free_energy(q, prior, A, obs))
    return q, history


class Agent:
    """Same agent as Lab 2.2. Carries only what CHANGES (see Lab 2.2)."""
    def __init__(self, num_states, initial_belief=None):
        if initial_belief is None:
            self.D = np.ones(num_states) / num_states
        else:
            self.D = initial_belief.copy()
        self.position = 0
        self.history = []
        self.observations = []

    # ---- WORLD SIDE (simulator) ----
    def sample_position(self, TRUE_START_D):
        self.position = np.random.choice(len(TRUE_START_D), p=TRUE_START_D)
        self.history.append(self.position)
        return self.position

    def act(self, B, action=0):
        self.position = np.random.choice(B.shape[0], p=B[:, self.position, action])
        self.history.append(self.position)
        return self.position

    def observe(self, A):
        obs = np.random.choice(A.shape[0], p=A[:, self.position])
        self.observations.append(obs)
        return obs

    # ---- AGENT SIDE (never reads self.position) ----
    def predict(self, B, action=0):
        self.D = B[:, :, action] @ self.D
        return self.D

    def bayes_update(self, A, obs):
        """Lab 2.1's exact update. Kept here as the ANSWER KEY to check against."""
        unnormalized = A[obs, :] * self.D
        evidence = unnormalized.sum()
        self.D = unnormalized / evidence
        return self.D, evidence

    # ---------------- NEW IN THIS LAB ----------------
    def update_by_free_energy(self, A, obs, steps=5000):
        """
        UPDATE, the second way: instead of dividing by the evidence, minimize F.
        The result replaces self.D, exactly as bayes_update() does.
        """
        q, trace = minimize_free_energy(self.D, A, obs, steps=steps)
        self.D = q
        return q, trace


def show(label, p):
    print(f"    {label:<28}{np.round(p, 3)}   H = {entropy(p):.2f} nats")


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # PART 1: score some candidate beliefs for ONE observation
    # ----------------------------------------------------------------
    # Setup: the unlucky-start story from the Lab 2.1 and 2.2 tutorials. The
    # agent has a flat prior (its belief D = [0.2, 0.2, 0.2, 0.2, 0.2], one
    # fifth per room) and the sensor says "Room A". The tutorial for this lab
    # develops the example further.
    #
    # Every number in Part 1 (F, complexity, accuracy, surprisal) belongs to
    # THIS one instance: the very first reading, before any move, with
    #   - this prior: the flat belief D = [0.2, 0.2, 0.2, 0.2, 0.2] set just
    #     below (the agent's belief before it has seen any reading), and
    #   - this A matrix: the observation model defined near the top of this
    #     file (the same A as Lab 1.4), read here at the row for "Room A".
    # These are the upstream inputs. F, complexity, accuracy and surprisal are
    # all downstream of them. A different prior or reading would give
    # different values. F is always the score of a belief for one reading.
    prior = np.ones(num_nodes) / num_nodes               # flat: no idea where
    obs = 1                                              # reading: "Room A"

    unnorm = A[obs, :] * prior
    evidence = unnorm.sum()
    posterior = unnorm / evidence                        # Bayes, for the answer key

    print("Lab 2.3 -- Variational Free Energy\n")
    print("PART 1: scoring candidate beliefs   (flat prior; reading = 'Room A')")
    print(f"  prior              : {np.round(prior, 3)}")
    print(f"  exact posterior    : {np.round(posterior, 4)}")
    print(f"  surprisal -ln p(y) : {-np.log(evidence):.4f}   <- F's lowest possible value\n")

    # These candidates are our own examples, chosen to show different ways a
    # belief can go wrong. ANY five weights that sum to 1 is a candidate q;
    # there are endlessly many. That we list five here is a coincidence (it
    # is not the five rooms). Each candidate gets ONE score, F.
    candidates = {
        "the exact posterior": posterior,                          # the Bayes answer
        "flat over 4 live rooms": np.array([0.25, 0.25, 0.25, 0.25, 0.0]),  # no opinion
        "sure it's Start (true room)": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),  # true, but the reading does not support it
        "sure it's Room B": np.array([0.0, 0.0, 1.0, 0.0, 0.0]),   # confident, and wrong
        "half Room A, half Goal": np.array([0.0, 0.5, 0.0, 0.0, 0.5]),  # Goal is impossible here
    }
    print(f"  {'candidate q':<30}{'F (nats)':>10}{'complexity':>12}{'accuracy':>10}")
    for name, q in candidates.items():
        F = free_energy(q, prior, A, obs)
        if np.isinf(F):
            print(f"  {name:<30}{'inf':>10}")
        else:
            _, c, a = free_energy_parts(q, prior, A, obs)
            print(f"  {name:<30}{F:>10.3f}{c:>12.3f}{a:>10.3f}")

    # The same F, three ways. Each column is a different rearrangement of the
    # same quantity, so every row must show the same number three times.
    print("\n  Same F, three ways (E-form, C-form, D-form):")
    print(f"  {'candidate q':<30}{'E-form':>9}{'C-form':>9}{'D-form':>9}")
    for name in ("the exact posterior", "flat over 4 live rooms", "sure it's Room B"):
        q = candidates[name]
        if np.isinf(free_energy(q, prior, A, obs)):
            continue
        Ef, Cf, Df = free_energy_forms(q, prior, A, obs)
        print(f"  {name:<30}{Ef:>9.3f}{Cf:>9.3f}{Df:>9.3f}")
        assert abs(Ef - Cf) < 1e-9 and abs(Cf - Df) < 1e-9, "the three forms must agree"
    print(f"  (floor: surprisal -ln p(y) = {-np.log(evidence):.3f}; F >= this always)")

    # ----------------------------------------------------------------
    # PART 2: let the agent FIND the best q by minimizing F
    # ----------------------------------------------------------------
    q_min, trace = minimize_free_energy(prior, A, obs)
    q_gen, trace_gen = minimize_free_energy_damped(prior, A, obs)
    print("\nPART 2: minimizing F two ways (neither road divides by the evidence)")
    print(f"  {'step':>6}{'main road (softmax)':>24}{'gentle road (damped)':>24}")
    for k in (0, 1, 2, 5, 10, 30):
        print(f"  {k:>6}{trace[k]:>24.4f}{trace_gen[k]:>24.4f}")
    print(f"  floor (surprisal)   {-np.log(evidence):>20.4f}")
    print(f"  q, main road   : {np.round(q_min, 4)}")
    print(f"  q, gentle road : {np.round(q_gen, 4)}")
    print(f"  exact Bayes    : {np.round(posterior, 4)}")
    print(f"  largest difference from Bayes: main {np.abs(q_min - posterior).max():.2e}"
          f",  gentle {np.abs(q_gen - posterior).max():.2e}")
    assert np.abs(q_gen - posterior).max() < 1e-8, "gentle road must reach the Bayes answer"

    # ----------------------------------------------------------------
    # PART 3: the whole Lab 2.2 loop again, UPDATE done by free energy
    # ----------------------------------------------------------------
    NUM_STEPS = 6
    free = Agent(num_nodes)                  # UPDATE by minimizing F
    exact = Agent(num_nodes)                 # UPDATE by Bayes (answer key)
    free.sample_position(TRUE_START_D)
    exact.position = free.position

    print("\nPART 3: PREDICT then (minimize F) over multiple steps")
    obs = free.observe(A)
    exact.bayes_update(A, obs)
    free.update_by_free_energy(A, obs)
    print(f"t=0  true: {nodes[free.position]:<7} observed: {nodes[obs]}")
    show("free-energy belief:", free.D)
    show("Bayes belief:", exact.D)

    worst = 0.0
    for t in range(1, NUM_STEPS):
        true_pos = free.act(B)
        exact.position = true_pos
        free.predict(B)
        exact.predict(B)
        obs = free.observe(A)
        q, trace = free.update_by_free_energy(A, obs)
        post, evidence = exact.bayes_update(A, obs)
        worst = max(worst, np.abs(q - post).max())
        print(f"\nt={t}  true: {nodes[true_pos]:<7} observed: {nodes[obs]}"
              f"   F: {trace[0]:.3f} -> {trace[-1]:.3f}  (-ln p(y) = {-np.log(evidence):.3f})")
        show("free-energy belief:", free.D)
        show("Bayes belief:", exact.D)

    print("\nChecks")
    print(f"  largest gap between the two methods over all steps: {worst:.2e}")
    print(f"  true path: {[nodes[i] for i in free.history]}")
