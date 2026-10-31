"""
L2-3-F-Sandbox.py  --  a small, stand-alone sandbox for Lab 2.3 (Tutorial Part 2)
Themesis Active Inference Lab Course | Building Active Inference in Python
(c) 2026 Themesis, Inc. All rights reserved.

Score any belief you like, four ways, without opening the big lab file.
EDIT ONLY THE TWO LINES MARKED  <-- EDIT ME.  Then run:  python L2-3-F-Sandbox.py

Same world, same numbers as the lab: five rooms, a flat prior, the A matrix from Lab 1.4.
"""
import numpy as np

ROOMS = ["Start", "Room A", "Room B", "Room C", "Goal"]

# ---- edit these two lines -------------------------------------------------------------
reading = 1                          # <-- EDIT ME: 0 Start, 1 Room A, 2 Room B, 3 Room C, 4 Goal
q = np.array([0, 0.5, 0.5, 0, 0])    # <-- EDIT ME: your belief, one weight per room, adding to 1
# ----------------------------------------------------------------------------------------

# The model (do not edit): the prior p(x) and the A matrix. Row = reading, column = true room.
prior = np.full(5, 0.2)
A = np.array([[0.9, 0.1,  0.0, 0.0, 0.0],
              [0.1, 0.8,  0.1, 0.1, 0.0],
              [0.0, 0.05, 0.8, 0.0, 0.1],
              [0.0, 0.05, 0.0, 0.8, 0.1],
              [0.0, 0.0,  0.1, 0.1, 0.8]])

likelihood = A[reading]              # p(y|x): how often each room gives this reading
joint      = prior * likelihood      # p(x,y): the model, kept whole
evidence   = joint.sum()             # p(y): ONE number, now that the reading is in
posterior  = joint / evidence        # p(x|y): the target (Bayes, for checking only)

assert np.isclose(q.sum(), 1), "Your belief must add up to 1."
if np.any((q > 0) & (joint == 0)):
    bad = [ROOMS[i] for i in range(5) if q[i] > 0 and joint[i] == 0]
    print(f"F is infinite: your belief puts weight on {bad}, which cannot give this reading.")
    raise SystemExit

live = q > 0                         # rooms with zero belief add nothing to any sum
ln = np.log

G = np.sum(q[live] * ln(q[live] / joint[live]))                       # the definition: one sum
energy, entropy = -np.sum(q[live] * ln(joint[live])), -np.sum(q[live] * ln(q[live]))
complexity, accuracy = np.sum(q[live] * ln(q[live] / prior[live])), np.sum(q[live] * ln(likelihood[live]))
gap, surprisal = np.sum(q[live] * ln(q[live] / posterior[live])), -ln(evidence)

print(f"Reading: '{ROOMS[reading]}'.   Your belief q = {q}")
print(f"  G-form  (one sum)                 F = {G:.3f}")
print(f"  E-form  energy - entropy          {energy:.3f} - {entropy:.3f} = {energy - entropy:.3f}")
print(f"  C-form  complexity - accuracy     {complexity:.3f} - ({accuracy:.3f}) = {complexity - accuracy:.3f}")
print(f"  D-form  gap + surprisal           {gap:.3f} + {surprisal:.3f} = {gap + surprisal:.3f}")
print(f"  The floor (surprisal): {surprisal:.3f}.   The exact posterior: {np.round(posterior, 3)}")
