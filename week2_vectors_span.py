# =============================================================================
# WEEK 2: Vectors, Span, and Linear Independence
# Topics: Linear combinations, spanning sets, independence, basis
# TBIL Sections: EV1 – EV5
# =============================================================================

import numpy as np
from sympy import Matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------------------------------------------------------
# PART 1: Linear Combinations and Span (EV1, EV2)
# -----------------------------------------------------------------------------

print("=" * 55)
print("PART 1: Linear Combinations and Span")
print("=" * 55)

# Is b in span{v1, v2, v3}?
# Equivalent to: does c1*v1 + c2*v2 + c3*v3 = b have a solution?
# Set up augmented matrix [v1 | v2 | v3 | b] and check RREF

v1 = Matrix([1, 0, 2])
v2 = Matrix([0, 1, -1])
v3 = Matrix([2, 1, 3])
b  = Matrix([3, 2, 7])

aug = Matrix.hstack(v1, v2, v3, b)
print(f"\nAugmented matrix [v1 | v2 | v3 | b]:")
print(aug)

rref, pivots = aug.rref()
print(f"\nRREF:")
print(rref)

# Check: if last column (column 3) is a pivot => inconsistent => b NOT in span
if 3 in pivots:
    print("\nb is NOT in span{v1, v2, v3}  (last column is a pivot)")
else:
    print("\nb IS in span{v1, v2, v3}")

# -----------------------------------------------------------------------------
# PART 2: Visualizing Span in R^2 (EV2)
# -----------------------------------------------------------------------------

print("\n" + "=" * 55)
print("PART 2: Visualizing Span in R^2")
print("=" * 55)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Span of Vectors in R²", fontsize=14, fontweight='bold')

# --- Plot 1: Two linearly independent vectors (span = all of R^2) ---
ax = axes[0]
ax.set_title("Span{v1, v2} = R²  (independent)")
v1_2d = np.array([2, 1])
v2_2d = np.array([-1, 2])

# Draw a grid of linear combinations
for c1 in np.linspace(-2, 2, 15):
    for c2 in np.linspace(-2, 2, 15):
        pt = c1 * v1_2d + c2 * v2_2d
        ax.plot(pt[0], pt[1], 'b.', alpha=0.3, markersize=4)

ax.quiver(0, 0, v1_2d[0], v1_2d[1], angles='xy', scale_units='xy', scale=1,
          color='red', width=0.015, label=f'v1 = {v1_2d}')
ax.quiver(0, 0, v2_2d[0], v2_2d[1], angles='xy', scale_units='xy', scale=1,
          color='green', width=0.015, label=f'v2 = {v2_2d}')
ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
ax.grid(True, alpha=0.3); ax.legend(); ax.set_aspect('equal')

# --- Plot 2: Two dependent vectors (span = a line) ---
ax = axes[1]
ax.set_title("Span{v1, v2} = a line  (dependent)")
v1_dep = np.array([2, 1])
v2_dep = np.array([4, 2])   # = 2 * v1

t = np.linspace(-3, 3, 100)
line = np.outer(t, v1_dep)
ax.plot(line[:, 0], line[:, 1], 'b-', lw=2, alpha=0.5, label='span (line)')
ax.quiver(0, 0, v1_dep[0], v1_dep[1], angles='xy', scale_units='xy', scale=1,
          color='red', width=0.015, label=f'v1 = {v1_dep}')
ax.quiver(0, 0, v2_dep[0], v2_dep[1], angles='xy', scale_units='xy', scale=1,
          color='green', width=0.015, label=f'v2 = {v2_dep} = 2v1')
ax.set_xlim(-6, 6); ax.set_ylim(-4, 4)
ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
ax.grid(True, alpha=0.3); ax.legend(); ax.set_aspect('equal')

plt.tight_layout()
plt.savefig("week2_span_visualization.png", dpi=150, bbox_inches='tight')
print("Saved: week2_span_visualization.png")
plt.show()

# -----------------------------------------------------------------------------
# PART 3: Linear Independence (EV4)
# -----------------------------------------------------------------------------

print("\n" + "=" * 55)
print("PART 3: Checking Linear Independence")
print("=" * 55)

# A set {v1, ..., vk} is linearly independent iff
# the only solution to c1*v1 + ... + ck*vk = 0 is the trivial solution.
# Equivalently: RREF of [v1 | ... | vk] has a pivot in every column.

def check_independence(vectors, names=None):
    """Check if a list of column vectors is linearly independent."""
    M = Matrix(np.column_stack(vectors).tolist())
    rref, pivots = M.rref()
    n_cols = M.cols
    if names is None:
        names = [f"v{i+1}" for i in range(n_cols)]
    print(f"\nMatrix [{', '.join(names)}]:")
    print(M)
    print(f"RREF:\n{rref}")
    print(f"Pivot columns: {pivots}")
    if len(pivots) == n_cols:
        print(f"=> LINEARLY INDEPENDENT (pivot in every column)")
    else:
        free_cols = [i for i in range(n_cols) if i not in pivots]
        print(f"=> LINEARLY DEPENDENT (columns {free_cols} are free)")
    return len(pivots) == n_cols

# Example 1: Independent set
v1 = [1, 0, 0]
v2 = [0, 1, 0]
v3 = [0, 0, 1]
check_independence([v1, v2, v3], ["e1", "e2", "e3"])

# Example 2: Dependent set
v1 = [1, 2, 3]
v2 = [4, 5, 6]
v3 = [7, 8, 9]    # v3 = 2*v2 - v1
check_independence([v1, v2, v3])

# -----------------------------------------------------------------------------
# PART 4: Basis and Dimension (EV5, EV6)
# -----------------------------------------------------------------------------

print("\n" + "=" * 55)
print("PART 4: Finding a Basis from RREF")
print("=" * 55)

# The pivot columns of a matrix form a basis for its column space.
# Example: find a basis for span{v1, v2, v3, v4}

v1 = Matrix([1, 2, 0, -1])
v2 = Matrix([0, 1, 1,  2])
v3 = Matrix([2, 5, 1,  0])   # v3 = v1 + 2*v2 ?  Let's check!
v4 = Matrix([1, 0, -2, -5])

M = Matrix.hstack(v1, v2, v3, v4)
print("\nMatrix M = [v1 | v2 | v3 | v4]:")
print(M)

rref, pivots = M.rref()
print(f"\nRREF:\n{rref}")
print(f"Pivot columns: {pivots}")
print(f"Dimension of span = {len(pivots)}")
print(f"Basis vectors are the pivot columns: " +
      ", ".join([f"v{p+1}" for p in pivots]))

# -----------------------------------------------------------------------------
# PART 5: YOUR TURN -- Exercises
# -----------------------------------------------------------------------------

print("\n" + "=" * 55)
print("PART 5: Exercises")
print("=" * 55)

print("""
Exercise 1:  Is b = [3, -1, 5] in span{[1,0,2], [0,1,-1]}?
  Set up the augmented matrix and use RREF. Interpret the result.

Exercise 2:  Are these vectors linearly independent?
  v1 = [1, -1, 2],  v2 = [3, 1, 0],  v3 = [5, -1, 4]
  Use check_independence() above. If dependent, find a dependence relation.

Exercise 3:  Consider the matrix below. Find a basis for its column space
  and state the dimension.
  A = [[1, 0, 2, 3],
       [0, 1, -1, 2],
       [0, 0, 0, 0]]

Exercise 4 (Challenge):  Modify the visualization in Part 2 to show span
  in R^3 -- what geometric shape do you get with one vector? Two independent
  vectors? Three independent vectors?
""")
