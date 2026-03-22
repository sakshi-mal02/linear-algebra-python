# =============================================================================
# WEEK 3: Linear Transformations and Matrices
# Topics: Standard matrices, image/kernel, injective/surjective
# TBIL Sections: AT1 – AT4, MX1 – MX2
# =============================================================================

import numpy as np
from sympy import Matrix
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

# -----------------------------------------------------------------------------
# PART 1: Applying a Linear Transformation (AT1, AT2)
# -----------------------------------------------------------------------------

print("=" * 55)
print("PART 1: Linear Transformations as Matrix Multiplication")
print("=" * 55)

# A linear transformation T: R^n -> R^m is given by T(x) = Ax
# where A is the standard matrix (m x n).

# Example: rotation by 45 degrees in R^2
theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

print(f"\nRotation matrix (45°):\n{np.round(R, 4)}")

x = np.array([1, 0])          # unit vector along x-axis
Tx = R @ x                    # apply transformation
print(f"\nT({x}) = {np.round(Tx, 4)}")   # should be [cos45, sin45]

# Visualize transformations on the unit square
def transform_and_plot(A, title="Linear Transformation"):
    """Visualize how matrix A transforms the unit square."""
    # corners of unit square
    square = np.array([[0, 1, 1, 0, 0],
                       [0, 0, 1, 1, 0]])
    # some interior points
    pts = np.random.rand(2, 50)

    T_square = A @ square
    T_pts = A @ pts

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    for ax, data, dot_data, t in zip(
            axes, [square, T_square], [pts, T_pts], ["Domain (input)", "Codomain (output)"]):
        ax.plot(data[0], data[1], 'b-', lw=2)
        ax.fill(data[0], data[1], alpha=0.15, color='blue')
        ax.plot(dot_data[0], dot_data[1], 'r.', alpha=0.5, markersize=4)
        ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
        ax.grid(True, alpha=0.3); ax.set_aspect('equal')
        ax.set_title(t)

    plt.tight_layout()
    fname = title.lower().replace(" ", "_") + ".png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.show()

# Rotation
transform_and_plot(R, "Rotation by 45 degrees")

# Shear
S = np.array([[1, 1],
              [0, 1]])
transform_and_plot(S, "Horizontal Shear")

# Projection onto x-axis
P = np.array([[1, 0],
              [0, 0]])
transform_and_plot(P, "Projection onto x-axis")

# -----------------------------------------------------------------------------
# PART 2: Image and Kernel (AT3)
# -----------------------------------------------------------------------------

print("\n" + "=" * 55)
print("PART 2: Image (Column Space) and Kernel (Null Space)")
print("=" * 55)

A = Matrix([
    [1,  2, -1,  3],
    [2,  4,  0,  8],
    [-1, -2, 3, -1]
])

print(f"\nMatrix A =\n{A}")

# --- Image = Column Space ---
rref, pivots = A.rref()
print(f"\nRREF of A:\n{rref}")
print(f"Pivot columns: {pivots}")
print(f"\nImage (column space) has dimension (rank) = {len(pivots)}")
print(f"Basis for image: columns {[p+1 for p in pivots]} of the ORIGINAL matrix A")
for p in pivots:
    print(f"  Column {p+1}: {list(A.col(p).T)}")

# --- Kernel = Null Space ---
nullspace_basis = A.nullspace()
print(f"\nKernel (null space) has dimension (nullity) = {len(nullspace_basis)}")
print(f"Basis for kernel:")
for i, v in enumerate(nullspace_basis):
    print(f"  Basis vector {i+1}: {list(v.T)}")

# Rank-Nullity theorem check
rank = len(pivots)
nullity = len(nullspace_basis)
n_cols = A.cols
print(f"\nRank-Nullity Check: rank({rank}) + nullity({nullity}) = {rank+nullity} == n_cols({n_cols}): {rank+nullity == n_cols}")

# -----------------------------------------------------------------------------
# PART 3: Injective and Surjective (AT4)
# -----------------------------------------------------------------------------

print("\n" + "=" * 55)
print("PART 3: Injective (One-to-One) and Surjective (Onto)")
print("=" * 55)

def analyze_transformation(A_np, name="T"):
    """Determine injectivity and surjectivity of T(x) = Ax."""
    A = Matrix(A_np.tolist())
    rref, pivots = A.rref()
    m, n = A.shape
    rank = len(pivots)

    print(f"\nTransformation {name}: R^{n} -> R^{m}")
    print(f"Matrix:\n{A_np}")
    print(f"Rank = {rank}")

    injective  = (rank == n)   # pivot in every column => trivial kernel
    surjective = (rank == m)   # pivot in every row    => image = codomain

    print(f"Injective  (one-to-one)? {'YES  -- kernel is trivial (only zero vector)' if injective  else 'NO   -- kernel has free variables'}")
    print(f"Surjective (onto)?       {'YES  -- image is all of R^' + str(m) if surjective else 'NO   -- image does not fill R^' + str(m)}")
    if injective and surjective:
        print(f"=> {name} is an ISOMORPHISM (bijective). The matrix is invertible.")

# Example 1: Invertible (square, full rank)
A1 = np.array([[2, 1],
               [5, 3]], dtype=float)
analyze_transformation(A1, "T1")

# Example 2: Not injective (more columns than rows)
A2 = np.array([[1, 2, 3],
               [0, 1, -1]], dtype=float)
analyze_transformation(A2, "T2")

# Example 3: Not surjective (more rows than columns)
A3 = np.array([[1, 0],
               [0, 1],
               [1, 1]], dtype=float)
analyze_transformation(A3, "T3")

# -----------------------------------------------------------------------------
# PART 4: Matrix Multiplication and Inverse (MX1, MX2)
# -----------------------------------------------------------------------------

print("\n" + "=" * 55)
print("PART 4: Matrix Multiplication and Inverses")
print("=" * 55)

A = np.array([[2., 1.],
              [5., 3.]])

B = np.array([[1., 0.],
              [0., 2.]])

print(f"\nA =\n{A}")
print(f"B =\n{B}")
print(f"\nA @ B (matrix product) =\n{A @ B}")
print(f"\nB @ A (note: order matters!) =\n{B @ A}")

# Matrix inverse
A_inv = np.linalg.inv(A)
print(f"\nA^(-1) =\n{np.round(A_inv, 4)}")
print(f"\nA @ A^(-1) (should be identity) =\n{np.round(A @ A_inv, 4)}")

# Using SymPy for exact inverse
A_sym = Matrix(A.tolist())
print(f"\nExact inverse (SymPy):\n{A_sym.inv()}")

# -----------------------------------------------------------------------------
# PART 5: YOUR TURN -- Exercises
# -----------------------------------------------------------------------------

print("\n" + "=" * 55)
print("PART 5: Exercises")
print("=" * 55)

print("""
Exercise 1:
  Let T: R^3 -> R^2 be defined by T([x1,x2,x3]) = [x1+x2, x2-x3].
  (a) Write the standard matrix A for T.
  (b) Compute T([1, -1, 2]).
  (c) Is T injective? Surjective? Explain using RREF.

Exercise 2:
  For the matrix A = [[1,3,-1],[2,7,0],[0,1,2]], find:
  (a) The image (column space) -- give a basis.
  (b) The kernel (null space) -- give a basis.
  (c) Verify the Rank-Nullity theorem.

Exercise 3:
  Use transform_and_plot() to visualize the transformation given by:
  A = [[2, 0], [0, 0.5]]  (scaling)
  What does this do geometrically to the unit square?

Exercise 4:
  Find A^(-1) for A = [[3, 1], [5, 2]] using both numpy and SymPy.
  Verify that A @ A^(-1) = I.
""")
