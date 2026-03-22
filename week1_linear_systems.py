# =============================================================================
# WEEK 1: Python & NumPy for Linear Algebra
# Topics: Arrays, Augmented Matrices, Row Reduction (RREF)
# TBIL Sections: LE1, LE2, LE3, LE4
# =============================================================================

import numpy as np
from sympy import Matrix, symbols, latex

# -----------------------------------------------------------------------------
# PART 1: Creating Matrices and Vectors
# -----------------------------------------------------------------------------

print("=" * 55)
print("PART 1: Matrices and Vectors in NumPy")
print("=" * 55)

# A vector in R^3
v = np.array([2, -1, 4])
print(f"\nVector v = {v}")

# A matrix (2D array)
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print(f"\nMatrix A =\n{A}")

# Augmented matrix [A | b]
b = np.array([3, 0, -2])
augmented = np.column_stack([A, b])  # stack b as last column
print(f"\nAugmented matrix [A | b] =\n{augmented}")

# Access rows and columns
print(f"\nRow 0 of A: {A[0, :]}")
print(f"Column 1 of A: {A[:, 1]}")
print(f"Entry A[1,2] = {A[1, 2]}")

# -----------------------------------------------------------------------------
# PART 2: Vector Operations (Linear Combinations)
# -----------------------------------------------------------------------------

print("\n" + "=" * 55)
print("PART 2: Vector Operations")
print("=" * 55)

u = np.array([1, -2, 3])
v = np.array([4, 0, -1])

print(f"\nu = {u}")
print(f"v = {v}")
print(f"u + v = {u + v}")
print(f"3*u = {3 * u}")
print(f"2*u - v = {2*u - v}")   # linear combination

# Check if a vector b is in the span of u and v:
# Is b = c1*u + c2*v solvable?
# This becomes a linear system -- see Part 3

# -----------------------------------------------------------------------------
# PART 3: RREF using SymPy
# -----------------------------------------------------------------------------

print("\n" + "=" * 55)
print("PART 3: Row Reduction (RREF) with SymPy")
print("=" * 55)

# SymPy gives exact (symbolic) RREF -- no floating point errors!

# Example from TBIL LE2:
#   x + 2y + z  = 3
#  -x -  y + z  = 1
#  2x + 5y + 3z = 7

aug = Matrix([
    [1,  2, 1,  3],
    [-1, -1, 1, 1],
    [2,  5, 3,  7]
])

print(f"\nOriginal augmented matrix:")
print(aug)

rref_matrix, pivot_cols = aug.rref()
print(f"\nRREF:")
print(rref_matrix)
print(f"Pivot columns: {pivot_cols}")

# -----------------------------------------------------------------------------
# PART 4: Solving a Linear System
# -----------------------------------------------------------------------------

print("\n" + "=" * 55)
print("PART 4: Solving Ax = b with NumPy")
print("=" * 55)

A = np.array([[2.0, 1.0, -1.0],
              [-3.0, -1.0, 2.0],
              [-2.0, 1.0, 2.0]])

b = np.array([8.0, -11.0, -3.0])

x = np.linalg.solve(A, b)
print(f"\nSolution x = {x}")

# Verify: Ax should equal b
print(f"\nVerification A @ x = {A @ x}")
print(f"Expected b      = {b}")
print(f"Match: {np.allclose(A @ x, b)}")

# -----------------------------------------------------------------------------
# PART 5: YOUR TURN -- Exercises
# -----------------------------------------------------------------------------

print("\n" + "=" * 55)
print("PART 5: Exercises")
print("=" * 55)

print("""
Exercise 1:
  Create the augmented matrix for the system:
     x1 + 3*x3 = 3
     3*x1 - 2*x2 + 4*x3 = 0
     -x2 + x3 = -2
  Then use SymPy to find its RREF.

Exercise 2:
  Let u = [1, 2, -1] and v = [0, -3, 4].
  Compute: 3u - 2v

Exercise 3:
  Is b = [5, 1, 3] in the span of u and v from Exercise 2?
  Set up the augmented matrix [u | v | b] and find its RREF.
  What does the RREF tell you?

Exercise 4:
  Solve the system:
     2x1 + x2 - x3 = 8
    -3x1 - x2 + 2x3 = -11
    -2x1 + x2 + 2x3 = -3
  using numpy.linalg.solve, then verify your answer.
""")

# =============================================================================
# STARTER CODE for Exercise 3 -- uncomment and complete
# =============================================================================
# u = np.array([1, 2, -1])
# v = np.array([0, -3, 4])
# b = np.array([5, 1, 3])
#
# aug = Matrix(np.column_stack([u, v, b]).tolist())
# rref_matrix, pivots = aug.rref()
# print(rref_matrix)
# # If the last column has a pivot => b is NOT in span(u, v)
# # If it doesn't => b IS in span(u, v)
