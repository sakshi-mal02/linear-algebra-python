# =============================================================================
# WEEK 4: Eigenvalues, Eigenvectors & Applications
# Topics: Determinants, eigenvalues, eigenvectors, PageRank, Markov chains
# TBIL Sections: GT1 – GT4, Appendix A (PageRank)
# =============================================================================

import numpy as np
from sympy import Matrix, symbols, det, factor, solve, eye, latex
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# PART 1: Determinants (GT1, GT2)
# -----------------------------------------------------------------------------

print("=" * 55)
print("PART 1: Determinants")
print("=" * 55)

# NumPy (float, fast)
A_np = np.array([[2., 1., 3.],
                 [0., -1., 4.],
                 [5., 2., 0.]])
print(f"\nMatrix A:\n{A_np}")
print(f"\ndet(A) [NumPy] = {np.linalg.det(A_np):.4f}")

# SymPy (exact, symbolic)
A_sym = Matrix([[2, 1, 3],
                [0, -1, 4],
                [5, 2, 0]])
print(f"det(A) [SymPy] = {A_sym.det()}")

# Geometric meaning: |det| = area/volume scaling factor
print("\n--- Geometric interpretation ---")
# 2x2 example: area of parallelogram spanned by columns
v1 = np.array([3, 1])
v2 = np.array([1, 2])
M = np.column_stack([v1, v2])
print(f"Columns v1={v1}, v2={v2}")
print(f"Area of parallelogram = |det| = {abs(np.linalg.det(M)):.2f}")

# Visualize
fig, ax = plt.subplots(figsize=(6, 6))
parallelogram = np.array([[0,0], v1, v1+v2, v2, [0,0]])
ax.fill(parallelogram[:,0], parallelogram[:,1], alpha=0.3, color='blue',
        label=f'Area = |det| = {abs(np.linalg.det(M)):.1f}')
ax.quiver(0,0,v1[0],v1[1], angles='xy',scale_units='xy',scale=1,color='red',width=0.015,label=f'v1={v1}')
ax.quiver(0,0,v2[0],v2[1], angles='xy',scale_units='xy',scale=1,color='green',width=0.015,label=f'v2={v2}')
ax.quiver(v1[0],v1[1],v2[0],v2[1],angles='xy',scale_units='xy',scale=1,color='green',width=0.015,alpha=0.4)
ax.quiver(v2[0],v2[1],v1[0],v1[1],angles='xy',scale_units='xy',scale=1,color='red',width=0.015,alpha=0.4)
ax.set_xlim(-0.5, 5.5); ax.set_ylim(-0.5, 4)
ax.axhline(0,color='k',lw=0.5); ax.axvline(0,color='k',lw=0.5)
ax.grid(True, alpha=0.3); ax.legend(); ax.set_aspect('equal')
ax.set_title("Determinant = Area of Parallelogram")
plt.tight_layout()
plt.savefig("week4_determinant_area.png", dpi=150, bbox_inches='tight')
print("Saved: week4_determinant_area.png")
plt.show()

# -----------------------------------------------------------------------------
# PART 2: Eigenvalues and Eigenvectors (GT3, GT4)
# -----------------------------------------------------------------------------

print("\n" + "=" * 55)
print("PART 2: Eigenvalues and Eigenvectors")
print("=" * 55)

# An eigenvector v satisfies A*v = lambda*v
# (A - lambda*I)*v = 0  => det(A - lambda*I) = 0  (characteristic polynomial)

A = Matrix([[3, 1],
            [0, 2]])

lam = symbols('lambda')
char_poly = (A - lam * eye(2)).det()
char_poly_factored = factor(char_poly)
print(f"\nMatrix A:\n{A}")
print(f"\nCharacteristic polynomial: det(A - λI) = {char_poly}")
print(f"Factored: {char_poly_factored}")

eigenvalues = solve(char_poly, lam)
print(f"Eigenvalues: λ = {eigenvalues}")

for lam_val in eigenvalues:
    print(f"\n--- Eigenspace for λ = {lam_val} ---")
    M = A - lam_val * eye(2)
    print(f"A - {lam_val}*I =\n{M}")
    rref, pivots = M.rref()
    print(f"RREF:\n{rref}")
    null_basis = M.nullspace()
    print(f"Eigenvectors (basis for eigenspace): {[list(v.T) for v in null_basis]}")

# NumPy (faster for numerical work)
print("\n--- NumPy eigenvalues and eigenvectors ---")
A_np = np.array([[3., 1.],
                 [0., 2.]])
eigenvalues_np, eigenvectors_np = np.linalg.eig(A_np)
print(f"Eigenvalues: {eigenvalues_np}")
print(f"Eigenvectors (columns):\n{eigenvectors_np}")

# Verify: A @ v = lambda * v
for i in range(len(eigenvalues_np)):
    lam_val = eigenvalues_np[i]
    v = eigenvectors_np[:, i]
    print(f"\nVerify λ={lam_val:.1f}: A@v = {A_np @ v}  |  λ*v = {lam_val * v}  |  Match: {np.allclose(A_np @ v, lam_val * v)}")

# Visualize eigenvectors
fig, ax = plt.subplots(figsize=(7, 7))
colors = ['red', 'blue']
for i in range(len(eigenvalues_np)):
    v = eigenvectors_np[:, i]
    lam_val = eigenvalues_np[i]
    Av = A_np @ v
    ax.quiver(0,0,v[0],v[1], angles='xy',scale_units='xy',scale=1,
              color=colors[i], width=0.02, label=f'Eigenvector (λ={lam_val:.1f})')
    ax.quiver(0,0,Av[0],Av[1], angles='xy',scale_units='xy',scale=1,
              color=colors[i], width=0.015, alpha=0.4, linestyle='dashed',
              label=f'A*eigenvec (=λ*v, scaled by {lam_val:.1f})')

ax.set_xlim(-2, 3); ax.set_ylim(-2, 3)
ax.axhline(0,color='k',lw=0.5); ax.axvline(0,color='k',lw=0.5)
ax.grid(True, alpha=0.3); ax.legend(fontsize=9); ax.set_aspect('equal')
ax.set_title("Eigenvectors of A: A*v = λ*v\n(eigenvectors only get scaled, not rotated)")
plt.tight_layout()
plt.savefig("week4_eigenvectors.png", dpi=150, bbox_inches='tight')
print("Saved: week4_eigenvectors.png")
plt.show()

# -----------------------------------------------------------------------------
# PART 3: Application -- PageRank (Appendix A.2)
# -----------------------------------------------------------------------------

print("\n" + "=" * 55)
print("PART 3: Application -- Simplified PageRank")
print("=" * 55)

# PageRank: the importance of each page is an eigenvector
# of the link matrix with eigenvalue 1.
#
# Small 4-page web:
#   Page 1 -> Pages 2, 3, 4
#   Page 2 -> Page 3
#   Page 3 -> Pages 1, 2
#   Page 4 -> Page 1

# Build the column-stochastic link matrix
# Column j = where page j's "votes" go (distributed equally)
L = np.array([
    [0,   0,   1/2, 1  ],   # page 1 receives from ...
    [1/3, 0,   1/2, 0  ],   # page 2
    [1/3, 1,   0,   0  ],   # page 3
    [1/3, 0,   0,   0  ]    # page 4
])

print("Link matrix L (column j = outbound links from page j):")
print(np.round(L, 3))

# PageRank = dominant eigenvector (eigenvalue = 1)
eigenvalues, eigenvectors = np.linalg.eig(L)
print(f"\nEigenvalues: {np.round(eigenvalues.real, 4)}")

# Find eigenvector for eigenvalue closest to 1
idx = np.argmin(np.abs(eigenvalues - 1.0))
pagerank = eigenvectors[:, idx].real
pagerank = pagerank / pagerank.sum()   # normalize so ranks sum to 1

print(f"\nPageRank scores (normalized):")
for i, pr in enumerate(pagerank):
    print(f"  Page {i+1}: {pr:.4f}  {'<-- most important' if pr == max(pagerank) else ''}")

# Bar chart
fig, ax = plt.subplots(figsize=(7, 4))
pages = [f"Page {i+1}" for i in range(4)]
bars = ax.bar(pages, pagerank, color=['#4285F4', '#EA4335', '#FBBC05', '#34A853'])
ax.set_ylabel("PageRank Score")
ax.set_title("PageRank: Eigenvector of Link Matrix")
for bar, score in zip(bars, pagerank):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig("week4_pagerank.png", dpi=150, bbox_inches='tight')
print("Saved: week4_pagerank.png")
plt.show()

# -----------------------------------------------------------------------------
# PART 4: YOUR TURN -- Exercises
# -----------------------------------------------------------------------------

print("\n" + "=" * 55)
print("PART 4: Exercises")
print("=" * 55)

print("""
Exercise 1:
  For A = [[4, 1], [2, 3]]:
  (a) Compute det(A) using both NumPy and SymPy.
  (b) Find the eigenvalues using the characteristic polynomial (SymPy).
  (c) Find the eigenvectors for each eigenvalue.

Exercise 2:
  Verify that if det(A) = 0, then A has 0 as an eigenvalue.
  Use A = [[1, 2], [2, 4]].
  What is the eigenvector for λ = 0?  What does it mean geometrically?

Exercise 3 (PageRank extension):
  Add a 5th page to the PageRank example above, with links:
    Page 5 -> Pages 2 and 4
    Page 1 -> also links to Page 5
  Update the link matrix and re-run PageRank.
  How does adding Page 5 change the ranking?

Exercise 4 (Challenge):
  The Fibonacci sequence can be expressed as a matrix power problem.
  Let A = [[1, 1], [1, 0]].  Show that A^n gives Fibonacci numbers.
  Use numpy.linalg.matrix_power(A, n) to compute the 20th Fibonacci number.
""")
