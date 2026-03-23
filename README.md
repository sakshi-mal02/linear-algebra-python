🧮 Applications in Linear Algebra — Python Lab
Spring 2026 | Powered by Python, NumPy & SymPy

Welcome! This page is your home for all Python lab materials used in our course.
Every notebook runs directly in your browser — no installation, no setup, nothing to download.
If you have never written a line of code before, that is completely fine.
We start from zero and build up gradually, 10–15 minutes at a time.

----
🚀 Getting Started
First time here? Start with the Python Basics notebook below before Week 1.

- Click any Open in Colab button below
- Sign in with your Google account
- Click File → Save a copy in Drive at the top — this saves your own editable copy
- Run cells one at a time with Shift + Enter
- That's it. You're coding. 🎉


----
Pre-Lab: Python Basics

Do this before the first lab session.
Covers variables, lists, NumPy arrays, and SymPy in about 5–10 minutes.
If you already know Python, feel free to skip it.

Running code in a notebook
- Variables, lists, and basic arithmetic
- NumPy arrays as vectors and matrices
- SymPy for exact math
- A cheat sheet to keep open during labs
- A final check cell to confirm you are ready for Week 1

  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sakshi-mal02/linear-algebra-python/blob/main/week0_python_basics.ipynb)

---- 
## Weekly Notebooks

**Week 1 — Linear Systems and Matrices** (TBIL: LE1–LE4)

In this lab you will set up your first matrices in Python, compute RREF using SymPy
(the same row reduction you do by hand, but instant), and solve systems with NumPy.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sakshi-mal02/linear-algebra-python/blob/main/week1_linear_systems.ipynb)

Topics covered:

- Creating vectors and matrices with np.array()
- Building augmented matrices with np.column_stack()
- Computing RREF with SymPy's .rref() — exact, no rounding
- Solving Ax = b with np.linalg.solve() and verifying your answer

---

**Week 2 — Vectors, Span, and Independence** (TBIL: EV1–EV6)

This week we use Python to answer two of the most important questions in Chapter 2:
Is this vector in the span of these others? and Are these vectors linearly independent?
We also visualize span geometrically — something very hard to see from equations alone.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sakshi-mal02/linear-algebra-python/blob/main/week2_vectors_span.ipynb)

Topics covered:

- Testing if b is in span{v₁, v₂, ...} via RREF
- Checking linear independence by counting pivot columns
- Visualizing span in R² — the difference between a line and the whole plane
- Finding a basis from pivot columns and stating dimension

---

**Week 3 — Linear Transformations** (TBIL: AT1–AT4)

Here is where things get visually exciting. We apply transformations as matrix multiplication
and watch what they do to the unit square — rotations, shears, projections.
We also compute image and kernel computationally and verify the Rank-Nullity theorem live.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sakshi-mal02/linear-algebra-python/blob/main/week3_transformations.ipynb)

Topics covered:

- Applying T(x) = Ax with the @ operator
- Visualizing what transformations do to shapes
- Computing image (column space) and kernel (null space)
- Checking injectivity and surjectivity from rank
- Finding matrix inverses with NumPy and SymPy

---

**Week 4 — Eigenvalues and Applications** (TBIL: GT1–GT4)

Determinants, eigenvalues, eigenvectors — and a real
application: PageRank, the algorithm that made Google possible.
It turns out the importance scores of web pages are just the dominant eigenvector of a matrix.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sakshi-mal02/linear-algebra-python/blob/main/week4_eigenvalues_applications.ipynb)

Topics covered:

- Determinants and their geometric meaning (area scaling)
- Finding eigenvalues via the characteristic polynomial
- Computing eigenvectors as null spaces of (A − λI)
- Building a PageRank link matrix and finding the dominant eigenvector
- Final project introduction

---

## Handouts and Documents
- [Student Handouts — all weeks](LinAlg_Student_Handouts.docx)
- [Course Schedule](LinAlg_Python_Schedule.docx)

---

## How to Use Colab
1. Click the **Open in Colab** button for your week
2. Click **Copy to Drive** at the top (saves your own editable copy)
3. Run each cell by clicking the play button ▶ on the left
4. Complete the exercises at the bottom of the notebook
