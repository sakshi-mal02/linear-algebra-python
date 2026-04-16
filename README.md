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

Beginner Version - 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/sakshi-mal02/linear-algebra-python/blob/main/Week3_Linear_Transformations%20(beginner)%20.ipynb)

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

---

🎓 Final Project
You will complete one of the four projects below. Each one takes a real-world problem
and shows how the linear algebra from this course is the engine underneath it.
You will submit two things:

A Google Colab notebook with your documented, working code
A PDF report (4–6 pages) explaining your problem, methods, and results

---

Option A — Image Compression via SVD 🖼️
The big idea: Every digital image is just a matrix of numbers.
Singular Value Decomposition (SVD) breaks that matrix into layers of decreasing importance.
By keeping only the most important layers you get a compressed image that still looks good —
this is the mathematics behind image compression used in photography and streaming.
What you will do:

Load a real grayscale image and treat it as a matrix
Apply numpy.linalg.svd() to decompose it
Reconstruct the image using only the top k singular values for several values of k
Plot a side-by-side comparison of original vs. compressed, and a quality vs. compression curve
Explain the connection between the number of singular values kept and the rank of the approximation

Linear algebra concepts used:
Rank · Column space · Matrix approximation · Singular values
Python tools: numpy.linalg.svd() · matplotlib.pyplot.imshow() · PIL or skimage for image loading

---

Option B — Markov Chain Steady State 📊
The big idea: Many real-world processes — weather patterns, board games, population movement,
customer behavior — can be modeled as a system that randomly jumps between states.
A Markov chain captures this as a matrix, and linear algebra tells us exactly where the
system ends up in the long run: the steady-state distribution, which is an eigenvector.
What you will do:

Choose a real-world process and model it as a Markov chain (at least 4 states)
Write the transition matrix where each column sums to 1
Find the steady-state distribution as the eigenvector with eigenvalue λ = 1
Simulate the chain for 500–1000 steps and plot how the distribution evolves over time
Compare the simulation result to the exact eigenvector answer

Linear algebra concepts used:
Eigenvalues · Eigenvectors · Column-stochastic matrices · Long-run behavior
Python tools: numpy.linalg.eig() · matplotlib · numpy.random for simulation

---

Option C — Least Squares Regression 📈
The big idea: Real data is messy — it rarely fits a line or curve exactly.
Least squares finds the best possible fit by solving a linear system that has no exact solution.
The key insight: instead of solving Ax = b (which may be inconsistent), we solve AᵀAx = Aᵀb,
which always has a solution and gives us the projection of b onto the column space of A.
What you will do:

Find or collect a real dataset with at least one input variable and one output variable
Set up the design matrix A and target vector b for a polynomial or multi-variable fit
Solve the normal equations using numpy.linalg.solve(A.T @ A, A.T @ b)
Plot the data and fitted model together; compute and visualize the residuals
Explain what the column space of A represents in the context of your regression

Linear algebra concepts used:
Column space · Projection · Normal equations · Inconsistent systems · Least squares solution
Python tools: numpy · matplotlib · optionally pandas for data loading

---

