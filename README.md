# 🎓 Master's Thesis – Solving BSDEs with the Finite Element Method

This project implements the main results of **Bender & Kohlmann (2008)** on backward stochastic differential equations (BSDEs) using a **finite element method** for spatial approximation and an orthonormal basis projection.

## 📘 Objective

The goal of this thesis is to **numerically solve a class of BSDEs** by projecting the solution onto an **orthonormal basis**, as proposed in:

> Bender, C., & Kohlmann, M. (2008). *Forward-backward stochastic differential equations and continuous time random walks*. Stochastic Processes and their Applications.

## 📁 Project Structure

- `Finite_differences.py` – Main script to run the simulation.

## 🧠 Methodology

1. **Time discretization** via Euler scheme.
2. **Finite element approximation** using a projection on an orthonormal basis.
3. **Forward simulation** of the stochastic process.
4. **Terminal condition projection** onto the finite element space.
5. **Backward iteration** for computing the BSDE solution, possibly under constraints.

## 🧪 Numerical Results

The numerical results are compared against known solutions or benchmark values.

## ⚙️ Requirements

- Python 3.8 or higher
- NumPy
- SciPy
- Matplotlib

