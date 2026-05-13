# simulation_of_linear_regression_in_Python

# Local Linear Regression Simulation Project

This project explores the performance and inherent limitations of **Linear Regression** when applied to non-linear growth functions, specifically the exponential function $f(x) = e^{2x}$. It demonstrates how local linear approximations can be used to model complex curves at a high computational cost.

## 1. Experimental Background
In standard linear regression, the model attempts to find a global best-fit line:
$$y = w_1x + w_0$$

* **The Problem**: A single straight line cannot capture the curvature of exponential growth.
* **The Solution**: By increasing data density and performing **Repeated Linear Approximations** (using a sliding window), we can achieve a precise fit through a series of "tangent-like" segments.

## 2. Dependencies
To run the simulation, ensure you have the following Python libraries installed:
```bash
pip install numpy scipy scikit-learn matplotlib keyboard
