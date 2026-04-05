2D Ideal MHD Solver: HLLD + GLM-MHD

Developed by Abdelrahman Shaltout Swansea University, BEng Aerospace Engineering

A high-performance 2D Ideal Magnetohydrodynamics (MHD) solver implemented in Python. This project simulates complex plasma dynamics, specifically focusing on the stability of twisted magnetic flux tubes.
Technical Specifications

 HLLD  for high-resolution shock and contact capturing.

Hyperbolic/Parabolic GLM-MHD to maintain ∇⋅B≈0.

2nd-order MUSCL TVD with MinMod/Van Leer limiters.

3rd-order Strong Stability Preserving Runge-Kutta.

The solver is validated against the Gold-Hoyle equilibrium. In production runs, the code successfully identifies the m=9 kink instability growth rate (γ≈79.2) when the safety factor q(a) drops below the Kruskal-Shafranov threshold.
