# Equations

## Multicompartment primitive

For local projections `P_b`, branch nonlinearities `phi_b`, and routed weights `r_b(x)`:

```math
D(x) = \rho\!\left(\sum_{b=1}^{B} r_b(x)\,v_b\,\phi_b(P_bx;\theta_b)\right).
```

The reference RBF branch uses:

```math
\phi_b(x)=\exp\!\left(-\frac{d_{g_b}(P_bx,c_b)^2}{2\sigma_b^2}\right),
```

where `g_b` may be Euclidean or hyperbolic.

## Exact bounded Boolean compilation

For `p` in the Boolean cube, the minterm indicator is:

```math
\chi_p(x)=\prod_{j:p_j=1}x_j\prod_{j:p_j=0}(1-x_j).
```

Every bounded Boolean function has the exact local construction:

```math
f(x)=\sum_{p\in\{0,1\}^d} f(p)\chi_p(x).
```

The implementation stores active minterms explicitly and also keeps an equivalent optimized truth-table lookup. Tests require exact equivalence between both paths.

## Recursive parity

A verified two-input XOR Dendritron composes as a balanced tree. For `n` inputs:

```math
N_D=n-1,\qquad N_{branches}=2(n-1),\qquad depth=\lceil\log_2 n\rceil.
```

This avoids compiling one exponential-width `n`-input truth table into a single unit.

## Poincare distance

For points in the curvature-`c` Poincare ball:

```math
d_c(u,v)=\frac{1}{\sqrt c}\operatorname{arcosh}\!\left(1+\frac{2c\lVert u-v\rVert^2}{(1-c\lVert u\rVert^2)(1-c\lVert v\rVert^2)}\right).
```

Inputs are clipped inside the open ball and denominators are floored to keep the numerical path finite.

## PPCA memory likelihood

For coordinate `z`, mean `mu`, retained basis `U`, retained variances `lambda`, and residual variance `sigma^2`, the verifier computes a low-rank Gaussian log likelihood:

```math
\log p(z)=-\frac12\left[(z-\mu)^T C^{-1}(z-\mu)+\log|C|+d\log(2\pi)\right],
```

with `C = U diag(lambda) U^T + sigma^2(I-UU^T)`. Eigenvalues and residual variance are floored defensively.

