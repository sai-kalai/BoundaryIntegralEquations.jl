
# Title: Design and implementation of a boundary integral method solver in Julia

Research questions:

## Intro

### Motivation

- why does it matter? -> dimensionality reduction
- what kinds of PDE can be solved? in what applications? -> Elliptic + Heat
- how does it fit in specific use cases? -> GVEC initial guess

### Research questions
- how can BIM be materialized into an efficient software implementation?
    - What optimizations can be applied to a BIM solver?
    -> parall kernel assembly and linear solve, simultaneous operator computation
- how can extensibility and maintainability be obtained through software architecture?
- Does the implementation reproduce reference solutions? -> show analytical manufactured solution
- what performance does the implementation achieve, and how is it influenced by optimization?
- How can AD be used for BIM?
    -> test cases for shape optimization, differentiable parallel gradients


### Objectives

Main objective: to design, implement, test, evaluate a software tool for solving
BVPs using the BIM

### Scope

- 2D
- Laplace, [Helmholtz], [Poisson]
- Zeta quadrature for hypersingular op through Richardson/FD/[AD]
- Internal, [External]
- Neumann, Dirichlet
- Direct, Indirect

future work:

- 3D
- Stokes, Heat
- galerkin


### Outline

## Theoretical framework
- boundary value problems basics
- functional analysis basics
- complex analysis basics: holomorphisms, cauchy integral
- riemann zeta func.
- quad. methods from Wu paper
- potential Theory, solutions to BVP
- singular ingegral operators/equations
- zeta quadrature, Kapu-Rokhlin
- discrete approximations, Nyström's method
- autodiff

 NOTE: work backwards, end up putting theory where it's needed

## Literature Review
Related approaches
- BEAST.jl
- Inti.jl


## Methodology

### Software Design

- SciML guidelines in julia

### Data collection: simulations, problem setups

- several clients: shape/coordinates optimization loop, map2disc, etc.
- data analysis
- validity: comparison with othe methods

## Results and discussion


## Conclusions and future work


future work:
- further equations: helmholtz, stokes
- further methods: galerkin


