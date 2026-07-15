next steps:

Try to make one branch for each


- [ ] change access pattern to column-major, record performance before and after
- [x] package more into one struct such that solve(instance) has all the info
- [x] fix the modules: use one single module for the whole code instead of per-file, see Inti
- [ ] write docstrings
- [x] write unit tests for correctness
- [ ] implement support for vector-valued functions

- [ ] change DiscreteClosedCurve to Boundary{2}, figure out typing
- [ ] rename hypersingular corrections: both "Sidi" and "Zeta" are of type Zeta


- [ ] implement 2nd derivative approximation for hypersingular kernel using AD

- [x] design better api instead of passing allocator function. maybe, pass already allocated memory
    - [x] adopt api in separate matrices branch, such that testing is homogeneous
    - [x] fix undef initializer
- [x] api for solving the  BIE attached to the BVP and reusing the density for computing at arbitary points
    struct containing side, bc type,
    solve()
    evaluate(x points)
    - [ ] consider extending the operators to also store source and target information

- [ ] implement distance policy for close evaluation

- [ ] move close evaluation and autodiff outside of scripts
- [x] enforce consistent order of arguments across the codebase (source, then target)
- [ ] extend the type system for representing geometry
    use cases:
    - boundary: smooth manifold, all information, incl. parametrization
    - target points: only locations
    - dummy boundary: unit weights, for producing manufactured solution results
    - target points with unit normals: for adjoint dlp, where normals at x are needed
    - [ ] consider using Manifolds.jl


- [ ] array of structures instead of structure of arrays

- [ ] Solution struct, following SciML guidelines

- [ ] write unit tests for autodiff, verify with ForwardDiff/analytical



Meeting with Dean


- documentation !!!
- bonus: use documentation to write thesis

- close field switching policy
    - error analysis: how to identify when error grows for a better policy
    - smoother transition in error


- exterior points

- map2disc
- Helmholtz on hold
- CI CD github action

- journal paper: this is the problem and method

- JOSS maybe publication
- Andreas Buchheit



# TIL
it's not possible to use different types of activity in two struct fields with
Enzyme



# Journal

It is possible to need derivatives of


the solution w.r.t.
- coordinates (i.e. spatial gradient of solution at evaluation points)
- relevant for evaluation of solved BIE

a functional, e.g. avg. jacobian of map2disc, for optimization w.r.t
- boundary geometry
- boundary conditions
- relevant for optimization loops of solve
