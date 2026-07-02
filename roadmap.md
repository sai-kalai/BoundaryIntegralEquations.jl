next steps:

Try to make one branch for each


- [ ] change access pattern to column-major, record performance before and after
- [ ] package more into one struct such that solve(instance) has all the info
- [x] fix the modules: use one single module for the whole code instead of per-file, see Inti
- [ ] write docstrings
- [ ] write unit tests for correctness
- [ ] implement support for vector-valued functions
- [ ] change DiscreteClosedCurve to Boundary{2}, figure out typing
- [ ] rename hypersingular corrections: both "Sidi" and "Zeta" are of type Zeta
- [ ] implement 2nd derivative approximation for hypersingular kernel using FD
- [x] design better api instead of passing allocator function. maybe, pass already allocated memory
    - [ ] adopt api in separate matrices branch, such that testing is homogeneous
- [ ] api for solving the  BIE attached to the BVP and reusing the density for computing at arbitary points
    struct containing side, bc type,
    solve()
    evaluate(x points)
    - [ ] consider extending the operators to also store source and target information
- [ ] implement distance policy for close evaluation
- [ ] move close evaluation and autodiff outside of scripts
- [ ] enforce consistent order of arguments across the codebase (source, then target)
- [ ] extend the type system for representing geometry
    use cases:
    - boundary: smooth manifold, all information, incl. parametrization
    - target points: only locations
    - dummy boundary: unit weights, for producing manufactured solution results
    - target points with unit normals: for adjoint dlp, where normals at x are needed
- [ ] consider using Manifolds.jl






