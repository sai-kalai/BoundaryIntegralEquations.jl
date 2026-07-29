next steps:

Try to make one branch for each


- [x] change access pattern to column-major, record performance before and after
- [x] package more into one struct such that solve(instance) has all the info
- [x] fix the modules: use one single module for the whole code instead of per-file, see Inti
- [inprogress] write docstrings
- [inprogress] write unit tests for correctness
    - [x] put convergence scripts in test
    - [ ] manifolds
    - [x] operators
    - [ ] solvers
    - [ ] holomorphism boundary limit
    - [x] cauchy ingegral
    - [x] autodiff vs analytical gradient

- [x] implement support for vector-valued functions

- [ ] rename hypersingular corrections: both "Sidi" and "Zeta" are of type Zeta


- [ ] implement 2nd derivative approximation for hypersingular kernel using AD

- [x] design better api instead of passing allocator function. maybe, pass already allocated memory
    - [x] adopt api in separate matrices branch, such that testing is homogeneous
    - [x] fix undef initializer in separate matrices branch
    - [x] homogenize interface further so that the only difference is a boolean flag `compute matrices`

- [x] api for solving the BIE attached to the BVP and reusing the density for computing at arbitrary points
    struct containing side, bc type,
    solve()
    evaluate(x points)

- [x] implement distance policy for close evaluation

- [x] move close evaluation and autodiff outside of scripts
- [x] enforce consistent order of arguments across the codebase (source, then target)


- [ ] implement SciML style solution for BIE and BVP

- [x] implement GPU kernel assembly and linear solve
- [ ] extend GPU module, match on type of array, support several backends
- [ ] integrate GPU and AD into main branch

- extend functionality
    - [ ] exterior problem
    - [ ] Helmholtz problem

applications
- [ ] inverse point lookup
- [ ] jacobian determinant boundary shape optim.


Backlog/ideas
- [ ] change DiscreteClosedCurve to Boundary{2}, figure out typing
    i.e. make boundary parametric on rank, and even maybe dimension?
- [ ] assemble function returns the correct BIE linear problem
- [ ] extend the type system for representing geometry
    use cases:
    - boundary: smooth manifold, all information, incl. parametrization
    - target points: only locations
    - dummy boundary: unit weights, for producing manufactured solution results
    - target points with unit normals: for adjoint dlp, where normals at x are needed

    - may be better to use an external package for managing geometry
- [ ] array of structures instead of structure of arrays
    implement homogeneous api, and test performance





Meeting with Dean

- documentation !!! DONE
- bonus: use documentation to write thesis DONE

- close field switching policy DONE
    - error analysis: how to identify when error grows for a better policy
    - smoother transition in error


- exterior points

- map2disc DONE
- Helmholtz on hold
- CI CD github action DONE

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

# Boundary refactor

goals for refactoring the boundary struct:

- provide AoS and SoA with homogeneous interface, test performance
- switch to row-major, measure performance
- make parametric with type of array
