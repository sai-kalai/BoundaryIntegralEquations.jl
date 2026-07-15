# BoundaryIntegralEquations

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://sai-kalai.github.io/BoundaryIntegralEquations.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sai-kalai.github.io/BoundaryIntegralEquations.jl/dev/)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://sai-kalai.github.io/BoundaryIntegralEquations.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sai-kalai.github.io/BoundaryIntegralEquations.jl/dev/)
[![Build Status](https://github.com/sai-kalai/BoundaryIntegralEquations.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/sai-kalai/BoundaryIntegralEquations.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Build Status](https://app.travis-ci.com/sai-kalai/BoundaryIntegralEquations.jl.svg?branch=main)](https://app.travis-ci.com/sai-kalai/BoundaryIntegralEquations.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/sai-kalai/BoundaryIntegralEquations.jl?svg=true)](https://ci.appveyor.com/project/sai-kalai/BoundaryIntegralEquations-jl)
[![Build Status](https://api.cirrus-ci.com/github/sai-kalai/BoundaryIntegralEquations.jl.svg)](https://cirrus-ci.com/github/sai-kalai/BoundaryIntegralEquations.jl)
[![Coverage](https://codecov.io/gh/sai-kalai/BoundaryIntegralEquations.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/sai-kalai/BoundaryIntegralEquations.jl)
[![Coverage](https://coveralls.io/repos/github/sai-kalai/BoundaryIntegralEquations.jl/badge.svg?branch=main)](https://coveralls.io/github/sai-kalai/BoundaryIntegralEquations.jl?branch=main)

> Boundary Integral Methods with Differentiation in Julia.

`BoundaryIntegralEquations.jl` is a Julia package for solving boundary integral equations with an emphasis on differentiable scientific computing. It provides high-order boundary integral operators, numerical quadrature, finite-difference utilities, and support for automatic differentiation, making it suitable for PDE-constrained optimization, inverse problems, and shape optimization.

The package is designed around modular operators and kernels, allowing researchers to prototype and experiment with boundary integral methods while leveraging Julia's performance and composability.

## Features

* High-order boundary integral operators
* Laplace kernel implementations
* Boundary density representations
* Numerical quadrature, including Kapur–Rokhlin corrections for logarithmic singularities
* Finite difference utilities for verification and validation
* Automatic differentiation support via Enzyme (extension module)
* Modular architecture for extending kernels, operators, and solvers

## Installation

```julia
using Pkg

Pkg.add(url="https://github.com/sai-kalai/bim-diff.git")
```

Or clone the repository and develop locally:

```bash
git clone https://github.com/sai-kalai/bim-diff.git
cd bim-diff
```

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Project Structure

```
src/
├── BoundaryIntegralEquations.jl                 # Package entry point
├── kernels.jl                 # Boundary integral kernels
├── operators.jl               # Integral operators
├── densities.jl               # Density representations
├── manifolds.jl               # Boundary/manifold geometry
├── solvers.jl                 # Linear solvers and algorithms
├── finite_differences.jl      # Finite difference utilities
├── close_evaluation.jl        # Near-boundary evaluation
├── kapur_rokhlin_sep_log.jl   # Singular quadrature corrections
└── utils.jl                   # Shared utilities

ext/
└── BoundaryIntegralEquationsEnzymeExt.jl        # Enzyme automatic differentiation extension

scripts/
├── main.jl
└── ellipse.jl                 # Example geometry

test/
└── Comprehensive unit and convergence tests
```

## Running Tests

Run the package test suite with

```julia
using Pkg
Pkg.test()
```

or

```bash
julia --project=. test/runtests.jl
```

## Examples

Example scripts are located in `scripts/`.

```bash
julia --project=. scripts/main.jl
```

or

```bash
julia --project=. scripts/ellipse.jl
```

## Documentation

Documentation sources are located in the `docs/` directory and can be built using Julia's documentation tooling.

## Research Focus

The package currently includes implementations related to

* Boundary integral formulations
* Laplace equations
* High-order quadrature
* Close evaluation techniques
* Differentiable numerical methods
* Automatic differentiation for scientific computing

## Development Status

`BoundaryIntegralEquations.jl` is an active research project under development. APIs may change as new algorithms and numerical methods are incorporated.

## Contributing

Contributions, bug reports, and feature requests are welcome. Please open an issue or submit a pull request.

## License

This project is distributed under the MIT License. See the `LICENSE` file for details.
