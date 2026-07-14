# BimDiff.jl

> Boundary Integral Methods with Differentiation in Julia.

`BimDiff.jl` is a Julia package for solving boundary integral equations with an emphasis on differentiable scientific computing. It provides high-order boundary integral operators, numerical quadrature, finite-difference utilities, and support for automatic differentiation, making it suitable for PDE-constrained optimization, inverse problems, and shape optimization.

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
├── BimDiff.jl                 # Package entry point
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
└── BimDiffEnzymeExt.jl        # Enzyme automatic differentiation extension

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

`BimDiff.jl` is an active research project under development. APIs may change as new algorithms and numerical methods are incorporated.

## Contributing

Contributions, bug reports, and feature requests are welcome. Please open an issue or submit a pull request.

## License

This project is distributed under the MIT License. See the `LICENSE` file for details.
