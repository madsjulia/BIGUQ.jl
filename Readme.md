BIGUQ
=======================================

[![BIGUQ](http://pkg.julialang.org/badges/BIGUQ_0.5.svg)](http://pkg.julialang.org/?pkg=BIGUQ&ver=0.5)
[![BIGUQ](http://pkg.julialang.org/badges/BIGUQ_0.6.svg)](http://pkg.julialang.org/?pkg=BIGUQ&ver=0.6)
[![BIGUQ](http://pkg.julialang.org/badges/BIGUQ_0.7.svg)](http://pkg.julialang.org/?pkg=BIGUQ&ver=0.7)
[![Build Status](https://travis-ci.org/madsjulia/BIGUQ.jl.svg?branch=master)](https://travis-ci.org/madsjulia/BIGUQ.jl)
[![Coverage Status](https://coveralls.io/repos/madsjulia/BIGUQ.jl/badge.svg?branch=master)](https://coveralls.io/r/madsjulia/BIGUQ.jl?branch=master)

BIGUQ pefroms Bayesian Information Gap Decision Theory (BIG-DT) analysis for Uncertainty Quantification, Experimental Design and Decision Analysis.
BIGUQ is a module of [MADS](http://madsjulia.github.io/Mads.jl).

Example
-------

```julia
import Mads

problemdir = Mads.getmadsdir()
md = Mads.loadmadsfile(joinpath(problemdir, "source_termination.mads"))
nsample = 1000
bigdtresults = Mads.dobigdt(md, nsample; maxHorizon=0.8, numlikelihoods=5)
Mads.plotrobustnesscurves(md, bigdtresults; filename=joinpath(problemdir, "source_termination-robustness-$nsample"))
Mads.plotrobustnesscurves(md, bigdtresults; filename=joinpath(problemdir, "source_termination-robustness-zoom-$nsample"), maxhoriz=0.4, maxprob=0.1)
```

MADS
====

[MADS](http://madsjulia.github.io/Mads.jl) (Model Analysis & Decision Support) is an integrated open-source high-performance computational (HPC) framework in [Julia](http://julialang.org).
MADS can execute a wide range of data- and model-based analyses:

* Sensitivity Analysis
* Parameter Estimation
* Model Inversion and Calibration
* Uncertainty Quantification
* Model Selection and Model Averaging
* Model Reduction and Surrogate Modeling
* Machine Learning and Blind Source Separation
* Decision Analysis and Support

MADS has been tested to perform HPC simulations on a wide-range multi-processor clusters and parallel environments (Moab, Slurm, etc.).
MADS utilizes adaptive rules and techniques which allows the analyses to be performed with a minimum user input.
The code provides a series of alternative algorithms to execute each type of data- and model-based analyses.

Documentation
=============

All the available MADS modules and functions are described at [madsjulia.github.io](http://madsjulia.github.io/Mads.jl)

Installation
============

```julia
Pkg.add("Mads")
```

Installation behind a firewall
------------------------------

Julia uses git for the package management.
To install Julia packages behind a firewall, add the following lines in the `.gitconfig` file in your home directory:

```git
[url "https://"]
        insteadOf = git://
```

or execute:

```bash
git config --global url."https://".insteadOf git://
```

Set proxies:

```bash
export ftp_proxy=http://proxyout.<your_site>:8080
export rsync_proxy=http://proxyout.<your_site>:8080
export http_proxy=http://proxyout.<your_site>:8080
export https_proxy=http://proxyout.<your_site>:8080
export no_proxy=.<your_site>
```

For example, if you are doing this at LANL, you will need to execute the
following lines in your bash command-line environment:

```bash
export ftp_proxy=http://proxyout.lanl.gov:8080
export rsync_proxy=http://proxyout.lanl.gov:8080
export http_proxy=http://proxyout.lanl.gov:8080
export https_proxy=http://proxyout.lanl.gov:8080
export no_proxy=.lanl.gov
```

MADS examples
=============

In Julia REPL, do the following commands:

```julia
import Mads
```

To explore getting-started instructions, execute:

```julia
Mads.help()
```

There are various examples located in the `examples` directory of the `Mads` repository.

For example, execute

```julia
include(Mads.madsdir * "/../examples/contamination/contamination.jl")
```

to perform various example analyses related to groundwater contaminant transport, or execute

```julia
include(Mads.madsdir * "/../examples/bigdt/bigdt.jl")
```

to perform Bayesian Information Gap Decision Theory (BIG-DT) analysis.

Developers
==========

* [Velimir (monty) Vesselinov](http://www.lanl.gov/orgs/ees/staff/monty) [(publications)](http://scholar.google.com/citations?user=sIFHVvwAAAAJ)
* [Daniel O'Malley](http://www.lanl.gov/expertise/profiles/view/daniel-o'malley) [(publications)](http://scholar.google.com/citations?user=rPzCVjEAAAAJ)
* [see also](https://github.com/madsjulia/BIGUQ.jl/graphs/contributors)

Publications, Presentations, Projects
=====================================

* [mads.lanl.gov/](http://mads.lanl.gov/)
* [ees.lanl.gov/monty](http://ees.lanl.gov/monty)