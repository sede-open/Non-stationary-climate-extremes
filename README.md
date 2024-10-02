# Non-stationary-climate-extremes

This repository holds the code relating to the paper `Changes in the 100-year return value for climate variables of engineering concern over the next 100 years`.

The software fits non-stationary versions of (a) generalised extreme value regression (GEVR) to estimate models for (temporal and spatio-temporal) block maxima data, and (b) non-homogeneous Gaussian regression (NHGR) to estimate models for (temporal and spatio-temporal) means. For both models, following Ewans and Jonathan (2023), we assume that model parameters vary linearly over the period of observation. The purpose of the analysis is then to estimate the parameters of the linear forms.
 
K. Ewans and P. Jonathan (2023). Uncertainties in estimating the effect of climate change on 100-year return value for significant wave height. Ocean Eng. 272:113840. arXiv:2212.11049.


## Package Purpose

The purpose of this repository is to provide a space to share and maintain the methods and code used in the above paper.
The package is split into two different modules:

- `gev_estimator` - Contains the code to fit a non-stationary generalised extreme value distribution using MCMC (Markov Chain Monte Carlo).
- `nhgr_estimator` - Contains the code to fit a non-homogeneous Gaussian regression model using MCMC (Markov Chain Monte Carlo).

## How to Use

 To install this package It is highly suggested to install within a python virtual environment. Use the following to install the package.

```shell
pip install --upgrade pip
pip install -e .
```

To get started follow the [example](src/examples/GEV_Fitting_example.ipynb) notebook. Sample data can be found in the `src/data` directory.

## Contributors

| role | name | Email |
| --- | --- | --- |
| maintainer | Callum Leach  | callumleach31@gmail.com |
| statistics | Philip Jonathan  | ygraigarw@gmail.com |