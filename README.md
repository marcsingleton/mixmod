# MixMod
MixMod is package for fitting mixture models with an arbitrary number of components to data. While it can store parameters and calculate probabilities for mixture models containing components from arbitrary continuous distributions, it can currently only fit model parameters for a relatively small subset of the named distributions defined in SciPy's stats module. The EM equations are explicitly solved for these distributions, which makes fitting the parameters fast and robust, however, at the cost of limiting the types of distributions which are supported as possible components.

The MixtureModel class is fully documented. However, the [tutorial](https://github.com/marcsingleton/mixmod/blob/main/tutorial.ipynb) is the recommended introduction to this package.

## Dependencies
MixMod is designed to be lightweight and only requires NumPy for array calculations and SciPy for optimization and some special functions. The minimum versions were set to the most recent at the time of initial release (1.17 and 1.8, respectively), but no functionality specific to these releases is needed to my knowledge. There is also a "soft" dependency on the SciPy's implementation of the supported distributions since MixMod expects the random variables to implement pdf and cdf methods with the parametrizations used by SciPy.

## Installation
To install MixMod, run the following command:

```
pip install mixmod
```
