# MixMod
MixMod is package for fitting mixture models with an arbitrary number of components to data. While it can store parameters and calculate probabilities for mixture models containing components from arbitrary continuous distributions, it can currently only fit model parameters for a relatively small subset of the named distributions defined in scipy's stats module. The EM equations are explicitly solved for these distributions, which makes fitting the parameters fast and robust, however, at the cost of limiting the types of distributions which are supported as possible components.

The MixtureModel class is fully documented. However, the tutorial is the recommended introduction to this package.

## Installation
To install mixmod, run the following command:

```
pip install mixmod
```
