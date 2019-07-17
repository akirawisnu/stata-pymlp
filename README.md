
pymlp
=================================

[Overview](#overview)
| [Installation](#installation)
| [Usage](#usage)
| [Benchmarks](#benchmarks)
| [To-Do](#todo)
| [Acknowledgements](#acknowledgements)
| [License](#license)

Regression and classification with multi-layer perceptrons in Stata

`version 0.11 17jul2019`


Overview
---------------------------------

pymlp is an implementation of multi-layer perceptrons in Stata 16 for classification and regression. It is essentially a wrapper around the popular scikit-learn library in Python, making use of the Stata Function Interface to pass data to and from Python from within the Stata window. 


Prequisites
---------------------------------

pymlp requires Stata version 16 or higher, since it relies on the Python integration introduced in Stata 16.0. It also requires Python 3.x and the scikit-learn library. If you have not installed Python or scikit-learn, I would highly recommend starting with the [Anaconda distribution](https://docs.anaconda.com/anaconda/).


Installation
---------------------------------

There are two options for installing pymlp.

1. The most recent version can be installed from Github with the following Stata command:

```stata
net install pymlp, from(https://raw.githubusercontent.com/mdroste/stata-pymlp/master/)
```

2. A ZIP containing pymlp.ado and pymlp.sthlp can be downloaded from Github and manually placed on the user's adopath.


Usage
---------------------------------

Basic usage of pymlp is pretty simple. The syntax looks similar to -regress-. Optional arguments have syntax that is nearly identical to [sklearn.neural_network.MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) and [sklearn.neural_network.MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html).

Here is a quick example demonstrating how to use pymlp for classification:

```stata
* load dataset of flowers
use http://www.stata-press.com/data/r10/iris.dta, clear

* mark approx half of the dataset for estimation
gen train = runiform()<0.5

* run MLP classification, save predictions as predicted_iris
pymlp iris seplen sepwid petlen petwid, type(classify) training_identifier(train) save_prediction(predicted_iris)
```

Here is a quick example demonstrating how to use pymlp for regression with two hidden layers, with 25 nodes in the first hidden layer and 50 in the second.

```stata
* load dataset of cars
sysuse auto, clear

* mark approx 30% of obs for estimation
gen train = runiform()<0.3

* run MLP regression, save predictions as predicted_price
pymlp price mpg trunk weight, type(regress) hidden_layer_sizes(25,50) training_identifier(train) save_prediction(predicted_price)
```

(Incomplete) internal documentation can be found within Stata. This documentation is still a work in progress:
```stata
help pymlp
```

Finally, since the option syntax in this package is inherited from scikit-learn, the documentation for the scikit methods [sklearn.neural_network.MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) and [sklearn.neural_network.MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html) may be useful. 

  
Todo
---------------------------------

The following items will be addressed soon:

- [ ] Finish off this readme.md and the help file
- [ ] Proide some benchmarking
- [ ] Make exception handling more robust
- [ ] Add support for weights
- [ ] Return some stuff in e()
- [ ] Post-estimation: permutation feature importance
- [ ] Model selection: cross-validation


Acknowledgements
---------------------------------

This program relies on the wonderful Python package scikit-learn.


License
---------------------------------

pymlp is [MIT-licensed](https://github.com/mdroste/stata-pymlp/blob/master/LICENSE).
