*===============================================================================
* Program: pymlp.ado
* Purpose: Stata wrapper for random forest classification and regression
*          with scikit-learn in Python, using Stata 16's new built-in Python
*          integration functions.
* More info: www.github.com/mdroste/stata-pymlp
* Version: 0.1
* Date: July 16, 2019
* Author: Michael Droste
*===============================================================================

program define pymlp, eclass
version 16.0
syntax varlist(min=2) [if] [in] [aweight fweight], ///
	[ ///
		type(string asis)            	 /// model type: classifier or regressor
		hidden_layer_sizes(string asis)	 /// Tuple with number of hidden layers and their 
		activation(string asis)			 /// xx
		solver(string asis)				 /// xx
		alpha (real 0.0001)			 	 /// L2 penalty parameter (default 0.0001)
		batch_size                       /// Size of minibatches for sotchastic optimizers (default: auto)
		learning_rate(string asis)		 /// learning rate schedule for weight updates
		learning_rate_init(real 0.001) 	 /// initial learning rate (for sgd or adam solvers)
		power_t							 /// learning rate schedule for weight updates
		max_iter(integer 200)			 /// max iterations for neural nets
		shuffle							 /// learning rate schedule for weight updates
		random_state(string asis)		 /// random seed
		tol(real 1e-4) 					 /// tolerance threshold
		verbose 						 /// verbosity of python
		warm_start 						 /// xx not implemented
		momentum(real 0.9)				 /// momentum
		nesterovs_momentum 				 /// nesterovs momentum
		early_stopping 				 	 /// early stopping rule
		validation_fraction(real 0.1)	 /// fraction of training data to set aside for early stopping
		beta_1(real 0.9) 		 		 /// exp parameter in adams smoother, 1st order moments
		beta_2(real 0.999) 				 /// exp parameter in adams smoother, 2nd order moments
		epsilon(real 1e-8) 				 /// some parameter
		n_iter_no_change(real 10) 		 /// max iterations w no change in loss
	    frac_training(real 0.5) 	 	 /// randomly assign fraction X to training
	    training_stratify 		 	 	 /// if nonmissing, randomize at this level
	    training_identifier(varname)	 /// training dataset identifier
		save_prediction(string asis) 	 /// variable name to save predictions
		save_training(string asis) 	 	 /// variable name to save training flag
		standardize                      ///
	]

*-------------------------------------------------------------------------------
* Handle arguments
*-------------------------------------------------------------------------------

* Pass varlist into varlists called yvar and xvars
*--------------
gettoken yvar xvars : varlist


*--------------
* type: string asis, either classify or regress
if "`type'"=="" {
	di as error "ERROR: type() option needs to be specified. Valid options: type(classify) or type(regress)"
	exit 1
}
if ~inlist("`type'","classify","regress") {
	di as error "Syntax error: invalid choice for type (chosen: `type'). Valid options are classify or regress"
	exit 1
}

*--------------
* hidden_layer_sizes: need to validate
if "`hidden_layer_sizes'"=="" local hidden_layer_sizes "100"
local layer_str "`hidden_layer_sizes'"
local layer_str = subinstr("`layer_str'",","," ",.)
local num_layers = wordcount("`layer_str'")
tokenize "`layer_str'"
forval i=1/`num_layers' {
	local nodes_per_layer "`nodes_per_layer', ``i''"
}
local nodes_per_layer = substr("`nodes_per_layer'",3,.)
if `num_layers'>1  local hidden_layer_sizes = "(" + "`hidden_layer_sizes'" + ")"
if `num_layers'==1 local hidden_layer_sizes = "(" + "`hidden_layer_sizes'" + ",)"

*--------------
* activation: activation function choice
if "`activation'"=="" local activation "relu"
if ~inlist("`activation'","identity","logistic","tanh","relu") {
	di as error "Syntax error: activation() must be one of: identity, logistic, tanh, or relu (was `activation')"
	exit 1
}

*--------------
* solver: solver for weight optimization
if "`solver'"=="" local solver "adam"
if ~inlist("`solver'","lbfgs","sgd","adam") {
	di as error "Syntax error: solver() must be one of: lbfgs, sgd, or adam (was `max_depth')"
	exit 1
}

*--------------
* alpha
* xx

*--------------
* batch size: size of minibatches for stochastic optimizers
if "`batch_size'"=="" local batch_size "auto"
if "`batch_size'"!="auto" {
	* xx check to make sure an integer
}

*--------------
* learning_rate: learning rate schedule for weight updates
if "`learning_rate'"=="" local learning_rate "constant"
if ~inlist("`learning_rate'","constant","invscaling","adaptive") {
	di as error "Syntax error: learning_rate() must be one of: constant, invscaling, adaptive (was `learning_rate')"
	exit 1
}

*--------------
* learning_rate_init: controls step size in weight updates, if solver is sgd or adam
if "`learning_rate_init'"=="" local learning_rate_init 0.001
* xx make sure positive number

*--------------
* power_t: exponent for inverse scaling learning rate, if solver is sgd and learning_rate is invscaling
if "`power_t'"=="" local power_t 0.5
* xx make sure positive number? is that required?

*--------------
* max_iter: Max number of iterations
if "`max_iter'"=="" local max_iter 200
if `max_iter'<1 {
	di as error "Syntax error: max_iter() needs to be a positive integer (was `max_iter')"
	exit 1
}

*--------------
* shuffle: Whether to shuffle samples in each iteration, if solver=sgd or adam
if "`shuffle'"=="" local shuffle True

*--------------
* power_t: exponent for inverse scaling learning rate, if solver is sgd and learning_rate is invscaling
if "`power_t'"=="" local power_t 0.5
* xx make sure positive number? is that required?

*--------------
* random_state: initialize random number generator
if "`random_state'"=="" local random_state None
if "`random_state'"!="" & "`random_state'"!="None" {
	if `random_state'<1 {
		di as error "Syntax error: random_state should be a positive integer."
		exit 1
	}
	set seed `random_state'
}

*--------------
* tol: tolerance threshold for optimizing
if `tol'<=0 {
	di as error "Syntax error: tol() can't be negative (was `tol')"
	exit 1
}

*--------------
* verbose: control verbosity of python output (boolean)
if "`verbose'"=="" local verbose 0
if "`verbose'"=="verbose" local verbose 1

*--------------
* warm_start: Unsupported scikit-learn option used to use pre-existing rf object 
if "`warm_start'"=="" local warm_start False

*--------------
* momentum: momentum for gradient descent update, between 0 and 1
if "`momentum'"=="" local momentum 0.9
if `momentum'<=0 | `momentum'>=1 {
	di as error "Syntax error: momentum should be between 0 and 1: 0 < momentum < 1 (was `momentum')"
	exit 1
}


*--------------
* nesterovs momentum: whether to use nesterovs momentum
if "`nesterovs_momentum'"=="" local nesterovs_momentum True

*--------------
* early stopping: Whether to use early stopping to terminate training when validation not improving
if "`early_stopping'"=="" local early_stopping False

*--------------
* validation_fraction: Proportion of training data to set aside for early stopping validation
* xx

*--------------
* beta_1: Exp decay rate used for 1st moment vector estimates in adam, [0,1).
* xx

*--------------
* beta_2: Exp decay rate used for 2nd moment vector estimates in adam, [0,1).
* xx

*--------------
* epsilon: Value for numerical stability in adam
* xx

*--------------
* n_iter_no_change: Max number of epochs to not meet tol improvement, for sgd or adam solvers
* xx

*--------------
* frac_training: fraction of dataset to sample randomly from
if `frac_training'<=0 | `frac_training'>1 {
	di as error "Syntax error: frac_training() should be in (0,1] (was `frac_training')"
	exit 1
}

*--------------
* prediction cant already be a variable name
if "`save_prediction'"=="" local save_prediction _mlp_prediction
capture confirm new variable `save_prediction'
if _rc>7 {
	di as error "Error: save_prediction() cannot specify an existing variable (`save_prediction' already exists)"
	exit 1
}

*--------------
* training dataset indicator cant already be a variable name
if "`save_training'"=="" local save_training _mlp_training
capture confirm new variable `save_training'
if _rc>7 {
	di as error "Error: save_training() cannot specify an existing variable (`save_training' already exists)"
	exit 1
}

*-------------------------------------------------------------------------------
* Manipulate data
*-------------------------------------------------------------------------------

* Generate an index of original data so we can easily merge back on the results
*  xx there is probably a better way to do this... feels inefficient
tempvar index
gen `index' = _n

* preserve original data
preserve

* restrict sample with if and in
marksample touse, strok novarlist
qui drop if `touse'==0

* if classification: check to see if y needs encoding to numeric
local yvar2 `yvar'
if "`type'"=="classify" {
	capture confirm numeric var `yvar'
	if _rc>0 {
		local needs_encoding "yes"
		encode `yvar', gen(`yvar'_encoded)
		noi di "Encoded `yvar' as `yvar'_encoded"
		local yvar2 `yvar'_encoded
	}
}

* restrict sample to jointly nonmissing observations
foreach v of varlist `varlist' {
	qui drop if mi(`v')
}

* Define a training subsample called `save_training'
if "`training_identifier'"!="" {
	rename `training_identifier' `save_training'
}
if "`training_identifier'"=="" {
	gen `save_training' = runiform()<`frac_training'
}

* Get number of obs in train and validate samples
qui count if `save_training'==1
local num_obs_train = `r(N)'
qui count
local num_obs_val = `r(N)' - `num_obs_train'

* Get number of hidden layers, obs per unit

*-------------------------------------------------------------------------------
* If type(regress), run random forest regression
*-------------------------------------------------------------------------------

* Store a macro to slightly change results table
if "`type'"=="regress" local type_str "regression"
if "`type'"=="classify" local type_str "classification"

* Display some info about options
di "{hline 80}"
di in ye "Multi-layer perceptron `type2'"
di in gr "Dependent variable: `yvar'" _continue
di in gr _col(55) "Num. training obs = " in ye `num_obs_train'
di in gr "Features: `xvars'" _continue
di in gr  _col(55) "Num. validation obs = " 	  in ye `num_obs_val'
di in gr "Hidden layers: " in ye `num_layers' in gr " (nodes: " in ye `nodes_per_layer' in gr ")"
di in gr "Activation function: " in ye "`activation'"
di in gr "Solver: " in ye "`solver'"
di in gr "Alpha (L2 penalty term): " in ye "`alpha'"
di in gr "Saved prediction: " in ye "`save_prediction'"
di "{hline 80}"

* Pass options to Python to import data, run MLP, return results
python: run_mlp( ///
	"`type'", ///
	"`save_training' `yvar' `xvars'", ///
	`hidden_layer_sizes', ///
	"`activation'", ///
	"`solver'", ///
	`alpha', ///
	"`batch_size'", ///
	"`learning_rate'", ///
	`learning_rate_init', ///
	`power_t', ///
	`max_iter', ///
	`shuffle', ///
	`random_state', ///
	`tol', ///
	`verbose', ///
	`warm_start', ///
	`momentum', ///
	`nesterovs_momentum', ///
	`early_stopping', ///
	`validation_fraction', ///
	`beta_1', ///
	`beta_2', ///
	`epsilon', ///
	`n_iter_no_change', ///
	"`save_prediction'", ///
	"`save_training'")


*-------------------------------------------------------------------------------
* Clean up before ending
*-------------------------------------------------------------------------------

* keep only index and new data
keep `index' `save_prediction' `save_training'
tempfile t1
qui save `t1'
restore
qui merge 1:1 `index' using `t1', nogen

* If save training was specified, delete temporary save_training var
if "`save_training'"!="" {
	drop `save_training'
}

* If y needed encoding, decode
* XX this is inefficient
if "`needs_encoding'"=="yes" {
	tempvar encode1
	encode `yvar', gen(`encode1')
	label values `save_prediction' `encode1'
	decode `save_prediction', gen(`save_prediction'_2)
	drop `save_prediction'
	rename `save_prediction'_2 `save_prediction'
}

end

*===============================================================================
* Python code
*===============================================================================

*-------------------------------------------------------------------------------
* Python script: runs MLP regressor
*-------------------------------------------------------------------------------

version 16.0
python:

# Import Python libraries
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sfi import Data

#------------------------------------------------------
# Define function: run_mlp_regressor
#------------------------------------------------------

def run_mlp(type, vars, hidden_layer_sizes, activation, solver, alpha, batch_size, learning_rate, learning_rate_init, power_t, max_iter, shuffle, random_state, tol, verbose, warm_start, momentum, nesterovs_momentum, early_stopping, validation_fraction, beta_1, beta_2, epsilon, n_iter_no_change, prediction, training):

	# Load data into data frame
	df = pd.DataFrame(Data.get(vars))
	colnames = []
	for var in vars.split():
		 colnames.append(var)
	df.columns = colnames

	# Split training data and test data into separate data frames
	df_train, df_test = df[df[training]==1], df[df[training]==0]

	# Create list of feature names 
	features = df.columns[2:]
	y        = df.columns[1]

    # Initialize MLP regressor
	if type=="regress":
		mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha, batch_size=batch_size, learning_rate=learning_rate, learning_rate_init=learning_rate_init, power_t=power_t, max_iter=max_iter, shuffle=shuffle, random_state=random_state, tol=tol, verbose=verbose, warm_start=warm_start,momentum=momentum, nesterovs_momentum=nesterovs_momentum, early_stopping=early_stopping, validation_fraction=validation_fraction, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, n_iter_no_change=n_iter_no_change)

	if type=="classify":
		mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha, batch_size=batch_size, learning_rate=learning_rate, learning_rate_init=learning_rate_init, power_t=power_t, max_iter=max_iter, shuffle=shuffle, random_state=random_state, tol=tol, verbose=verbose, warm_start=warm_start,momentum=momentum, nesterovs_momentum=nesterovs_momentum, early_stopping=early_stopping, validation_fraction=validation_fraction, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, n_iter_no_change=n_iter_no_change)

    # Train MLP regressor on training data
	mlp.fit(df_train[features], df_train[y])

    # Fit data with trained MLP regressor
	pred = mlp.predict(df[features])

	# Export predictions to stata
   	Data.addVarFloat(prediction)
	Data.store(prediction,None,pred)

end
