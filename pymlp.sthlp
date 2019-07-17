{smcl}
{* *! version 0.1  16jul2019}{...}
{viewerjumpto "Syntax" "pymlp##syntax"}{...}
{viewerjumpto "Description" "pymlp##description"}{...}
{viewerjumpto "Options" "pymlp##options"}{...}
{viewerjumpto "Examples" "pymlp##examples"}{...}
{viewerjumpto "Author" "pymlp##author"}{...}
{viewerjumpto "Acknowledgements" "pymlp##acknowledgements"}{...}
{viewerjumpto "References" "pymlp##references"}{...}
{title:Title}
 
{p2colset 5 14 21 2}{...}
{p2col :{hi:pymlp} {hline 2}}Multi-layer perceptron regression and classification with Python and scikit-learn{p_end}
{p2colreset}{...}
 
 
{marker syntax}{title:Syntax}
 
{p 8 15 2}
{cmd:pymlp} depvar indepvars {ifin}, type(string) [{cmd:}{it:options}]
                               
 
{synoptset 40 tabbed}{...}
{synopthdr :options}
{synoptline}
 
{syntab :Main}
{synopt :{opt type(string)}}{it:string} may be {bf:regress} or {bf:classify}.{p_end}

{syntab :Pre-processing}
{synopt :{opt frac_training(#)}}fraction of observations to place in training dataset{p_end}
{synopt :{opt training_identifier(varname)}}Use varname as indicator for training sample{p_end}
{synopt :{opt standardize}}Standardize features to mean zero and unit variance in training data{p_end}
 
{syntab :Network options}
{synopt :{opt hidden_layer_sizes(tuple)}}Tuple containing number of neurons in each hidden layer{p_end}
{synopt :{opt activation(string)}}Activation function for hidden layer (see details below){p_end}
{synopt :{opt alpha(#)}}L2 norm penalty on weights{p_end}
{synopt :{opt class_weight}}Not yet implemented{p_end}

{syntab :Optimizer options}
{synopt :{opt solver(string)}}Algorithm used for weight optimization (see details below){p_end}
{synopt :{opt batch_size(#)}}Size of minibatches for stochastic optimization{p_end}
{synopt :{opt max_iter(#)}}Maximum iterations (epochs) of backpropgation{p_end}
{synopt :{opt tol(#)}}Tolerance threshold for updating{p_end}
{synopt :{opt learning_rate(string)}}Learning rate schedule for weight updates{p_end}
{synopt :{opt learning_rate_init(#)}}Controls step size in weight updates for solver(sgd) or solver(adam){p_end}
{synopt :{opt power_t(#)}}Exponent for learning rate when learning_rate(invscaling) and solver(sgd){p_end}
{synopt :{opt beta_1(#)}}Exponential decay for 1st moment vector estimate with solver(adam){p_end}
{synopt :{opt beta_2(#)}}Exponential decay for 2nd moment vector estimate with solver(adam){p_end}
{synopt :{opt noshuffle}}If specified, do not shuffle samples for solver(sgd) or solver(adam).{p_end}
{synopt :{opt momentum(#)}}Momentum for gradient descent with solver(sgd){p_end}
{synopt :{opt no_nesterovs_momentum}}Do not use nesterovs momentum if solver=sgd, momentum>0{p_end}
{synopt :{opt early_stopping}}Terminate training when validation score is not improving{p_end}
{synopt :{opt validation_fraction(#)}}Fraction of training to set aside when early_stopping specified{p_end}

{syntab :Output options}
{synopt :{opt save_predictions(newvar)}}Save prediction as {bf: newvar}{p_end}
{synopt :{opt save_training(newvar)}}Save indicator for training sample as {bf: newvar}{p_end}

{syntab :Miscellaneous options}
{synopt :{opt random_state(#)}}Random seed used in Python optimization / pre-processing{p_end}
{synopt :{opt n_jobs(#)}}Number of cores to use when processing data{p_end}
 
 
{marker description}{...}
{title:Description}
 
{pstd}
{opt pymlp} performs regression or classification with a multi-layer perceptron, featuring a single input layer, a single output layer, and arbitrary number of hidden layers/nodes.

{pstd} In particular, {opt pymlp} implements a multi-layer perceptron as a wrapper around the Python modules sklearn.neural_network.MPLRegressor and sklearn.neural_network.MPLClassifier, distributed as components of the scikit-learn Python library.

{pstd} {opt pymlp} relies critically on the Python integration functionality introduced with Stata 16. Therefore, users will need Stata 16, Python (preferably 3.x), and the scikit-learn library installed in order to run.

 
 
{marker options}{...}
{title:Options}
 
{dlgtab:Main}
 
{phang}
{opth type(string)} declares whether this is a regression or classification problem. In general, type(classify) is more appropriate when the dependent variable is categorical, and type(regression) is more appropriate when the dependent variable is continuous.
 
{dlgtab:Training options}
 
{phang}
{opt frac_training(real)} determines the fraction of observations to be randomly sampled for the training dataset. Note that frac_training should be in (0,1]. By default, this is 0.5.

{phang}
{opt training_identifier(varname)} identifies an indicator variable in the current dataset that is equal to 1 when an observation should be used for training and 0 otherwise. If this option is specified, frac_training() is ignored.

{dlgtab: Network options}
 
{phang}
{opt hidden_layer_sizes(string)} determines the number of hidden layers and nodes per hidden layer. The default is hidden_layer_sizes((100,)), which is one hidden layer with 100 nodes. To choose two hidden layers with 25 nodes in the first layer and 50 nodes in the second, one would use hidden_layer_sizes((25,50)). 

{phang}
{opt activation(string)} determines the activation function used for the hidden layer. The default is activation("relu"), the rectified linear unit function f(x)=max(0,x). Alternative options are activation("identity"), activation("logistic"), and activation("tanh").

{phang}
{opt alpha(real)} is a regularization parameter than specifies an L2 norm penalty on the weights.


{dlgtab:Optimizer settings}

{phang}
{opt solver(string)} specifies the wolver for weight optimization. Valid options are solver("lbfgs"), a quasi-Newton optimizer; solver("sgd"), stochastic gradient descent; and solver("adam"), a stochastic gradient descent-based optimizer proposed by Kingma, Diederik, and Ba. The default is solver("adam"). 

{phang}
{opt max_iter(integer)} specifies the maximum number of iterations (or epochs) by the solver. The solver will interate either until convergence (determined by tol()) or the maximum number of iterations is reached. 

{phang}
{opt tol(real)} specifies a tolerance threshold for updating. The default is tol(1e-4). 

{phang}
{opt batch_size(integer)} specifies the size of minibatches when using a stochastic optimizer; that is, solver("adam") or solver("sgd").  By default, this is min(200,# obs in training data).

{phang}
{opt learning_rate(string)} specifies the learning rate schedule for weight updates. By default, learning_rate("constant") is used, which is a constant learning rate given by learning_rate_init(). Other valid options are learning_rate("invscaling"), which decreases the learning rate at each iteration using an inverse scaling exponent power_t(), and learning_rate("adaptive"), which keeps the learning rate constant at learning_rate_init() if the training loss decreases. If two consecutive iterations fail to decrease training loss by at least tol, the learning rate is divided by 5.

{phang}
{opt learning_rate_init(real)} specifies the initial learning rate used; controls the step-size in updating weights when solver("sgd") or solver("adam") is used.

{phang}
{opt power_t(real)} specifies the exponent for inverse scaling learning rate. Only used when learning_rate("invscaling") and solver("sgd") are used. 

{phang}
{opt beta_1(real)} specifies the exponential decay rate of the 1st momemnt vector estimate with solver("adam"). By default, this is 0.9.

{phang}
{opt beta_2(real)} specifies the exponential decay rate of the 2nd momemnt vector estimate with solver("adam"). By default, this is 0.999.

{phang}
{opt epsilon(real)} specifies a small constant used for numerical stability with solver("adam"). By default, this is 1e-8.

{phang}
{opt noshuffle} if specified, do not permute observations in each iteration. This option is only relevant when the stochastic optimization algorithms solver("sgd") or solver("adam") are used. By default, this option is not specified.

{phang}
{opt n_iter_no_change(integer)} Maximum number of iterations (epochs) to allow the solver to not meet tol() improvement. This option is only effective with one of the stochastic solvers, solver("sgd") or solver("adam").


 


{dlgtab:Save Output}
 
{phang}
{opt save_prediction(newvar)} specifies a new variable name for storing predicted outcomes. By default, _rf_prediction is used. The prediction is populated for both the training and validation subsamples.
{opt save_training(varname)} species the variable name for an indicator denoting whether an observation is in the training data or not. By default, _rf_prediction is used.

{phang}
{opt save_training(newvar)} specifies a new variable name for storing an indicator variable for whether an observation is in the training dataset.  If training_identifier() is used, this option does nothing, since training_identifier() points to a pre-existing variable.


{dlgtab:Miscellaneous options}

{phang}
{opt random_state(integer)} sets a random seed for both the drawing of training data (if applicable) and the MLP solving, if a stochastic solver is specified.



{marker examples}{...}
{title:Examples}
 
{pstd}See the Github page.{p_end}

{pstd}Example 1: Classification with MLP neural network: two hidden layers, 100 layers per node{p_end}
{phang2}. {stata sysuse iris, clear}{p_end}
{phang2}. {stata pymlp iris seplen sepwid petlen petwid, type(classify) save_prediction(iris_predicted) hidden_layer_sizes((100,100))}{p_end}

{pstd}Example 2: Classification with MLP neural network, more options{p_end}
{phang2}. {stata sysuse iris, clear}{p_end}
{phang2}. {stata pymlp iris seplen sepwid petlen petwid, type(classify) frac_training(0.3) save_prediction(iris_predicted) save_training(training_sample) criterion(entropy)}{p_end}
 
{marker author}{...}
{title:Author}
 
{pstd}Michael Droste{p_end}
{pstd}mdroste@fas.harvard.edu{p_end}
 
 
 
{marker acknowledgements}{...}
{title:Acknowledgements}

{pstd}This program owes a lot to the wonderful scikit-learn library in Python.


{marker references}{...}
{title:References}

 
