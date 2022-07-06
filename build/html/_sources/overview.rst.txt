Overview
========

This document explains how to run experiments with :term:`Seldonian algorithms<Seldonian Algorithm>` (SAs) using this library. For a detailed description of what SAs are, see `the UMass AI Safety page <http://aisafety.cs.umass.edu/overview.html>`_, specifically `the 2019 Science paper <http://aisafety.cs.umass.edu/paper.html>`_. 

This document references the `Seldonian Engine library <https://seldonian-framework.github.io/Engine>`_, the core library for running Seldonian algorithms.  


Seldonian experiments
---------------------
A Seldonian experiment is a way to evaluate the safety and performance of an SA. It involves running the SA many times with varying amounts of input data.  The way we evaluate safety and performance in this library is with the "Three Plots".

Three Plots
-----------

The "Three Plots" that we implement in this library are:

1. Performance: the value of your primary objective, for example accuracy, evaluated at the solution returned by the SA. Satisfying the behavioral constraints sometimes comes at the cost of reduced performance. 
2. Solution rate: the probability that a solution is returned by the SA. If the behavioral constraints cannot be satisfied given the data provided to the SA, the SA will return No Solution Found.
3. Failure rate: the probability that a solution is not safe despite the SA returning a solution it determined to be safe (i.e. it passed the safety test). A  ground truth dataset is required to determine the failure rate. 

All three quantities are plotted against the amount of data input to the algorithm on the horizontal axis.  

Figure 1 shows a supervised learning (classification) example of these three plots: the GPA prediction problem discussed in the Science paper. 

.. figure:: _static/disparate_impact.png
   :width: 100 %
   :alt: disparate_impact
   :align: left

   **Figure 1**: Accuracy (left), solution rate (middle), and failure rate (right) plotted as a function of number of training samples for GPA prediction problem discussed in Thomas et al. 2019. The fairness constraint in this case is disparate impact. Two Seldonain algorithms, Seldonian Classification (green dotted) and Quasi-Seldonian Classification (green dashed), are compared to several standard ML classification algorithms (red) that do not include the fairness constraint. Also shown are two fairness-aware libraries, Fairlearn (blue) and Fairness Constraints (magenta). In this example, only Seldonian algorithms satisfy the disparate impact criteria (right). 



Plot generator
--------------

Depending on the :term:`regime<Regime>` of your problem, i.e. supervised learning or reinforcement learning (RL), the object used to produce the three plots is either :py:class:`.SupervisedPlotGenerator`. or :py:class:`.RLPlotGenerator`, respectively. While the inputs for both of these classes are described in the API documentation, we will describe their inputs in more detail here. 

Regardless of regime, the following inputs are required:

Spec object 
+++++++++++

Often, a `Seldonian interface <https://seldonian-toolkit.github.io/Engine/build/html/overview.html#interface>`_ is used to create the `Spec <https://seldonian-toolkit.github.io/Engine/build/html/overview.html#spec-object>`_ object. The Spec object contains everything that is needed to run the SA, such as the original dataset, the parse trees (containing the behavioral constraints), the underlying machine learning model, etc...

n_trials
++++++++
The number of times the SA is run at each data fraction (point on the horizontal axis, see: `data_pcts`_). Used to estimate uncertainties in the quantities in the three plots. 

data_pcts
+++++++++
A list of fractions of the original dataset size at which to run the SA n_trials times. This list comprises the horizontal axis of each of the three plots. The original dataset is contained within the Spec object. 

datagen_method
++++++++++++++
The method for generating data that is used to run the Seldonian algorithm for each trial. For supervised learning, the only currently supported option for this parameter is "resample". In this case, the original dataset is resampled with replacement n_trials times to obtain n_trials different datasets of the same length as the original dataset. At each data fraction, frac, in data_pcts, the first frac fraction of points in each of the n_trials datasets is used as input to the SA.

For RL, the only currently supported option for this parameter is "generate_episodes". In this case, n_trials new datasets are generated with the same number of episodes as the original dataset. At each data fraction, frac, in data_pcts, the first frac fraction of episodes in each of the n_trials generated datasets is used as input to the SA.


n_workers
+++++++++
The number of parallel workers to use when running an experiment, if multiple cores are available on the machine running the experiment. Because each trial is independent of all other trials, Seldonian experiments are `embarrassingly parallel <https://en.wikipedia.org/wiki/Embarrassingly_parallel>`_ programs. If the number of cores on the machine running the experiment is less than n_workers, then the max number of cores available will be used. 

	
perf_eval_fn
++++++++++++
The function or method used to evaluate the performance of the SA in each trial (plot 1/3). This can be the same as the primary objective specified in the Spec object, but it must be explicitly specified. The only required input to this function is the solution returned by the SA. If NSF is returned for a given trial, then this function will not be evaluated for that trial. 

perf_eval_kwargs
++++++++++++++++
If the perf_eval_fn has more arguments than the solution, pass them as a dictionary in this parameter.

constraint_eval_fns
+++++++++++++++++++
In order to make plot 3/3 (failure rate) the behavioral constraints are evaluated on a ground truth dataset. If this parameter is left as an empty list (default), the constraints will be evaluated using built-in methods in the parse trees. If instead you have custom functions that you want to use to evaluate the behavioral constraints, pass them as a list in this parameter. The list must be the same length as the number of behavioral constraints. 

constraint_eval_kwargs
++++++++++++++++++++++
If your constraint_eval_fns have more arguments than the solution returned by the SA, pass them as a dictionary in this parameter.


results_dir
+++++++++++
The directory in which to save the results of the experiment. 


Files generated in an experiment
--------------

The directory structure inside results_dir will look like this after running an experiment:

.. code::

	├── qsa_results
	│ ├── qsa_results.csv
	│ └── trial_data
	│     ├── data_pct_0.0010_trial_0.csv
	│     ├── data_pct_0.0010_trial_1.csv
	│     ├── data_pct_0.0010_trial_2.csv
	│     ├── data_pct_0.0010_trial_3.csv
	│     ├── data_pct_0.0010_trial_4.csv
	│     ├── data_pct_0.0022_trial_0.csv
	│     ├── data_pct_0.0022_trial_1.csv
	│     ├── data_pct_0.0022_trial_2.csv
	│     ├── data_pct_0.0022_trial_3.csv
	│     ├── data_pct_0.0022_trial_4.csv
	│     ├── data_pct_0.0046_trial_0.csv
	│     ├── data_pct_0.0046_trial_1.csv
	│     ├── data_pct_0.0046_trial_2.csv
	│     ├── data_pct_0.0046_trial_3.csv
	│     ├── data_pct_0.0046_trial_4.csv
	│     ├── data_pct_0.0050_trial_0.csv
	│     ├── data_pct_0.0100_trial_0.csv
	│     ├── data_pct_0.0100_trial_1.csv
	│     ├── data_pct_0.0100_trial_2.csv
	│     ├── data_pct_0.0100_trial_3.csv
	│     ├── data_pct_0.0100_trial_4.csv
	│     ├── data_pct_0.0215_trial_0.csv
	│     ├── data_pct_0.0215_trial_1.csv
	│     ├── data_pct_0.0215_trial_2.csv
	│     ├── data_pct_0.0215_trial_3.csv
	│     ├── data_pct_0.0215_trial_4.csv
	│     ├── data_pct_0.0464_trial_0.csv
	│     ├── data_pct_0.0464_trial_1.csv
	│     ├── data_pct_0.0464_trial_2.csv
	│     ├── data_pct_0.0464_trial_3.csv
	│     ├── data_pct_0.0464_trial_4.csv
	│     ├── data_pct_0.1000_trial_0.csv
	│     ├── data_pct_0.1000_trial_1.csv
	│     ├── data_pct_0.1000_trial_2.csv
	│     ├── data_pct_0.1000_trial_3.csv
	│     ├── data_pct_0.1000_trial_4.csv
	│     ├── data_pct_0.2154_trial_0.csv
	│     ├── data_pct_0.2154_trial_1.csv
	│     ├── data_pct_0.2154_trial_2.csv
	│     ├── data_pct_0.2154_trial_3.csv
	│     ├── data_pct_0.2154_trial_4.csv
	│     ├── data_pct_0.4642_trial_0.csv
	│     ├── data_pct_0.4642_trial_1.csv
	│     ├── data_pct_0.4642_trial_2.csv
	│     ├── data_pct_0.4642_trial_3.csv
	│     ├── data_pct_0.4642_trial_4.csv
	│     ├── data_pct_1.0000_trial_0.csv
	│     ├── data_pct_1.0000_trial_1.csv
	│     ├── data_pct_1.0000_trial_2.csv
	│     ├── data_pct_1.0000_trial_3.csv
	│     ├── data_pct_1.0000_trial_4.csv
	└── resampled_datasets
	    ├── resampled_data_trial0.pkl
	    ├── resampled_data_trial1.pkl
	    ├── resampled_data_trial2.pkl
	    ├── resampled_data_trial3.pkl
	    ├── resampled_data_trial4.pkl

In this example, :code:`n_trials=5` and the default was used for data_pcts, i.e. :code:`np.logspace(-3,0,10)`, which creates an log-spaced array of length 10 starting at :code:`10**(-3)=0.001` and ending at `10**0=1.0`		

