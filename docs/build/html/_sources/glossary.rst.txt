Glossary
========

.. glossary::
    :sorted: 

    Behavioral constraint
      Criteria for fairness or safety provided by the user. The Seldonian algorithm ensures that constraints are met while simultaneously optimizing the primary objective function.

    Candidate Selection
      TBD

    Confidence level
      Often simply called delta. Provided by the user, the confidence level is used to define the probability (1-delta) with which the behavioral constraints are to be satisfied by the seldonian algorithm.  

    Delta
      See :term:`Confidence level`

    Interface
      The system with which the user interacts to provide the behavioral constraints and other inputs to the Seldonian algorithm.

    Machine learning model 
      The model from the normal machine learning paradigm that is adopted by the Seldonian Algorithm to make predictions from features (supervised learning) and apply policies (reinforcement learning).

    Primary objective function
      The objective function (also called loss function) that the machine learning model optimizes, e.g. mean squared error.  

    Regime
      The machine learning problem type, e.g. supervised learning or reinforcement learning. 

    Safety test
      TBD

    Seldonian algorithm
      An algorithm designed to enforce high probability constraints in a machine learning problem

    Sensitive attribute
      In a fairness constraint, a sensitive attribute is one against which the model should not discriminate. Gender and race are common examples. Also sometimes called the protected attribute. Only pertains to supervised learning.

    Sub-regime
      Within supervised learning, the sub-regimes supported by this library are classification and regression. Reinforcement learning does not have sub-regimes in this library. 
    
