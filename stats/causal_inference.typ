#import "../template.typ": template 
 
#let cov = [Cov]

#show: doc => template(
  title: "A review of nested sampling", 
  abstract: [Causal inference is a often-negleted topic from standard statistical textbooks. Historically, there have been two frameworks for thinking about causality: the potential outcomes  framework, developed by Neyman and Rubin from 1923 to 1950s, and the structural framework, introduced by Judea Peal. In this short document, we introduce the structural causal model, causal effects, counterfactual analysis and the do operator. Also, we show how Neyman's potential outcome framework may be interpreted as a special case of the Pearl's structural equations.], 
  doc
)


= Causal vs. Associational 

Causal questions are questions that cannot be answered from the joint distribution of the observed variables alone. Properties that can be derived exclusively from the calculus of probabilities are associtational in nature and do not require further theoretical development. Randomization, influence, effect and "holding constant" are examples of causal concepts; "controlling for", conditionalization and likelihood are examples of associational concepts. 

= Structural causal models 

To deal with causal queries, we must extend the syntax of probability calculus. @fig:structural shows an example of a structural model; there, $U_X$ and $U_Y$ are exogeneous variables which cannot be directly measured, and $X$ and $Y$ are the endogeneous variables. When $U_X$ and $U_Y$ are not uncorrelated, we may estimate the _direct effect_ $beta$ by $beta = cov(X, Y)$ (we are assuming that variables are centered with unit variance). 

#figure(
  image(
    "images/structural.png"
  ), 
  caption: [Example of structural model.]
) <fig:structural> 


= Interventions 

= Counterfactual Analysis and Causal Effects  

A causal effect is measured as the difference in expectation between a target variable given the operated treatment and given a counterfactual treatment.   

= Potential outcomes 