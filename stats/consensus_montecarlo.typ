#import "../template.typ": template 
 
#show: doc => template(
  title: "A review of nested sampling", 
  abstract: [Consensus Monte Carlo @Scott2016 was introduced as a general-purpose mechanism for distributed Bayesian computation for continuous data sets. In a nutshell, independent shards of data are first assigned to a set of workers, which sample from the corresponding subposterior via MCMC. Then, the obtained samples are communicated to a central server and aggregated to obtain an approximation of the resulting posterior. The aggregation scheme assumes (based on the Berstein-von Mises theorem) that the resulting posterior is approximately multivariate Gaussian.], 
  doc
)

= Introduction 

Multi-threaded algorithms often require specialized programming skills and the development of computer programs that are challenging even for expert software engineers. On the other hand, multi-machine code can be easily scaled and executed without any algorithmic modification of the program. In this case, however, the cost of between-machine communication and program reinialization due to catasthropic failures in hardware becomes a bottleneck. To address these issues, embarrassingly parallel algorithms enable distributed Bayesian inference with minimal communication. 

= Method 

Firstly, we partition the posterior distribution as 
$
  p(bold(theta) | bold(y)) = product_(1 <= s <= S) p(bold(y)_s | bold(theta)) p(bold(theta))^(1/S).  
$

Then, each worker $s in [[1, S]]$ samples $G$ parameters $theta_(s 1), dots, theta_(s G)$ from the corresponding subposterior distribution, $p(bold(y)_s | bold(theta)) p(bold(theta))^(1/S)$. Finally, these samples are aggregated according to the rule for $g in [[1, G]]$ 

$ 
  bold(theta)_g = (sum_s W_s)^(-1) sum W_s theta_(s g), 
$

in which $W_s = Sigma_s^(-1)$ are weights fixed as the inverse of the covariance matrix of each worker. For Gaussian models, this approach yields asymptotically exact samples. For non-Gaussian models, the algorithm is inherently biased. For hierarchical models, the within-group samples should not be partitioned to avoid issues emanating from non-independent samples.  

= Empirical analysis 

In spite of the posited Gaussian nature of the posterior distribution, which is roughly assured by Berstein-von Mises theorem under sufficiently regular conditions, the resulting aggregated distribution is shown to accurately approximate even non-Gaussian targets. The absence of theoretical assurances, however, is quite troublesome for the wide acceptance of such a method. 

#bibliography("../bibliography.bib") 
