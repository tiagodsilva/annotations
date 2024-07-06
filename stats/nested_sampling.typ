#import "../template.typ": template 
 
#show: doc => template(
  title: "A review of nested sampling", 
  abstract: [Accurately estimating the evidence is the main obstacle in Bayesian model selection. However, standard approaches to Bayesian inference, such as MCMC, are exclusively tailored to posterior sampling and do not provide satisfactory estimates of the evidence. To address this issue, _nested sampling_ frames the evidence as a one-dimensional integral via the tail formula for expectations, which is approximated by quadrature methods. As a byproduct, weighted samples from the posterior are obtained.], 
  doc
)


= Introduction 

Let $pi$ and $cal(L)$ be the prior and likelihood functions defined on a parameter space $Theta$. The evidence is defined as 

$ Z = integral_(theta) pi(theta) cal(L)(theta) dif theta = EE_(theta ~ pi)[cal(L)(theta)]. $  

Define $phi(lambda) = Pr[cal(L) >= lambda]$. Then, 

$ Z = integral_(0)^(infinity) phi(lambda) dif lambda = integral_(0)^(1) lambda(phi) dif phi. $ <tail> 

@tail transforms a potentially high-dimensional problem into a one-dimensional problem, which can be solved via quadrature methods. The challenge now is to estimate $lambda(phi)$. 

= Algorithm 

Skilling (2006) @skilling proposed an algorithm that concomitantly yields quadrature points and estimates of $lambda(phi)$ to approximate @tail. The algorithm works as follows. First, we sample $N$ points from the prior. Then, we select the point with smallest likelihood, $theta_1$, and the corresponding likelihood, $cal(L)(theta_1) = L_1$. Then, we discard $theta_1$ and sample from the constrained prior $pi(theta) 1_(cal(L)(theta) > L_1)$. We proceed in this fashion until we have a collection $(L_1, dots, L_n)$ of likelihoods, which are embedded into the quadrature formula with placeholder points $x_i = exp(-i/n)$. 

= Explanation 

We follow Betancourt's approach @Betancourt2011 to better understand this algorithm. Firstly, define $tilde(alpha) = {alpha colon cal(L)(alpha) > L}$. The prior mass associated to this quantity is denoted by 

$ x(L) = integral_(tilde(alpha))^(infinity) dif^(m) alpha pi(alpha). $ 

Intuitively, the differential $dif x(L)$ may be computed as $integral_(partial tilde(alpha))^(infinity) dif^(m) alpha pi(alpha)$. By letting $alpha_(perp)$ and $alpha_(parallel)$ the coordinates perpendicular and parallel to the surface $partial cal(alpha) = {alpha colon cal(L)(alpha) = L }$, we observe that 

$ dif x(L) = dif alpha_(perp) pi(alpha_(perp)), $ 

i.e., we're simply marginalizing over $alpha_(parallel)$. Under these conditions and noticing that changes in $alpha_(parallel)$ do not affect $cal(L)(alpha)$, the evidence may be computed as 

$ 
  Z = integral dif alpha_(perp) dif^(m - 1) alpha_(parallel) cal(L)(alpha_(perp)) pi(alpha) \ 
  = integral dif alpha_(perp)  cal(L)(alpha_(perp)) integral dif^(m - 1) alpha_(parallel) pi(alpha) \
  = integral dif alpha_(perp)  cal(L)(alpha_(perp)) pi(alpha_(perp)),  
$

which is a one-dimensional integral. By letting $L(x)$ represent the likelihood associated to the prior mass $x$, the prior integral may be written as 

$ 
  Z = integral dif x L(x). 
$

Intuitively, this computation is grounded on Adam's law and the conditioning of $cal(L)(alpha_(perp))$ on $cal(L)(alpha_(perp)) = L$. 

To find a collection of points $(x_k, L_k)$ for numerical integration, we first note that, when the $alpha$ are sampled from $pi$, $x$ are uniformly distributed, 

$ 
  pi(x) = integral_(partial tilde(alpha)) d^(m - 1) alpha_(parallel) pi(alpha(x)) abs(frac(dif alpha, dif x)) \
  = integral_(partial tilde(alpha)) d^(m - 1) alpha_(parallel) pi(alpha(x)) abs(frac(1, pi(alpha_(perp)))) \ 
  = abs(frac(1, pi(alpha_(perp)))) integral_(partial tilde(alpha)) d^(m - 1) alpha_(parallel) pi(alpha(x)) \
  = abs(frac(1, pi(alpha_(perp)(x)))) pi(alpha_(perp)(x)) = 1 
$

when $x in (0, 1)$ (we used the change of variables' formula, the fact that $dif alpha_(parallel)$ does not depend on $x$ and $dif x = dif alpha_(perp) pi(alpha_(perp))$). Then, we notice that our best estimate for the largest value $x_(max)$ is the sample associated with the minimum likelihood. Since $pi(x)$ is uniform, $x_(max)$ has distribution $p(x_(max)) = n x_(max)^(n - 1)$. Given this first sample, which we call $(x_1, L_1)$, the following sample will be distributed as $pi(x) = 1 / x_1$ when $0 <= x <= x_1$. By following this algorith, we note that the shrinkage operators, $t_i = x_i / x_(i - 1)$ are independently and identically distributed with $p(t_k) = n t_k^(n - 1)$. In particular, one may readily notice that 

$ 
log x_k = log x_o + sum log t_k,    
$

and hence that the expected value of $log x_k$ is $-i / n$ (recall that $n$ is the number of samples). This provides the value for the placement points. 

Overall, the algorithm seems to work well in practice; however, a deeper understanding of its convergence rates to the evidence are mostly lacking from the literature. The method is widely adopted in the literature of natural sciences, but is often neglected by standard computational statistics textbooks. Our approach here was quite heuristic. 

Betancourt @Betancourt2011 implemented a constrained HMC algorithm to sample from the likelihood-constrained prior distribution. @lemos2023improvinggradientguidednestedsampling, on the other hand, showed that the HMC scheme may rely on reverse-mode autodiff for evaluateing the gradients and developed an adaptive step-size when simulating the Hamiltonian dynamics. Interestingly, the resulting samples were used to train a GFlowNets with both forward (from a fixed initial state) and backward (from the sampled trajectories) sampling, and the results were quite impressive.  




#bibliography("../bibliography.bib") 
