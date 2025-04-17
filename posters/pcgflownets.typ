  #import "typst-poster/poster.typ": *

  #let forestgreen = rgb("#228b22")
  #let darkblue = rgb("00008B") 
  #let brickred = rgb("AC1616") 
  #let argmin = [argmin] 
  #let pf = text(fill: brickred)[$p_F (tau)$]
  #let pb = text(fill: forestgreen)[$p_B (tau|x)$]

  #show: poster.with(
    size: "48x36",
    title: "Generalization and Distributed Learning of GFlowNets",
    authors: "Tiago da Silva, Amauri Souza, Omar Rivasplata, Vikas Garg, Samuel Kaski, Diego Mesquita",
    departments: none,
    univ_logo: ("../logos/fgv.png", "../logos/aalto.svg", "../logos/ifce.png", "../logos/manchester.png"),
    footer_text: "International Conference on Learning Representations 2025",
    footer_url: "https://github.com/ML-FGV/pc-gflownets",
    footer_email_ids: "{tiago.henrique, diego.mesquita}@fgv.br, {omar.rivasplata}@manchester.ac.uk, {amauri.souza, sami.kaski}@aalto.fi",
    footer_color: "ebcfb2", 
    univ_logo_column_size: (5in, 2.5in, 3.5in, 5in),
    univ_logo_column_gutter: (-3.5in, -.1in, -.5in, -.1in, -.1in), 
    title_column_size: "32", 
    title_font_size: "88", 
    authors_font_size: "46", 
    // Modifying the defaults
    keywords: ("GFlowNets", "Distributed learning, PAC-Bayes"),
  )

  #set text(size: 27pt) 

  #block(
    fill: rgb(0, 0, 155, 128),
    inset: 32pt,
    radius: 24pt,
    [
      #text(fill: white)[
      *TL;DR*     
      - we introduce the first non-vacuous generalization bounds for GFlowNets, 
      - we develop the first Azuma-type PAC-Bayesian bounds for understanding the generalization of GFlowNets under the light of Martingale theory, 
      - we demonstrate the harmful effect of the trajectory length on the proven learnability of a generalizable policy for GFlowNets,  
      - we introduce the first distributed algorithm for learning GFlowNets, Subgraph Asynchronous Learning, and show that it drastically accelerates learning convergence and mode discovery when compared against a centralized approach for relevant benchmark tasks
      ]
    ]
  )

#block(
  inset: 24pt, 
  stroke: none, 
)[
  = Background: GFlowNets 

  *GFlowNets* are amortized algorithms for sampling from distributions over discrete and compositional objects (such as graphs). 

  #v(12pt)

  #figure(
    image("figures/tb.svg", 
          width: 45%),
    caption: [A GFlowNet learns a #text(fill: brickred)[forward policy] on a state graph.]
  )

  #v(12pt)

  A *flow network* is defined over an extension $cal(S)$ of $cal(G)$, which then represents the sink nodes. To navigate through this network and sample from $cal(G)$ in proportion to a *reward function* $R colon cal(G) arrow.r RR_(+)$, a forward (resp. backward) policy #pf (#pb) is used. 

  $ 
  #pf = product_((s, s') in tau) p_F(s' | s) #text[ and ] sum_(tau arrow.r.squiggly g) #pf = R(g). 
  $

  To achieve this, we parameterize $#pf$ as a neural network trained by minimizing 

  $ cal(L)_(T B)(p_F) = EE [ ( log frac(#text(fill: brickred)[$p_F (tau)$] Z, #text(fill:forestgreen)[$p_B (tau | x)$] R(x)))^(2) ]. $ 

  for a given #pb. GFlowNets can be trained in an *off-policy* fashion and the above expectation can be under any full-support distribution over trajectories. 
]

  = Background: Probably Approximately Correct Bayesian Bounds   

  Let $cal(L)$ be a loss function on a parameter space $Theta$, e.g., the squared loss. Also, let $hat(cal(L))(theta, bold(X))$ be its empirical counterpart evaluated on a dataset $bold(X)$. \
  
  
  #block(
    fill: none,
    stroke: 2pt + darkblue,   
    inset: 12pt, 
    [
      #text(fill: darkblue)[
      *PAC-Bayesian bounds*. Given "prior" $Q$ (independent of $bold(X)$) and posterior $P$ distributions over $Theta$, a PAC-Bayesian bound establishes an upper limit for the expectation of (unobserved) $cal(L)$ based on the (observed) $hat(cal(L))$ and a complexity term $phi$ and a confidence level $delta$, 
      $
        EE_(theta tilde P)[cal(L)(theta)] <= EE_(theta tilde P)[hat(cal(L))(theta, bold(X))] + phi(delta, P, Q, |bold(X)|). 
      $
      ]
    ]
  )

  When $cal(L)(theta) <= B$ a.e., we refer to a bound as _vacuous_ if   

  $
    EE_(theta tilde P)[hat(cal(L))(theta, bold(X))] + phi(delta, P, Q, |bold(X)|) >= B. 
  $

  Otherwise, the bound is _non-vacuous_. Historically, the search for non-vacuous PAC-Bayesian bounds has been associated to the search for provably generalizable learning algorithms. In this regard, recent works have built upon the basic PAC-Bayesian inequalities to obtain theoretical guarantees for GANs, transformers, armed bandits, and variational autoencoders. 

  
  #block(
    fill: none,
    stroke: 2pt + darkblue,   
    inset: 12pt, 
    [
      #text(fill: darkblue)[
      *Data-dependent priors for PAC-Bayesian bounds*. Often, $phi$ involves the KL divergence between $P$ and $Q$, which dominates the upper bound and commonly results in vacuously true statements. To circumvent this issue, we separate $bold(X) = bold(X)_(alpha) union bold(X)_(1 - alpha)$ into disjoint and independent subsets. A posterior $P$ is learned on $bold(X)_(1 - alpha)$ through conventional methods and a prior $Q$ is subsequently learned on $bold(X)_(alpha)$ by minimizing the PAC-Bayesian upper bound, 
      $
        Q^(star) = argmin_Q EE_(theta tilde P)[hat(cal(L))(theta, bold(X)_(alpha))] + phi(delta, P, Q, alpha |bold(X)|). //,  // . 
      $
      ]
    ]
  )
  
  = Non-vacuous Generalization Bounds for GFlowNets 

  There are four ingredients for a PAC-Bayesian bound: a bounded risk functional $cal(L)$, a prior distribution $Q$, a posterior distribution $P$, and a learning algorithm. In alignment with the broader literature, we use a diagonal Gaussian distribution for both $P$ and $Q$ with fixed (small) variance. For learning, we use SGD. 

  
  #block(
    fill: none,
    stroke: 2pt + darkblue,   
    inset: 12pt, 
    [
      #text(fill: darkblue)[
      *A bounded risk functional for GFlowNets*. In a recent work, we demonstrated that Flow Consistency in Subgraphs (FCS) is a sound and tractable learning objective for GFlowNets.

        $
          "FCS"(pi, p_top) = EE_(cal(B))  [ "TV"(pi^(cal(B)), p_top^(cal(B))) ], // .
        $ 

        in which $pi prop R$ is the target distribution and 
        $
          p_top(x) = sum_(tau "finishing at" x) p_F(tau)
        $  
        is the probability of $x in cal(G)$ under $p_F$; the expectation is under a distribution of random independent subsets of $cal(G)$. Intuitively, FCS measures the total variation $"TV"$ between $pi$ and $p_top$ on random subgraphs of the underlying flow network.
      ]
    ]
  )
   
  The stochastic and bounded nature of FCS make it a suitable candidate for pursuing a PAC-Bayesian analysis of GFlowNets. We will refer to FCS by $L$ and to its empirical counterpart by $hat(L)$ to emphasize its use as a risk functional for assessing the generalization of GFlowNets.      
  

  #block(
    fill: none,
    stroke: 2pt + brickred,   
    inset: 12pt, 
    [
      #text(fill: brickred)[
      *Non-vacuous generalization bounds*. Let $cal(T)_n$ be a $n$-sized set of trajectories sampled from a stationary distribution. Also, let $P$ and $Q$ be distributions over $Theta$. Then, 

      $
        L_"FCS" (P) <= hat(L)_"FCS" (P, cal(T)_(1 - alpha)) + min(
          cases(
            eta + sqrt(eta(eta + 2 hat(L)_"FCS" (P, cal(T)_(1 - alpha)))),
            sqrt(eta/2),
          )
        )
      $ <riskfunctional>

      in which the complexity term $eta$ is, for chosen $alpha in (0, 1)$, 

      $
        eta colon = frac("KL"(P || Q) + log frac(2 sqrt(floor((1 - alpha) n)), delta), floor((1 - alpha) n)). 
      $    
      ]
    ]
  )

  We optimize @riskfunctional to obtain data-dependent priors $Q$ over $Theta$. @nonvacuous shows the resulting bounds are non-vacuous. These are the first positive and rigorous results regarding the genearlization GFlowNets in the literature. 

  #figure(
    image("figures/bayesian_learning.svg", width: 35%),
    caption: [Non-vacuous generalization bounds for GFlowNets.]
  ) <nonvacuous> 


  #block(
    fill: none,
    stroke: 2pt + brickred,   
    inset: 12pt, 
    [
      #text(fill: brickred)[
      *Oracle generalization bounds for GFlowNets*. Let $cal(L)$ be the within-trajectory detailed balance loss function and assume that $cal(L) <= U$ a.e.. Additionally, define $t_m$ as the maximum trajectory length within the flow network and $T$ as a budget for the number of transitions.  
      
      $
        EE_(theta ~ P) [ cal(L)(theta) ] <= frac(1, beta) EE_(theta ~ P) [ hat(cal(L))(theta) ] + alpha_(T, n) ( "KL"(P || Q) + log frac(2, delta) ) + frac(log t_m, beta T lambda) + gamma frac(lambda 2U^2, beta T) 
      $ <oracles> 

      in which $beta in (0, 1)$, $lambda > 0$, and 

      $
        alpha_(T, n) =  frac(U, 2 beta (1 - beta) n) + frac(1, beta T lambda).
      $
      ]
    ]
  )

  = Subgraph Asynchronous Learning (SAL) 

  @oracles demonstrates that the larger trajectory length $t_m$ play a key role in constraining the generalization potential of GFlowNets. To mitigate this effect, we propose a distributed divide-and-conquer learning algorithm that breaks up the state graph into smaller subgraphs and learns a GFlowNet for each subgraph. The resulting GFlowNets are aggregated in a final, efficient step. We refer to this approach as *Subgraph Asynchronous Learning* (SAL). 

  #grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    [
      #figure(
        image("figures/multipage/stategraphshierarchy-1.png", width: 100%),
        caption: [An illustration of SAL.]
      )
    ],
    [
      #figure(
        image("figures/subgraphs.png", width: 100%)
      )
    ]
  )
  

  #block(
    fill: none,
    stroke: 2pt + brickred,   
    inset: 12pt, 
    [
      #text(fill: brickred)[
      *SAL*. Let ${ (p_F^1, F_1), ..., (p_F^m, F_m) }$ be $m$ GFlowNets defined over each of the $m$ components of the partition defining SAL. Also, let $q_j$ be a distribution over the initial states and $p_E^j$ be an distribution over trajecotires within the $j$th component $S_j$. Then, each $(p_F^j, F_j)$ is learned by minimizing the _amortized trajectory balance loss_ over $S_j$
      $
        cal(L)_("ATB")^j (p_F^j, F_j) = EE_(s ~ q_j) EE_(tau ~ p_E^j(dot | s)) [ ( log frac(F_j (s) p_F^j (tau | s), R(x) p_B^j (tau | x)) )^2 ], // .
      $
      which replaces $Z$ by the flow function $F^j$ in the conventional $cal(L)_(T B)$. 
      ]
    ]
  )

  Our experimental results in @salreinforcements and @salsequences show that SAL often accelerate training convergence and increase the mode-finding capability of the GFlowNet. These observations are in alignment with our theoretical analysis in @oracles (@salreinforcements) and corroborate the intuition that a divide-and-conquer approach enhances the exploration of the learning agent (@salsequences).  

  #figure(
    image("figures/asynchronous_learning_grids_large.svg", width: 100%),
    caption: [SAL improves learning convergence for the hypergrid environment.]
  ) <salreinforcements> 

  #figure(
    image("figures/asynchronous_learning_time_analysis_single_row=True.svg", width: 100%),
    caption: [SAL drastically accelerates mode-finding for benchmark tasks.]
  ) <salsequences> 

  *Recursive SAL*. SAL can be hierarchically extended to accommodate nested partitions of the state graph. This is illustrated in Figure [ref], and the resulting method is referred to as Recursive SAL. Notably, the depth of the nested partition characterizes a trade-off between the number of trainable models and the difficult of the problem that each model solves. The balance between these factors should be addressed in a case-by-case basis in future endeavors.    

  #figure(
    image("figures/multipage/stategraphshierarchy-2.png", width: 70%), 
    caption: [Illustration of Recursive SAL as an hierarchical extension of SAL.]
  ) 

  // = Take-home message
