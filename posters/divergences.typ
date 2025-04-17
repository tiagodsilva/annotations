  #import "typst-poster/poster.typ": *
  #import "@preview/ouset:0.2.0"

  #let forestgreen = rgb("#228b22")
  #let darkblue = rgb("00008B") 
  #let brickred = rgb("AC1616") 
  #let argmin = [argmin] 
  #let pf = text(fill: brickred)[$p_F (tau)$]
  #let pb = text(fill: forestgreen)[$p_B (tau|x)$]
  #let pbtau = text(fill: forestgreen)[$p_B (tau)$]
  #let KL = text[$mono(K L)$] 
  #let ceq = $ouset.overset(=, "C")$ 
  #let pfprime = text(fill: brickred)[$p_F (tau')$]

  // From https://github.com/Enter-tainer/delimitizer/blob/main/impl.typ
  #let base-size = 1.2em
  #let sizes = (base-size, base-size * 1.5, base-size * 2, base-size * 2.5)

  #let scaled-delimiter(delimiter, size) = math.lr(delimiter, size: size)

  #let big(delimiter) = scaled-delimiter(delimiter, sizes.at(0))
  #let Big(delimiter) = scaled-delimiter(delimiter, sizes.at(1))
  #let bigg(delimiter) = scaled-delimiter(delimiter, sizes.at(2))
  #let Bigg(delimiter) = scaled-delimiter(delimiter, sizes.at(3))

  #show: poster.with(
    size: "48x36",
    title: "On Divergence Measures for Training GFlowNets",
    authors: "Tiago da Silva, Eliezer da Silva, Diego Mesquita",
    departments: none,
    univ_logo: ("../logos/fgv.png",),
    footer_text: "Conference on Advances in Neural Information Processing 2024",
    footer_url: "https://github.com/ML-FGV/divergences-gflownets",
    footer_email_ids: "{tiago.henrique, eliezer.silva, diego.mesquita}@fgv.br",
    footer_color: "ebcfb2", 
    univ_logo_column_size: (8in,),
    univ_logo_column_gutter: (3in,), 
    title_column_size: "32", 
    title_font_size: "88", 
    authors_font_size: "46", 
    // Modifying the defaults
    keywords: ("GFlowNets", "Variational Bayesian inference"),
  )

  #set text(size: 22pt) 

  #block(
    fill: rgb(0, 0, 155, 128),
    inset: 32pt,
    radius: 24pt,
    [
      #text(fill: white)[
      *TL;DR*     
      - we revisit the relationship between GFlowNets and VI in continuous spaces,   
      - we empirically demonstrate that the well-known difficult of training GFlowNets by minimizing traditional divergence measures arises from the large gradient variance of the corresponding stochastic learning objectives,   
      - we develop variance-reduced gradient estimators for $alpha-$ and KL-divergences,  //.
      - we verify that the resulting learning algorithms significantly improve upon conventional log-squared losses in terms of convergence speed,  // .
      - we re-open the once-dismissed research line focused on VI-inspired algorithmic improvements of GFlowNets.  
      ]
    ]
  )

#block(
  inset: 24pt, 
  stroke: none, 
[
  = Background: GFlowNets 

  *GFlowNets* are amortized algorithms for sampling from distributions over compositional objects, i.e., over objects that can be sequentially constructed from an initial state through the application of simple actions (e.g., graphs via edge-addition). 

  #figure(
    image("figures/tb.svg", 
          width: 39%),
    caption: [
      For finite state spaces, the state graph might be represented as a DAG on $cal(S)$. 
    ] 
    // [A GFlowNet learns a #text(fill: brickred)[forward policy] on a state graph.]
  )


  #block(
    fill: none,
    stroke: 2pt + darkblue,   
    inset: 12pt, 
    [
      #text(fill: darkblue)[
        In a nutshell, a GFlowNet is composed of two three ingredients. 

        1. An extension $cal(S)$ of the target distribution support's $cal(X)$. 
        
        2. A _measurable pointed DAG_ $cal(G)$ on $cal(S)$ dictating how the _states_ in $cal(S)$ are connected to one another. We refer to $cal(G)$ as the _state graph_.  

        3. A #text(fill: brickred)[forward] and #text(fill: forestgreen)[backward] policies defining the stochastic transitions within $cal(G)$. 
      ]
    ]
  )

  Our objective is to learn a forward #pf and a backward #pb policies such that the marginal of #pf over $cal(X)$ matches a given unnormalized density $r colon cal(X) arrow.r RR_(+)$.   

  $ 
    #pf = product_((s, s') in tau) p_F (s' | s) #text[ and ] integral_(cal(T)) 1_(tau arrow.r.squiggly x) #pf mono(d)tau = r(x); // . 
  $

  $cal(T)$ denotes the space of trajectories in $cal(G)$ and $tau arrow.r.squiggly$, the event in which $tau$ finishes on $x in cal(X)$.  

  = Background: GFlowNets and VI   

  To ensure that Equation (1) is satisfied, we parameterize #pf with a neural network and search for a parameter configuration satisfying 
  $
    #pf prop #pb r(x)   
  $
  for every trajectory $tau$ finishing on $x$. This condition is sufficient for Equation (1); indeed,    
  $
    integral_(cal(T)) 1_(tau arrow.r.squiggly x) #pf mono(d) tau = integral_(cal(T)) 1_(tau arrow.r.squiggly x) #pb r(x) mono(d) tau = r(x) underbrace(integral_({tau colon tau arrow.r.squiggly x}) #pb mono(d), "= 1") = r(x). 
  $

   #block(
    fill: none,
    stroke: 2pt + darkblue,   
    inset: 12pt, 
    [
      #text(fill: darkblue)[
        When #pb is fixed, Equation (2) corresponds to a standard VI problem having #pf as proposal and $#pbtau colon prop #pb r(x)$ as target. As such, we solve 
        $
          argmin_(p_F in cal(H)) D(#pf, #pbtau) // .  
        $
        for a given divergence measure $D$ and a hypothesis space $cal(H)$ for the policy networks. 
      ]
    ]
  )

  $D$ is conventionally set as the expected log-square difference between #text(fill: brickred)[$p_F$] and #text(fill: forestgreen)[$p_B$], 
  $
    underbrace(cal(L)_(T B)(#pf, #pb), "Trajectory balance objective") colon= D(#pf, #pb) = underbrace(EE_(tau ~ q), "Off-policy sampling") [ ( log frac(Z #pf, #pbtau))^2 ],   
  $
  which requires learning the constant $Z$ corresponding to the partition function of $r$. 


   #block(
    fill: none,
    stroke: 2pt + forestgreen,   
    inset: 12pt, 
    [
      #text(fill: forestgreen)[
        Attempts to utilize traditional divergence measures, such as the $alpha-$ and KL-divergences, for training GFlowNets have failed. Our objectives are to *understand the reason for this* and *improve the effectiveness of these learning objectives*. 
      ]
    ]
  )

])

= KL-, Renyi-$alpha$, and Tsallis-$alpha$ divergences 

We investigate the training of GFlowNets with four different divergence measures: forward KL, reverse KL, Renyi-$alpha$ and Tsallis-$alpha$, which we recall in the next definitions. 

#block(
    fill: none,
    stroke: 2pt + darkblue,   
    inset: 12pt, 
    [
      #text(fill: darkblue)[
        *Forward and reverse Kullback-Leibler divergences* are respectively defined as 
        $
          cal(D)_#KL (#pbtau || #pf) = EE_(tau ~ #pbtau) [ log frac(#pbtau, #pf)] #text[ and ] cal(D)_#KL (#pf || #pbtau) = EE_(tau ~ #pf) [ log frac(#pf, #pbtau)]. 
        $ 
      ]
    ]
  )

  
  #block(
    fill: none,
    stroke: 2pt + darkblue,   
    inset: 12pt, 
    [
      #text(fill: darkblue)[
        *Renyi-$alpha$ divergence* is defined as 
        $
          cal(R)_(alpha) (#pf || #pbtau) = 1 / (alpha - 1) log integral_(cal(T)) #pf^alpha #pbtau^(1 - alpha) mono(d)tau. 
        $ 
      ]
    ]
  )

  #block(
    fill: none,
    stroke: 2pt + darkblue,   
    inset: 12pt, 
    [
      #text(fill: darkblue)[
        *Tsallis-$alpha$ divergence* is defined as 
        $
          cal(T)_(alpha) (#pf || #pbtau) = 1 / (alpha - 1) ( integral_(cal(T)) #pf^alpha #pbtau^(1 - alpha)  mono(d)tau - 1). 
        $ 
      ]
    ]
  )

  Importantly, we can only evaluate each of these divergences up to a (multiplicative or additive) constant, as #pbtau cannot be directly computed. We demonstrate that, for the adaptive gradient-based optimization algorithms utilized in practice, this is not an issue.   


  #block(
    fill: none,
    stroke: 2pt + darkblue,   
    inset: 12pt, 
    [
      #text(fill: darkblue)[
        Let $theta$ be the parameters of the neural network parameterizing $p_F$ and 
        $
          b(tau ; theta) = frac(#pf, #pb r(x)) #text[ and ] s(tau ; theta) = log #pf. 
        $
        Then, 
        $
          nabla_theta cal(D)_KL (#pbtau, #pf) #ceq - EE_(tau ~ #pf) [b(tau ; theta) nabla_theta s(tau ; theta)] #text[ and ]
           // .  
        $
        $
          nabla_theta cal(D)_KL (#pf, #pbtau) = EE_(tau ~ #pf) [ nabla_theta s(tau ; theta) + log b(tau ; theta) nabla_theta s(tau ; theta)], // .  
        $
        in which #ceq denotes equality up to a multiplicative constant. 
      ]
    ]
  )

  For the Renyi-$alpha$ and Tsallis-$alpha$ divergences, we similarly derive

  $
    nabla_theta cal(R)_alpha (#pf || #pbtau) = frac(
      EE[nabla_theta b(tau ; theta)^(alpha - 1) + b(tau ; theta)^(alpha - 1) nabla_theta s(tau ; theta)], (alpha - 1) EE[b(tau ; theta)^(alpha - 1)] 
    )  
  $
  and 
  $
    nabla_theta cal(T)_alpha (#pf || #pbtau) #ceq frac(
      EE[nabla_theta b(tau ; theta)^(alpha - 1) + b(tau ; theta)^(alpha - 1) nabla_theta s(tau ; theta)], (alpha - 1)   
    ).   
  $

= Variance-reduced gradient estimation of divergences 

  To train a GFlowNet, we stochastically update #pf with Monte Carlo estimates of the above gradients. However, this suffers from high variance and training instability; see Fig. 2. 

  #figure(
    image("figures/variance_losses.svg", 
          width: 99%),
    caption: [
      Monte Carlo estimation of the gradients leads to highly unstable training.  
    ] 
    // [A GFlowNet learns a #text(fill: brickred)[forward policy] on a state graph.]
  )    

  To mitigate these issues, we devise control variates (CVs) for provable variance-reduced and unbiased gradient estimation of divergence measures in the context of GFlowNets. 
  
  #block(
    fill: none,
    stroke: 2pt + darkblue,   
    inset: 12pt, 
    [
      #text(fill: darkblue)[
        A CV is a zero-expectation random variable. As our objective is to estimate $EE_(tau ~ #pf)[f(tau)]$ for some $f$,  we introduce $nabla_theta log #pf$ as our CV and select a _baseline_ $a$ s.t.  
        $
          f(tau) - a dot.c nabla_theta log #pf 
        $
        has minimum variance. We demonstrated that the optimal choice for $a$ is 
        
        #math.equation(numbering: none, block: true)[
        $
          a^star = argmin_a #text[Tr] #text[Cov] [ f(tau) - a dot.c nabla_theta log #pf ] = frac(
            EE_(#pf) [ (nabla_theta log #pf)^(T) (f(tau) - EE_(#pfprime) (f(tau')) )] , 
            EE_(#pf)[ (nabla_theta log #pf)^(T) (nabla_theta log #pf) ]
          ). 
        $
        ]

        A Monte Carlo approximation of this quotient cannot be written as a Jacobian-vector product. Hence, it *cannot be efficiently computed in autodiff frameworks*. To circumvent this, we utilize the first-order delta method to obtain a linear approximation to $a^star$, // . 

        $
          hat(a) = frac( 
            angle.l.curly sum_(n=1)^(N) nabla_theta s(tau_n; theta) \, sum_(n=1)^(N) nabla_theta f(tau_n) angle.r.curly,
            epsilon + || sum_(n=1)^(N) nabla_theta s(tau_n ; theta) ||^(2) 
        ) // . 
        $

        (given a batch ${tau_1, dots, tau_N}$ of trajectories). As $nabla_theta f(tau_n)$ and $nabla_theta s(tau_n ; theta)$ are naturally computed by standard learning algorithms, the computational overhead imposed by $hat(a)$ is negligible. // The same can be said about the small bias incurred from the use of a batch-based estimate of $a^star$. 
      ]
    ]
  )

  #block(
    fill: none,
    stroke: 2pt + darkblue,   
    inset: 12pt, 
    [
      #text(fill: darkblue)[
        *Leave-one-out (LOO) estimator.* We also construct a sample-dependent baseline $a$ based on the LOO estimator. Given a batch ${tau_1, dots, tau_N}$ of trajectories, we let 
        $
          a(tau_i) = frac(1, N - 1) sum_(n=1, n != i)^(N) f(tau_n). 
        $
        Clearly, the i.i.d.-ness of the trajectories ensures the unbiasedness of the resulting estimator, which can be efficiently computed in an autodiff framework through the formula 
        $
          nabla_theta frac(1, N) bigg(angle.l.curly) #text[sg] ( bold(f) - frac(1, N - 1) (bold(1) - bold(I)) bold(f) ), bold(p) bigg(angle.r.curly), // . 
        $
        in which $bold(f) = (f(tau_i))_(i=1)^(N)$ and $bold(p) = (nabla_theta s(tau_i ; theta))_(i=1)^(N)$ are computed during training. 
      ]
    ]
  )

  For a REINFORCE-based expectation of the form $EE[ nabla_theta f(tau) + f(tau) nabla_theta log p_F (tau)]$, we utilize a sample-invariant baseline for the first term and a LOO baseline for the second term. // Figure below shows that the resulting is drastically more statistically efficient than its non-controlled counterpart. 

  #figure(
    image("figures/variance_reduction.svg", 
          width: 99%),
    caption: [
      Our proposed CVs drastically reduce the gradient variance.  
    ] 
    // [A GFlowNet learns a #text(fill: brickred)[forward policy] on a state graph.]
  )    

= Experimental analysis 

#figure(
    image("figures/line_plots_divergences.svg", 
          width: 99%),
    caption: [
      Our proposed CVs drastically reduce the gradient variance.  
    ] 
    // [A GFlowNet learns a #text(fill: brickred)[forward policy] on a state graph.]
  )    
#figure(
    image("figures/learning_objectives_topk_divergences.svg", 
          width: 99%),
    caption: [
      Our proposed CVs drastically reduce the gradient variance.  
    ] 
    // [A GFlowNet learns a #text(fill: brickred)[forward policy] on a state graph.]
  )    


  #block(
    fill: none,
    stroke: 2pt + darkblue,   
    inset: 12pt, 
    [
      #text(fill: darkblue)[
        Our proposed estimators lead to faster training convergence (in terms of the TV distance between the sampling and target distributions) and mode coverage wrt traditional log-squared-based learning objectives across a broad range of generative tasks. 
      ]
    ]
  )

= What lies ahead? 


  #block(
    fill: none,
    stroke: 2pt + forestgreen,   
    inset: 12pt, 
    [
      #text(fill: forestgreen)[
        Our work narrows the gap between GFlowNets and VI and paves the way for the development of VI-based enhancements of GFlowNet training. We believe that the design of importance-weighted learning objectives is a promising direction for future works.  
      ]
    ]
  )


  #block(
    fill: none,
    stroke: 2pt + brickred,   
    inset: 12pt, 
    [
      #text(fill: brickred)[
        *On-policy vs. off-policy learning.* Albeit effective, divergence-based objectives are constrained to either on-policy or simple off-policy (e.g., $epsilon$-greedy) sampling strategies, which hampers the performance of the trained GFlowNet for highly sparse target distributions. Choosing a loss function is a problem that should be considered in a case-by-case basis. 
      ]
    ]
  )
