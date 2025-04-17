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

  #set text(size: 32pt) 

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

  #figure(
    image("figures/tb.svg", 
          width: 45%),
    caption: [A GFlowNet learns a #text(fill: brickred)[forward policy] on a state graph.]
  )


  Briefly, a *flow network* is defined over an extension $cal(S)$ of $cal(G)$, which then represents the sink nodes. To navigate within this network and sample from $cal(G)$ proportionally to a *reward function* $R colon cal(G) arrow.r RR_(+)$, a forward (resp. backward) policy #pf (#pb) is used. 

  $ 
  #pf = product_((s, s') in tau) p_F(s' | s) #text[ and ] sum_(tau arrow.r.squiggly g) #pf = R(g). 
  $

  To achieve this, we parameterize $#pf$ as a neural network trained by minimizing 

  $ cal(L)_(T B)(p_F) = EE [ ( log frac(#text(fill: brickred)[$p_F (tau)$] Z, #text(fill:forestgreen)[$p_B (tau | x)$] R(x)))^(2) ]. $ 

  for a given #pb. GFlowNets can be trained in an *off-policy* fashion and the above expectation can be under any full-support distribution over trajectories. 
]

  = Background: Probably Approximate Correct Bayesian Bounds   

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
      *Data-dependent priors for PAC-Bayesian bounds*. Given "prior" $Q$ (independent of $bold(X)$) and posterior $P$ distributions over $Theta$, a PAC-Bayesian bound establishes an upper limit for the expectation of (unobserved) $cal(L)$ based on the (observed) $hat(cal(L))$ and a complexity term $phi$ and a confidence level $delta$, 
      $
        EE_(theta tilde P)[cal(L)(theta)] <= EE_(theta tilde P)[hat(cal(L))(theta, bold(X))] + phi(delta, P, Q, |bold(X)|). 
      $
      ]
    ]
  )
