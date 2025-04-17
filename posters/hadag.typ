  #import "typst-poster/poster.typ": *

  #let forestgreen = rgb("#228b22")
  #let darkblue = rgb("00008B") 
  #let brickred = rgb("AC1616") 
  #let argmin = [argmin] 
  #let pf = text(fill: brickred)[$p_F (tau)$]
  #let pb = text(fill: forestgreen)[$p_B (tau|x)$]

  #show: poster.with(
    size: "24x36",
    title: "Human-aided Discovery" + linebreak(justify: false) + "of Ancestral Graphs",
    authors: "Tiago da Silva, Eliezer da Silva, Antonio Góis," +  linebreak(justify: false) +  "Samuel Kaski, Dominik Heider, Diego Mesquita, Adèle Ribeiro",
    footer_email_ids: [], 
    departments: none,
    univ_logo: ("../logos/aalto.png", "../logos/marburg.png", "../logos/latinx.png", "../logos/fgv.png", "../logos/mila.png", "../logos/neurips_logo.png"),
    footer_text: "LatinX @ NeurIPS 2024",
    footer_url: "https://github.com/ML-FGV/agfn",
    footer_color: "ebcfb2", 
    univ_logo_column_size: (5in, 5in, 2.5in), 
    univ_logo_column_gutter: (-3.5in, -.25in, -.25in, -.05in, -.05in), 
    univ_logo_grid_row_size: (1.8in, 2in),  
    univ_logo_grid_col_size: (3.6in, 3.6in, 2in), 
    univ_logo_scale: (50%, 50%, 80%, 110%, 110%, 80%), 
    title_column_size: "16", 
    title_font_size: "64", 
    authors_font_size: "32", 
    num_columns: "2", 
    // Modifying the defaults
    keywords: ("Causal Discovery", "Human in the loop", "Probabilistic inference"), 
  )

  #set text(size: 22pt) 

  #block(
    fill: rgb(0, 0, 155, 128),
    inset: 32pt,
    radius: 24pt,
    [
      #text(fill: white)[
      *TL;DR*     
        - we introduce *Ancestral GFlowNets* (AGFNs) as a new amortized inference method for sampling from a belief distribution on the space of ancestral graphs, 
        - we develop the first human-in-the-loop framework for ancestral causal discovery (CD), 
        - we design an optimal strategy for elicitation of an expert's feedback regarding the nature of a specific causal relationship among the observed variables, 
        - we demonstrate that our human-aided CD method drastically outperforms traditional CD algorithms after just a few expert interactions.  
      ]
    ]
  )

#block(
  inset: 24pt, 
  stroke: none, 
[
  = Background: Causal Discovery  

  Let $bold(X) in RR^(n times d)$ be a $d$-dimensional i.i.d. data set. 
  A *causal discovery* (CD) algorithm takes $bold(X)$ as input and returns 
  a _causal diagram_ over the variables $cal(V) = {1, dots, d}$ of $bold(X)$. 

  #align(
    center, 
    grid(
      rows: (1.8in, .15in), columns: 3,  
      align(horizon, image("figures/chain4.svg")), 
      align(right, image("figures/collfork.svg")), 
      align(left, image("figures/iv.svg")),   
      grid.cell(colspan: 3, align(center, text(size: 18pt)[Examples of ancestral graphs.]))  
      )
  )

  
  In the absence of causal sufficiency, ancestral graphs (AGs) are used to represent both ancestral causal relationships (directed edges) and associations due to latent confounding (bidirected edges) among variables.  

  #block(
      fill: none,
      stroke: 2pt + darkblue,   
      inset: 12pt, 
      [
          #text(fill: darkblue)[*We take a Bayesian stance* and estimate a 
          probability distribution over the space $cal(G)$ of AGs on $cal(V)$. For this, we introduce a _score function_ 
          $s colon RR^(n times d) times cal(G) arrow.r RR$ and define the posterior distribution over the space of AGs as 
          $
            pi(G | bold(X)) prop exp(s(bold(X), G) ).  
          $
	  Alas, exact Bayesian inference on $pi$ is not possible. Instead, we use a GFlowNet to tractably approximate $pi$.
        ]
      ]
    )

  = Background: GFlowNets    
  
  *GFlowNets* are amortized algorithms for sampling 
  from unnormalized distributions on a 
  compositional space $cal(G)$. 

  #figure(
    image("figures/tb.svg", width: 39%)
  ) <gfn> 

  We construct a *state graph* on the extended space ${s_o} union cal(S) union cal(G)$ 
  endowed with an _initial state_ $s_o$. Then, we learn a #text(fill: brickred)[forward] 
  (#text(fill: forestgreen)[backward]) $pf$ (#pb) policy s.t., for every $g in cal(G)$,  

  $ 
    #pf = product_((s, s') in tau) p_(F) (s' | s) #text[ and ] sum_(tau arrow.r.squiggly g) #pf prop R(g), // . 
  $

  in which $tau arrow.r.squiggly g$ is a trajectory starting at $s_o$ and finishing at $g$. @gfn illustrates a state graph on $cal(G) = {g_1, g_2, g_3}$.  

#block(
      fill: none,
      stroke: 2pt + darkblue,   
      inset: 12pt, 
      [
          #text(fill: darkblue)[To achieve this, we parameterize #pf and #pb as neural networks trained by stochastically minimizing   
          $ 
            cal(L)_(T B)(p_F, p_B) = EE [ ( log frac(#text(fill: brickred)[$p_F (tau)$] Z, #text(fill:forestgreen)[$p_B (tau | x)$] R(x)))^(2) ]. 
          $ 
        ]
      ]
    )


])

  = Ancestral Generative Flow Networks 

  AGFN builds upon a GFlowNet to approximate the posterior in (1); it is composed of a *state graph* and a *score function*.  

  1. The *state graph* (SG) is defined by an edge-addition process illustrated below. Importantly, we remove the transitions leading to non-ancestral graphs from the SG.   

  2. Given a model $f(bold(X) | cal(G), theta)$ indexed by parameters $theta$, we define the score function $s$ as the opposite of the BIC, i.e.,  
  $
    s(bold(X), G) = 2 max_(theta) f(bold(X) | cal(G), theta) - |E| log n - 2 |E| log |V|,   
  $ 
  in which $G = (V, E)$ and $n$ is the size of $bold(X)$. In this work, $f(dot.c | cal(G), theta)$ is represented by a Gaussian Structural Equation Model.  

  #figure(
    image("figures/agfn.svg",  width: 70%), 
    caption: [AGFN iteratively adds edges to an initially edgeless AG. In doing so, it ensures the sampled graphs' ancestrality.] 
  )

  #block(
    fill: none,
    stroke: 2pt + darkblue,   
    inset: 12pt, 
    [
      #text(fill: darkblue)[
        In contrast to prior art, AGFN is strictly supported on the space of AGs. In this regard, it is the *only probabilistic method* suitable for Bayesian ancestral causal discovery.    
      ]
    ]
  )

  = Optimal Knowledge Elicitation 

  Our human-in-the-loop framework has two ingredients. 

  1. A model of a *potentially noisy expert*: for variables $V, W$, 
  $
    q(V hat(cal(R)) W | cal(R)) = pi dot.c 1_(V hat(cal(R)) W = V cal(R) W) + ( (1 - pi) / 3 ) dot.c 1_(V hat(cal(R)) W != V cal(R) W)
  $
  in which $cal(R) in {arrow.r, arrow.l, arrow.l.r, emptyset}$ ($hat(cal(R))$) is the expert-provided (estimated) relationship between $V$ and $W$; $pi$ is an hyperparameter.

  2. A *scheme for integrating the expert's knowledge* into AGFN's learned model. Given feedbacks $cal(F) = {V_i cal(R)_i W_i}_(i=1)^(n)$, 
  $
    p(G | cal(F)) prop underbrace(p(G), "AGFN") product_(1 <= i <= n) q(underbrace(V_i cal(R)_i^(G) W_i, "Relation in G") | underbrace(V_i cal(R)_i W_i, "Feedback")). 
  $

  #figure(
    image("figures/hitlpipeline.svg", width: 60%), 
    caption: [We progressively refine the learned AGFN through the incorporation of feedbacks from an human expert.] 
  )

#block(
    fill: none,
    stroke: 2pt + darkblue,   
    inset: 12pt, 
    [
      #text(fill: darkblue)[
        We probe the expert on the relation $cal(R)$ minimizing the cross-entropy between distributions $p(dot.c | cal(F) union  {cal(R)})$ and $p(dot.c | cal(F))$.    
      ]
    ]
  )


  = Experimental evaluation 

  #align(
    center, 
    block(
        fill: none,
        stroke: 2pt + darkblue,   
        inset: 12pt, 
        [
          #text(fill: darkblue)[
            Human-aided AGFN largely outperforms baselines.
          ]
        ]
      )
  ) 

  #figure(
    image("figures/hitl_benchmarks.svg", width: 80%),
  )

  // #figure(
  //   image("figures/criteria_v2.svg",  width: 100%), 
  //   caption:[#text(fill:teal)[$cal(L)_(C B)$] often outperforms #text(fill:blue)[$cal(L)_(T B)$], #text(fill:orange)[$cal(L)_(D B)$], and #text(fill: brickred)[$cal(L)_(D B text(mod))$] in terms of convergence speed.] 
  // )


