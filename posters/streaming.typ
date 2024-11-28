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
    title: "Streaming Bayes GFlowNets",
    authors: "Tiago da Silva, Daniel Augusto de Souza, Diego Mesquita",
    departments: none,
    univ_logo: ("../logos/fgv.png", "../logos/ucl.png"),
    footer_text: "Conference on Advances in Neural Information Processing 2024",
    footer_url: "https://github.com/ML-FGV/streaming-gflownets",
    footer_email_ids: "{tiago.henrique, diego.mesquita}@fgv.br, daniel.souza.21@ucl.ac.uk",
    footer_color: "ebcfb2", 
    univ_logo_column_size: (8in, 8in),
    univ_logo_column_gutter: (-4in, -.1in), 
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
      - we propose _Streaming Bayes GFlowNets_ (SB-GFlowNets) as // the first 
        a general-purpose variational inference tool for streaming Bayesian inference in discrete spaces, 
      - in simple terms, SB-GFlowNet employs a GFlowNet as a surrogate prior to update the current posterior based on newly observed data, eliminating the need to process the entire dataset repeatedly,   
      - we introduce off-policy and on-policy algorithms for training SB-GFlowNets, 
      - we demonstrate that SB-GFlowNets are susceptible to catastrophic error propagation and discuss potential workarounds, 
      - we empirically verify that SB-GFlowNets can drastically reduce the training time of a GFlowNet in a streaming Bayes setting,    
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
          width: 50%),
    caption: [
      An illustration of the GFlowNet's state graph as a DAG on $cal(S)$. 
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

])

#colbreak() 


  #block(
    fill: rgb(0, 0, 155, 128),
    inset: 32pt,
    radius: 24pt,
    [
      #text(fill: white, size: 48pt)[
      *TL;DR*     
      - we propose _Streaming Bayes GFlowNets_ (SB-GFlowNets) as // the first 
        a general-purpose variational inference tool for streaming Bayesian inference in discrete spaces, 
      - in simple terms, SB-GFlowNet employs a GFlowNet as a surrogate prior to update the current posterior based on newly observed data, eliminating the need to process the entire dataset repeatedly,   
      - we introduce off-policy and on-policy algorithms for training SB-GFlowNets, 
      - we demonstrate that SB-GFlowNets are susceptible to catastrophic error propagation and discuss potential workarounds, 
      - we empirically verify that SB-GFlowNets can drastically reduce the training time of a GFlowNet in a streaming Bayes setting,    
      ]
    ]
  )