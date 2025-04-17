  #import "typst-poster/poster.typ": *
  #import "@preview/ouset:0.2.0"

  #let forestgreen = rgb("#228b22")
  #let darkblue = rgb("00008B") 
  #let brickred = rgb("AC1616") 
  #let argmin = [argmin] 
  #let pf = text(fill: brickred)[$p_F (tau)$]
  #let pb = text(fill: forestgreen)[$p_B (tau|x)$]

  #let pfT = text(fill: brickred)[$p_F^((T)) (tau)$]
  #let pbT = text(fill: forestgreen)[$p_B^((T)) (tau|x)$]
  
  #let pfTm1 = text(fill: brickred)[$p_F^((T - 1)) (tau)$]
  #let pbTm1 = text(fill: forestgreen)[$p_B^((T - 1)) (tau|x)$]
  
  #let pbtau = text(fill: forestgreen)[$p_B (tau)$]
  #let KL = text[$mono(K L)$] 
  #let ceq = $ouset.overset(=, "C")$ 
  #let pfprime = text(fill: brickred)[$p_F (tau')$]

  #let sgray(this) = {
    text(fill: gray, size: 24pt)[#this]  
  }
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
    univ_logo_column_gutter: (-2in, -.1in), 
    title_column_size: "32", 
    title_font_size: "88", 
    authors_font_size: "46", 
    poster_margin: (top: 1in, left: .5in, right: .5in, bottom: 2in), 
    // Modifying the defaults
    keywords: ("GFlowNets", "Variational Bayesian inference"),
  )

  #set text(size: 32pt) 

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
      An illustration of the state graph as a DAG on $cal(S)$. 
    ] 
    // [A GFlowNet learns a #text(fill: brickred)[forward policy] on a state graph.]
  )


  #block(
    fill: none,
    stroke: 2pt + darkblue,   
    inset: 12pt, 
    [
      #text(fill: darkblue)[      
        To accomplish this, we learn a forward #pf and backward #pb policies on a state graph (illustrated above) such that 
        $
          Z #pf = #pb R(x), 
        $ 
        in which $R$ is the unnormalized distribution of interest and $Z$ is its partition function. If this condition is satisfied for each $tau$, 
        $
          p_(top)(x) = sum_(tau arrow.r.squiggly x) p_F (tau) prop R(x),   
        $
        ensuring that the correctness of the generative process. 
      ]
    ]
  )

  = Streaming Bayes GFlowNets  

])

#colbreak() 

  #block(
    fill: rgb(0, 0, 155, 128),
    inset: 32pt,
    radius: 24pt,
    [ 
      #align(
        center, 
        text(fill: white,  size: 64pt)[
        We introduce #text(weight: "black")[Streaming Bayes GFlowNets] (SB-GFlowNets) as a general-purpose tool for streaming Bayesian inference over *discrete spaces*. Our model leverages a *GFlowNet* as a surrogate prior when updating the current posterior approximation based on new data, thereby *avoiding to repeatedly process old data* and *significantly accelerating* training convergence in a streaming setting.  
      ]
      ) 
    ]
  )


#let fig = figure(
  placement: bottom, 
  scope: "parent", 
  image("figures/streaming_diagrams.svg", 
        width: 100%),
  caption: [
    Streaming amortized inference with SB-GFlowNets. 
  ] 
  // [A GFlowNet learns a #text(fill: brickred)[forward policy] on a state graph.]
)

#let body = block(
  inset: .9pt 
)[
  Let ${R_t}_(t >= 1)$ be a sequence of unnormalized distributions. For each $T$, we train a GFlowNet $G_T$ sampling in proportion to  
  $
    product_(1 <= t <= T) R_t (x). 
  $ 
  These GFlowNets approximately satisfy 
  $
    Z_T #pfT = #pbT product_(1 <= t <= T) R_t (x), 
  $ 
  By noticing that 
  $
    Z_T #pfT = #pbT R_T (x) 
    underbrace(
      product_(1 <= t <= T - 1) R_t (x), 
      approx #pfTm1 slash #pbTm1  
    ) 
  $ 
  we train the $T$th GFlowNet by minimizing 
  #math.equation(numbering: none, block: true)[
    $
      EE_tau [ (log frac(Z_T #pfT, #pbT R_T (x)) -  log frac(#pfTm1, #pbTm1))^2 ]. // , 
    $ 
  ]
]

#let phylo_table = table(
    columns: (.5fr, .5fr, 2fr, 2fr, .6fr),
    rows: (64pt, 64pt, 64pt, 64pt, 64pt), 
    inset: 10pt, 
    stroke: none, 
    align: top, 
    [], [], table.cell(colspan: 2)[Model], table.vline(), [], 
    table.hline(), 
    [], [], [GFlowNet], [SB-GFlowNet], [\% $arrow.t$], 
    table.hline(),  
    table.cell(
          rowspan: 3, 
          rotate(
            -90deg, reflow: true, origin: center + bottom 
          )[\# of leaves]
        ), table.vline(),  
    [7], table.vline(), [$2846.88$ #sgray[s]], [$bold(1279.68)$ #sgray[s]], [$0\%$] , 
    table.hline(), 
    [9], [$3779.11$ #sgray[s]], [$bold(1714.49)$ #sgray[s]], [$-2\%$], 
    table.hline(), 
    [11], [$4821.74$ #sgray[s]], [$bold(2303.99)$ #sgray[s]], [$0\%$], 
    table.hline(), 
)

#let table_legend = block(
    inset: 32pt,
    radius: 24pt,
)[
  SB-GFlowNets *achieve faster training convergence* than conventional GFlowNets in a streaming context --- while *maintaining a comparable performance* in terms of the TV distance (right column).  
  
  (Results averaged across 3 runs.) 
]

#place(
  center + bottom,
  float: true,
  scope: "parent",   
  grid(
    columns: (1fr, 3fr, 1fr),
    gutter: .3in, 
    align(left, body), 
    fig, 
    grid(rows: 2, gutter: .3in, phylo_table, table_legend)   
  )
)

#colbreak() 


#align(
  left,   
  block(
        fill: none,
        stroke: 2pt + brickred,   
        inset: 12pt, 
        radius: 24pt,
        [
            #text(fill: brickred)[
              - *Under which circumstances are SB-GFlowNets useful?* 
              
              1. when $product_(1 <= t <= T) R_t (x)$ is 
                expensive to compute (e.g., in large-scale Bayesian inference
                --- where each $R_t (x)$ is a likelihood function), // . 
              2. and each GFlowNet is relatively cheap to evaluate 
                (e.g., $#pf$ is an MLP or a small GNN --- which covers most applications).  
              
              - *What factors should we consider when training SB-GFlowNets?* 

              SB-GFlowNets are amenable to catastrophic error propagation; 
              the accumulated errors should be carefully tracked. 
          ]
        ]
      )
) 

#v(35.75pt, weak: true)
#line(length: 100%)

#align(
  center, 
  block(
      fill: none,
      stroke: 2pt + darkblue,   
      inset: 12pt, 
      [
          #text(fill: darkblue)[
            Linear preference learning with integer-valued features. 
        ]
      ]
    )
) 

#figure(
    image("figures/streaming_distributional_approx.svg", 
          width: 80%),
    caption: [
      SB-GFlowNets accurately sample from the posterior distribution over the utility in integer-valued preference learning.  
    ] 
    // [A GFlowNet learns a #text(fill: brickred)[forward policy] on a state graph.]
  )


// #figure(

//     image("figures/streaming_eval_dags.svg", 
//           width: 100%),
//     caption: [
//       SB-GFlowNets accurately sample from an evolving belief distribution in a structure learning setting. 
//     ] 
//     // [A GFlowNet learns a #text(fill: brickred)[forward policy] on a state graph.]
//   )

#v(35.75pt, weak: true)
#line(length: 100%)

#align(
  center, 
  block(
        fill: none,
        stroke: 2pt + forestgreen,   
        inset: 12pt, 
        [
            #text(fill: forestgreen)[
              Online Bayesian phylogenetic inference. 
          ]
        ]
      )
) 

#figure(

    image("figures/phylogenetic_posterior_approx.svg", 
          width: 100%),
    caption: [
      SB-GFlowNet's probability mass associated to the true phylogenetic tree increases as we observe more sequences. 
    ] 
    // [A GFlowNet learns a #text(fill: brickred)[forward policy] on a state graph.]
  )
