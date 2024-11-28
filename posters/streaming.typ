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
    univ_logo_column_gutter: (-2in, -.1in), 
    title_column_size: "32", 
    title_font_size: "88", 
    authors_font_size: "46", 
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

#let body = lorem(90)
#place(
  center + bottom,
  float: true,
  scope: "parent",   
  grid(
    columns: (1fr, 3fr, 1fr),
    body, 
    fig, 
    body  
  )
)

#colbreak() 


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
          width: 100%),
    caption: [
      SB-GFlowNets accurately sample from the posterior distribution over the utility in integer-valued preference learning.  
    ] 
    // [A GFlowNet learns a #text(fill: brickred)[forward policy] on a state graph.]
  )

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

#v(35.75pt, weak: true)
#line(length: 100%)

#align(
  center,   
  block(
        fill: none,
        stroke: 2pt + brickred,   
        inset: 12pt, 
        [
            #text(fill: brickred)[
              Streaming Bayesian structure learning with DAG-GFlowNets. 
          ]
        ]
      )
) 


#figure(

    image("figures/streaming_eval_dags.svg", 
          width: 100%),
    caption: [
      SB-GFlowNets accurately sample from an evolving belief distribution in a structure learning setting. 
    ] 
    // [A GFlowNet learns a #text(fill: brickred)[forward policy] on a state graph.]
  )
