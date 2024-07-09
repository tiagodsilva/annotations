#import "typst-poster/poster.typ": *

#let forestgreen = rgb("#228b22")
#let darkblue = rgb("00008B") 
#let argmin = [argmin] 
#let pf = text(fill: red)[$p_F (tau)$]
#let pb = text(fill: forestgreen)[$p_B (tau|x)$]

#show: poster.with(
  size: "48x36",
  title: "Embarrassingly Parallel GFlowNets",
  authors: "T. Silva, L. Carvalho, A. Souza, S. Kaski, D. Mesquita",
  departments: none,
  univ_logo: ("../logos/fgv.png", "../logos/aalto.svg", "../logos/manchester.png"),
  footer_text: "International Conference of Machine Learning 2024",
  footer_url: "https://github.com/ML-FGV/ep-gflownets",
  footer_email_ids: "{tiago.henrique, luiz.carvalho, diego.mesquita}@fgv.br, {amauri.souza, sami.kaski}@aalto.fi",
  footer_color: "ebcfb2", 
  univ_logo_column_size: (5.8in, 3in, 7in), 
  title_column_size: "28", 
  title_font_size: "88", 
  authors_font_size: "46", 
  // Modifying the defaults
  keywords: ("GFlowNets", "Distributed Bayesian inference"),
)

#set text(size: 22pt) 

#block(
  fill: darkblue,
  inset: 12pt,
  radius: 4pt,
  [
    #text(fill: white)[
    *TL;DR*     
    - we introduce the *contrastive balance condition (CBC)* as a provably sufficient criterion for sampling correctness in GFlowNets,    
    - we develop the *first general-purpose algorithm*, called *Embarrassingly Parallel GFlowNets* (EP-GFlowNets), enabling minimum-communication parallel inference for probabilistic models supported on discrete and  compositional spaces,  
    - we show that EP-GFlowNets can accurately and efficiently learn to sample from a target in a distributed setting in many benchmark tasks, including phylogenetic inference and Bayesian structure learning,  //.
    - we verify that minimizing the *CB loss*, derived from the CBC, often leads  to faster convergence than alternative learning objectives. 
    ]
  ]
)

= GFlowNets 

*GFlowNets* are amortized algorithms for sampling from distributions over discrete and compositional objects (such as graphs). 

#figure(
  image("figures/tb.svg", 
        width: 45%),
  caption: [A GFlowNet learns a #text(fill: red)[forward policy] on a state graph.]
)


Briefly, a *flow network* is defined over an extension $cal(S)$ of $cal(G)$, which then represents the sink nodes. To navigate within this network and sample from $cal(G)$ proportionally to a *reward function* $R colon cal(G) arrow.r RR_(+)$, a forward (resp. backward) policy #pf (#pb) is used. 

$ 
#pf = product_((s, s') in tau) p_F(s' | s) #text[ and ] sum_(tau arrow.r.squiggly g) #pf = R(g). 
$

To achieve this, we parameterize $#pf$ as a neural network trained by minimizing 

$ cal(L)_(T B)(p_F) = EE [ ( log frac(#text(fill: red)[$p_F (tau)$] Z, #text(fill:forestgreen)[$p_B (tau | x)$] R(x)))^(2) ]. $ 

for a given #pb. GFlowNets can be trained in an *off-policy* fashion and the above expectation can be under any full-support distribution over trajectories. 

= Embarrassingly Parallel Inference   

Reward functions can often be multiplicatively decomposed in simpler primitives, 

$
#text(fill: forestgreen)[$R$] (g) = product_(1 <= i <= N) #text(fill:red)[$R_i$] (g).  
$

Each $R_i$ may be a *subposterior* conditioned on a subsample of the data (@fig:bayesian). Often, the $R_i$'s cannot be disclosed due to privacy or computational constraints. 

#figure(
  image("figures/variational_dist.svg", 
        width: 50%),
  caption: [Approximated and embarrassingly parallel Bayesian inference.]
) <fig:bayesian> 

Commonly, an approximation $q_i$ to each $R_i$ is locally learned and publicly shared to a centralizing server. An approximation to $R$, then, is obtained by approximating 

$
#text(fill:forestgreen)[$q$] (g) approx product_(1 <= i <= n) #text(fill:red)[$q_i$] (g).  
$ <eq:dist> 

= Contrastive Balance Condition 

Our objective is to solve the approximation problem in @eq:dist when each $q_i$ is a trained GFlowNet. To achieve this, we develop the *CB condition*.  

// Our objective is to enable distributed approximate inference when each client learns a GFlowNet to sample proportionally to $R_i$. First,    

#block(
  fill: none,
  stroke: 2pt + darkblue,   
  inset: 12pt, 
  [
    #text(fill: darkblue)[
    *Contrastive balance condition*. Let $p_F$ and $p_B$ be the policies of a GFlowNet. Then, 
    $
      frac(p_F (tau), R(x) p_B (tau|x)) = frac(p_F   (tau'), R(x') p_B (tau'|x')) 
    $
    for all trajectories $tau, tau'$ finishing at $x,  x'$ is a sufficient condition for ensuring that a GFlowNet samples sink nodes from $cal(G)$ proportionally to $R$. 
    ]
  ]
)

Differently from alternative balance conditions, the CB *does not rely* on auxiliary quantities such as $Z$. Clearly, enforcing CB is a sound learning objective for training GFlowNets. 

#block(
  fill: none,
  stroke: 2pt + darkblue,   
  inset: 12pt, 
  [
    #text(fill: darkblue)[
    *Contrastive balance loss*. Let $p_F$ and $p_B$ be the policies of a GFlowNet. Define   
    $
    cal(L)_(C B) (p_F) = EE [ (log frac(p_F (tau), R(x) p_B (tau|x)) - log frac(p_F (tau'), R(x') p_B (tau'|x')))^2 ]. 
    $
    Then, $p_F^star = argmin cal(L)_(C B)(p_F)$ samples from $cal(G)$ proportionally to $R$. 
    ]
  ]
)

Our empirical analysis shows that minimizing $cal(L)_(C B)$, which has minimal parameterization, often leads to faster convergence relatively to previously proposed methods.  

#figure(
  image("figures/criteria_v2.svg",  width: 100%), 
  caption:[#text(fill:teal)[$cal(L)_(C B)$] often outperforms #text(fill:blue)[$cal(L)_(T B)$], #text(fill:orange)[$cal(L)_(D B)$], and #text(fill:red)[$cal(L)_(D B text(mod))$] in terms of convergence speed.] 
)
= EP-GFlowNets and Aggregating Balance Condition 

#figure(
  image("figures/epgflownets.svg", 
        width: 100%),
  caption: [Comparison between learning objectives in terms of convergence speed.]
)


#block(
  fill: none,
  stroke: 2pt + darkblue,   
  inset: 12pt, 
  [
    #text(fill: darkblue)[
    *Aggregating balance condition*. 
    $
      a 
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
    *Aggregating balance loss*.
    $
      a 
    $ 
    ]
  ]
)

= Empirical results on benchmark tasks 

// #set align(center)
// #table(
//   columns:(auto, auto, auto), 
//   inset:(10pt),
//  [#lorem(4)], [#lorem(2)], [#lorem(2)],
//  [#lorem(3)], [#lorem(2)], [$alpha$],
//  [#lorem(2)], [#lorem(1)], [$beta$],
//  [#lorem(1)], [#lorem(1)], [$gamma$],
//  [#lorem(2)], [#lorem(3)], [$theta$],
// )

// #set align(left)
// #lorem(80)
// $ mat(
//   1, 2, ..., 8, 9, 10;
//   2, 2, ..., 8, 9, 10;
//   dots.v, dots.v, dots.down, dots.v, dots.v, dots.v;
//   10, 10, ..., 10, 10, 10;
// ) $

// #block(
//   fill: luma(230),
//   inset: 8pt,
//   radius: 4pt,
//   [
//     #lorem(80),
//     - #lorem(10),
//     - #lorem(10),
//     - #lorem(10),
//   ]
// )
// #lorem(75)
// ```rust
// fn factorial(i: u64) -> u64 {
//     if i == 0 {
//         1
//     } else {
//         i * factorial(i - 1)
//     }
// }
// ```

// $ sum_(k=1)^n k = (n(n+1)) / 2 = (n^2 + n) / 2 $

// #block(
//   fill: luma(230),
//   inset: 8pt,
//   radius: 4pt,
//   [
//     #lorem(30),
//   ]
// )

