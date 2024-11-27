  #import "typst-poster/poster.typ": *

  #let forestgreen = rgb("#228b22")
  #let darkblue = rgb("00008B") 
  #let brickred = rgb("AC1616") 
  #let argmin = [argmin] 
  #let pf = text(fill: brickred)[$p_F (tau)$]
  #let pb = text(fill: forestgreen)[$p_B (tau|x)$]

  #show: poster.with(
    size: "24x36",
    title: "Human-aided Discovery of Ancestral Graphs",
    authors: "Tiago da Silva, Eliezer da Silva, Samuel Kaski," +  linebreak(justify: false) +  "Dominik Heider, Diego Mesquita, Adèle Helena Ribeiro",
    departments: none,
    univ_logo: ("../logos/aalto.png", "../logos/marburg.png", "../logos/fgv.png", "../logos/mila.png"),
    footer_text: "LatinX @ NeurIPS 2024",
    footer_url: "https://github.com/ML-FGV/agfn",
    footer_color: "ebcfb2", 
    univ_logo_column_size: (5in, 5in), 
    univ_logo_column_gutter: (-3.5in, -.25in, -.25in, -.05in, -.05in), 
    univ_logo_grid_row_size: (1.8in, 2in),  
    univ_logo_grid_col_size: (3.6in, 3.6in), 
    univ_logo_scale: (50%, 50%, 110%, 110%), 
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
      - we introduce the *contrastive balance condition (CBC)* as a provably sufficient and minimally parameterized criterion for sampling correctness in GFlowNets,    
      - we develop the *first general-purpose algorithm*, called *Embarrassingly Parallel GFlowNets* (EP-GFlowNets), enabling minimum-communication parallel and federated inference for probabilistic models with compositional and finite supports,  
      - we show that EP-GFlowNets can accurately and efficiently learn to sample from a target in a distributed setting in many benchmark tasks, including phylogenetic inference and Bayesian structure learning,  //.
      - we verify that minimizing the *CB loss*, derived from the CBC, often leads  to faster convergence than alternative learning objectives. 
      ]
    ]
  )

#block(
  inset: 24pt, 
  stroke: none, 
[
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

  = Background: Embarrassingly Parallel Inference   

  Reward functions can often be multiplicatively decomposed in simpler primitives, 

  $
  #text(fill: forestgreen)[$R$] (g) = product_(1 <= i <= N) #text(fill: brickred)[$R_i$] (g).  
  $

  Each $R_i$ may be a *subposterior* conditioned on a subsample of the data (@fig:bayesian). Often, the $R_i$'s cannot be disclosed due to privacy or computational constraints. 

  #figure(
    image("figures/variational_dist.svg", 
          width: 50%),
    caption: [Approximated and embarrassingly parallel Bayesian inference.]
  ) <fig:bayesian> 

  Commonly, an approximation $q_i$ to each $R_i$ is locally learned and publicly shared to a centralizing server. An approximation to $R$, then, is obtained by approximating 

  $
  #text(fill:forestgreen)[$q$] (g) approx product_(1 <= i <= n) #text(fill: brickred)[$q_i$] (g).  
  $ <eq:dist> 
])

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
    caption:[#text(fill:teal)[$cal(L)_(C B)$] often outperforms #text(fill:blue)[$cal(L)_(T B)$], #text(fill:orange)[$cal(L)_(D B)$], and #text(fill: brickred)[$cal(L)_(D B text(mod))$] in terms of convergence speed.] 
  )

  = EP-GFlowNets and Aggregating Balance Condition 

  #figure(
    image("figures/epgflownets-file.svg", 
          width: 85%),
    caption: [An overview of EP-GFlowNets for learning GFlowNets in a distributed setting.]
  ) <fig:epgflownets>


  We develop a *divide-and-conquer* algorithm to train GFlowNets in a parallel.   

  The condition below shows how to aggregate locally trained GFlowNets in a *single communication step* without directly evaluating the individual reward functions in the server. 

  #block(
    fill: none,
    stroke: 2pt + darkblue,   
    inset: 14pt, 
    [
      #text(fill: darkblue)[
      *Aggregating balance condition*. Let $#text(fill: brickred)[$(p_F^((1)), p_B^((1))), dots, (p_F^((N)), p_B^((N)))$]$ be the policies of $N$ independently trained GFlowNets. Assume each #text(fill: brickred)[$(p_F^((i)), p_B^((i)))$] samples proportionally to $R_i$. If  
      $
      frac( 
        ( product_(1 <= i <= N) #text(fill: brickred)[$p_F^((i)) (tau)$] ),
        ( product_(1 <= i <= N) #text(fill: brickred)[$p_B^((i)) (tau | x)$] )
      )  #text(fill: forestgreen)[$p_F (tau') p_B (tau | x)$]  = 
        frac(
          ( product_(1 <= i <= N) #text(fill: brickred)[$p_F^((i)) (tau')$] ), 
          ( product_(1 <= i <= N) #text(fill: brickred)[$p_B^((i)) (tau' | x')$] )
        )  #text(fill: forestgreen)[$p_F (tau) p_B (tau' | x')$], // . 
      $ <eq:agg> 
      then the GFlowNet #text(fill: forestgreen)[$(p_F, p_B)$] samples from $cal(G)$ proportionally to $product_(1 <= i <= N) R_i$.   
      ]
    ]
  )

  Similarly to the CB condition, we enforce the condition above by minimizing the expected log-squared difference between the left- and right-hand sides. 

  #block(
    fill: none,
    stroke: 2pt + darkblue,   
    inset: 14pt, 
    [
      #text(fill: darkblue)[
      *Aggregating balance loss*. Under the conditions of @eq:agg, define 
      $
        cal(L)_(A B)(p_F) = EE [ ( log frac( 
        ( product_(1 <= i <= N) #text(fill: brickred)[$p_F^((i)) (tau)$] ),
        ( product_(1 <= i <= N) #text(fill: brickred)[$p_B^((i)) (tau | x)$] )
      )  #text(fill: forestgreen)[$frac(p_F (tau'), p_B (tau' | x'))$] - log 
        frac(
          ( product_(1 <= i <= N) #text(fill: brickred)[$p_F^((i)) (tau')$] ), 
          ( product_(1 <= i <= N) #text(fill: brickred)[$p_B^((i)) (tau' | x')$] )
        ) #text(fill: forestgreen)[$frac(p_F (tau), p_B (tau | x))$]  )^2 ]. 
      $
      Then, $cal(L)_(A B)$ is globally minimized at a policy #text(fill: forestgreen)[$p_F$] sampling proportionally to $product_(1 <= i <= N) R_i$. 
      ]
    ]
  )

  Realistically, each GFlowNet will *only partially satisfy* their local balance conditions. Yet, we show the aggregated model can be *accurate* even under such *imperfect conditions*.  


  #block(
    fill: none,
    stroke: 2pt + darkblue,   
    inset: 14pt, 
    [
      #text(fill: darkblue)[
      *Influence of local failures*. Under the notations of @eq:agg, assume that 
      $
        1 - alpha_n <= min_(x in cal(G), tau arrow.r.squiggly x) frac(#text(fill: brickred)[$p_F^((n)) (tau)$], #text(fill: brickred)[$p_B^((n))(tau | x) R_(n)(x)$]) <= max_(x in cal{X}, tau arrow.r.squiggly x) frac(#text(fill: brickred)[$p_(F)^((n)) (tau)$], #text(fill: brickred)[$p_(B)^((n)) (tau | x) R_(n)(x)$]) <= 1 + beta_(n)  // . 
      $
      for each $n in [[1, N]]$. Also, assume that the aggregated model satisfies @eq:agg. Then, the Jeffrey divergence between the learned $hat(R)$ and target $R$ distributions is bounded by 
      $
        cal(D)_(J)(R, hat(R)) <= sum_(n=1)^(N) log  (frac(1 + beta_(n), 1 - alpha_(n))). 
      $
      ]
    ]
  )


= Empirical results on benchmark tasks 

We assess the performance of EP-GFlowNets in distributed versions of set and sequence generation, grid exploration, Bayesian phylogenetic inference and structure learning.   

#figure(
  image("figures/grids.svg", width: 100%), 
  caption: [Results for the Grid environment showcasing the correctness of EP-GFlowNets.] 
)


#figure(
  image("figures/phylogenetics_elapsed.svg", width: 100%), 
  caption: [Results for Bayesian phylogenetic inference highlight that EP-GFlowNets can achieve a significant speed-up in learning while incurring a negligible accuracy loss.] 
)


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

