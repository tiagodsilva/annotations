  #import "typst-poster/poster.typ": *

  #let forestgreen = rgb("#228b22")
  #let darkblue = rgb("00008B") 
  #let brickred = rgb("AC1616") 
  #let argmin = [argmin] 
  #let pf = text(fill: brickred)[$p_F (tau)$]
  #let pb = text(fill: forestgreen)[$p_B (tau|x)$]

  #show: poster.with(
    size: "48x36",
    title: "When do GFlowNets Learn the Right Distribution?",
    authors: "Tiago da Silva, Rodrigo Alves, Eliezer Souza, Amauri Souza, Samuel Kaski, Vikas Garg, Diego Mesquita",
    departments: none,
    univ_logo: ("../logos/fgv.png", "../logos/aalto.svg", "../logos/yaiyai.png", "../logos/ifce.png", "../logos/manchester.png"),
    footer_text: "International Conference on Learning Representations 2025",
    footer_url: "https://github.com/ML-FGV/analyzing-gflownets",
    footer_email_ids: "{tiago.henrique, rodrigo.alves, eliezer.souza, diego.mesquita}@fgv.br, {amauri.souza, vikas.garg, sami.kaski}@aalto.fi",
    footer_color: "ebcfb2", 
    univ_logo_column_size: (4.5in, 2.2in, 2.7in, 3.1in, 4.5in),
    univ_logo_column_gutter: (.9in, -.1in, -.5in, -.5in, -.1in, -.1in), 
    title_column_size: "27", 
    title_font_size: "88", 
    authors_font_size: "46", 
    // Modifying the defaults
    keywords: ("GFlowNets", "GNNs"),
  )

  #set text(size: 22pt) 

  #block(
    fill: rgb(0, 0, 155, 128),
    inset: 32pt,
    radius: 24pt,
    [
      #text(fill: white)[
      *TL;DR*    
      - we demonstrate that the impact of an imbalanced transition on the GFlowNet's correctness is unevenly distributed across the network, 
      - we delineate the expressiveness of a GFlowNet parameterized by a 1-WL GNN in terms of the distributions it can learn, 
      - we introduce FCS as a tractable and accurate goodness-of-fit measure for GFlowNets 
      - and we show that conventionally used diagnostic procedures often misrepresent the distributional correctness of a GFlowNet.
      ]
    ]
  )

#block(
  inset: 24pt, 
  stroke: none, 
)[
  = Background: What is a GFlowNet?

  *GFlowNets* are amortized algorithms that sample from distributions over discrete and compositional objects (such as graphs) by learning a Markov Decision Process (MDP) over an extension of the target space. 

  #figure(
    image("figures/tb.svg", 
          width: 45%),
    caption: [A GFlowNet learns a #text(fill: brickred)[forward policy] on a state graph.]
  )


  To achieve this, a *flow network* is defined over an extension $cal(S)$ of $cal(G)$, which represents the sink nodes. To navigate through this network and sample from $cal(G)$ proportionally to a *reward function* $R colon cal(G) arrow.r RR_(+)$, a forward (resp. backward) policy #pf (#pb) is used. 

  $ 
  #pf = product_((s, s') in tau) p_F(s' | s) #text[ and ] sum_(tau arrow.r.squiggly g) #pf = R(g). 
  $

  To achieve this, we parameterize $#pf$ as a neural network trained by minimizing 

  $ cal(L)_(T B)(p_F) = EE [ ( log frac(#text(fill: brickred)[$p_F (tau)$] Z, #text(fill:forestgreen)[$p_B (tau | x)$] R(x)))^(2) ]. $ 

  for a given #pb. GFlowNets can be trained in an *off-policy* fashion and the above expectation can be under any full-support distribution over trajectories. When $cal(L)_(T B)(p_F) = 0$, 

  $
    p_top (x) = sum_(tau #text[ finishes at] x) p_F(tau) prop R(x).
  $

  #figure(
    image("figures/multipage/compositional-06.png", 
          width: 45%),
    caption: [A GFlowNet may also learn a #text(fill: blue)[forward policy] and a #text(fill: brickred)[flow function] on a state graph.]
  )

  When trajectories are long and memory is constrained, we instead parameterize a flow function $log F colon cal(G) arrow.r RR$ and minimize the *detailed balance* (DB) loss

  $
    cal(L)_(D B)(s, s') = ( log ( F(s) p_F (s'|s) ) / (F(s') p_B (s | s'))).
  $  
  
  Conventionally, $F$ and $p_F$ are learned by optimizing an uniformly weighted averaged of $cal(L)_(D B)(s, s')$ over a trajectory, i.e., $frac(1, |tau|) sum_((s, s') in tau) cal(L)_(D B)(s, s')$, in which $|tau|$ is the trajectory's length.  
]

  = Question: What are the limits of GFlowNets?    

  Our work addresses three major challenges in the GFlowNet literature: 

  1. How do an insufficiently minimized $cal(L)_(D B)$ affect the distributional accuracy of a GFlowNet?   

  2. What functions $R$ cannot be sampled from when $p_F$ is parameterized by a GNN with 1-WL expressivity?

  3. How can we tractably measure the proximity of a GFlowNet to its target distribution?


  Jointly, we seek a deeper understanding on the consequences (1), causes (2), and assessment (3) of GFlowNets.
  
  = What are the *consequences* of an imbalanced transition?  

  We estimate the extent to which a lack of balance within an edge of the flow network affect the marginal distribution over the terminal states of the corresponding MDP.

  #figure(
    image("figures/imbalance.png", width: 45%), 
    caption: [An imbalanced flow network.] 
  ) <imbalanced> 

  #block(
    fill: none,
    stroke: 2pt + darkblue,   
    inset: 12pt, 
    [
      #text(fill: darkblue)[
      *Error propagation in GFlowNets*. @imbalanced presents a scenario in which an extra volume of $delta$ flows through one of the flow network's edges. Under these conditions, the perturbed marginal distribution $tilde(p)_top$ over $cal(G)$ satisfies  
      $
            epsilon(delta, g, F(s_0)) <= "TV"(tilde(p)_T, pi) <= epsilon(delta, g^h, F(s_0)), #text[ with ]
            epsilon(delta, x, t) = (1 - 1/x) delta/(t + delta),
      $
      $pi(x) prop R(x)$, and $"TV"(tilde(p)_top, pi) = 1/2 sum_(x in cal(G)) |pi(x) - tilde(p)_top (x)|$ is the total variation distance. 
      ]
    ]
  )

  In doing so, we observe that the impact of each transition over the downstream distribution is heterogeneously distributed throughout the network. 
  
  Drawing on this result, we repurpose the DB loss to allow for non-uniform weighting of the transitions within a trajectory.
  More precisely, we let $gamma colon cal(S) arrow.r RR$ be state-weighting and define 

  $
    cal(L)_(W D B) (tau) = frac(1, |tau|) sum_(s, s') gamma(s) cal(L)_(D B) (s, s') 
  $

  as the weighted detailed balance. During training, we let $gamma(s) prop ("depth of" s)$ linearly anneal $gamma(s)$ to $1$. Empirical results show that our $cal(L)_(W D B)$ often leads to faster or comparable convergence than the traditional DB loss.   

  #figure(
    image("figures/stability_analysis_extended_thesis.svg",
    width: 100%),
    caption: [$cal(L)_(W D B)$ often leads to faster learning convergence than $cal(L)_(D B)$ in benchmark tasks.]
  )


  #block(
    fill: none,
    stroke: 2pt + brickred,   
    inset: 12pt, 
    [
      #text(fill: brickred)[
        *Open question*. While our findings demonstrate the suboptimality of an unweighted detailed balance loss, the search for an optimal weighting scheme for $cal(L)_(D B)$ remains an open challenge in the GFlowNet literature.
      ]  
    ]
  )

  = What are the *causes* for an imbalanced transition?  

  The flow network will only be balanced if the corresponding GFlowNet perfectly minimizes its learning objective. When a GNN is used to parameterize the $p_F$, as it commonly is, this minimization may not be achievable. 
  

  #block(
    fill: none,
    stroke: 2pt + darkblue,   
    inset: 12pt, 
    [
      #text(fill: darkblue)[
      *Limitations of GFlowNets parameterized by 1-WL GNNs*. We consider distributions over graph-structured objects. Assume that a GFlowNet is parameterized as 

      $
        p_F (s, s') prop exp{ "MLP"(phi_((u|s)) || phi_((v|s)))) }, 
      $ <gnn>

      in which $phi_(u|s)$ (resp. $phi_((v|s))$) is a representation of the node $u$ (resp. $v$) computed by a GNN with 1-WL expressivity; $u$ and $v$ are the endpoints of the added edge in the transition $s arrow.r s'$. 
      
      Then, there are specific combinations of state graphs and reward functions for which a GFlowNet cannot solve the underlying flow assignment problem. @regular and @samplings shows an example.
      ]
    ]
  )
  
  #figure(
    image("figures/multipage/samplings-15.png", width: 50%), 
    caption: [A GNN GFlowNet cannot distinguish the left and right children of $s$ - in spite of their different rewards.]
  ) <samplings> 

  
  #grid(
    columns: (75%, 50%),
    gutter: -144pt,
    [
      To circumvent this issue, we reparameterize $p_F$ to incorporate the #text(fill: brickred)[children state] into the current transition $s arrow.r s'$.   
      $
        p_F (s, s') prop exp{ "MLP"(psi_1 ({phi_((u|s)), phi_((v|s))}) || #text(fill: brickred)[$psi_2 ({phi_((w|s'))}_(w in V(s')))$] },
      $
    ],
    [#figure(
      image("figures/test_reg_graphs.py_plot.svg", width: 135pt)
    )]
  )
  
  in which $psi_1$ and $psi_2$ are permutation-invariant functions and $V(s')$ is the set of nodes of $s'$. The resulting method, termed Look Ahead (LA) GFlowNets, is strictly more expressive than a standard GFlowNet.

  #block(
    fill: none,
    stroke: 2pt + brickred,   
    inset: 12pt, 
    [
      #text(fill: brickred)[
        *Open question*. Although we have established the first results regarding the limited expressivity of a GFlowNet, a complete characterization of the model's expressivity is open.
      ]  
    ]
  )

  #figure(
    image("figures/regular_graphs.svg", width: 45%), 
    caption: [A GFlowNet parameterized as in @gnn cannot distinguish the actions $P arrow.r R$ and $P arrow.r L$.]
  ) <regular> 

  = How to measure imbalance?  

  Due to the combinatorial nature of the state space, we cannot directly compare the distribution learned by a GFlowNet against the target distribution 
  cannot be tractably computed, i.e., 

  $
    "TV"(p_top, pi) = frac(1, 2) sum_(x in cal(G)) |p_top (x) - pi(x)| "is intractable". 
  $  

  Inspired by the Reinforcement Learning literature, prior works have computed quantities such as the top-quantile of the reward ($"TopQuantile"$) and the number of modes ($"NumModes"$) to decide upon which GFlowNet converges faster, which are respectively defined as 
  // .  

  $
    "TopQuantile"(R, alpha, cal(X)_n) = "Avg"({ R(x) colon x in cal(X)_n and R(x) >= "Quantile"(R(cal(X)_n), alpha)})
  $

  $
    "NumModes"(R, cal(X)_n, xi) = | { x in cal(X)_n colon R(x) >= xi}| 
  $

  in which $cal(X)_n = {x_1, ..., x_n} subset cal(G)$ is a set of samples that a GFlowNet found during training and $xi > 0$ is an arbitrary threshold. We show that neither of these quantities accurately reflect the distributional accuracy of a GFlowNet.  

  #block(
    fill: none,
    stroke: 2pt + darkblue,   
    inset: 12pt, 
    [
      #text(fill: darkblue)[
      *Flow Consistency in Subgraphs*. Instead, we propose measuring the Flow Consistency in Subgraphs (FCS) for assessing GFlowNets. In a nutshell, FCS is the expected TV between $p_top$ and $pi$ in random subsets of $cal(G)$, 

      $
        "FCS"(pi, p_top) = EE_(cal(B))  [ "TV"(pi^(cal(B)), p_top^(cal(B))) ].
      $ 

      Here, $p_top^(cal(B))$ and $pi^(cal(B))$ are the restrictions of the corresponding distributions to $cal(B)$, and the expectation is under a full-support distribution over subsets of $cal(G)$. 
      ]
    ]
  )

  Our experiments show that FCS is both sound and easy to evaluate (@metrics), while prior approaches fail at correctly measuring the distributional accuracy of a GFlowNet (@modes).  

  #figure(
    image("figures/fcs_vs_l1.py_plot.png", width: 50%),
    caption: [$"FCS"$ is strongly correlated and drastically faster to evaluate than the TV distance.]
  ) <metrics>

  #figure(
    image("figures/test_alternative_metrics_with_fcs.svg", width: 50%),
    caption: [$"TopQuantile"$ (left) assings a high-score to an incorrectly learned GFlowNet (according to $"FCS"$; right).]
  ) <modes> 


  *Case study.* 
  To illustrate the effectiveness of FCS, we evaluate its performance on variants of LED- and FL-GFlowNets to demonstrate that, despite their impressive mode-finding capability, they do not correctly sample from the target distribution (@flgflownets). 

  #figure(
    image("figures/test_alternative_metrics.py_log_tgt_q90_plot.svg", width: 100%),
    caption: [Despite finding modes very quickly (3-6), LED- and FL- GFlowNets fail at sampling from $pi$ (1-2).]
  ) <flgflownets> 
  
    #block(
    fill: none,
    stroke: 2pt + brickred,   
    inset: 12pt, 
    [
      #text(fill: brickred)[
        *Open question*. An understanding of the impressive mode-finding rate of LED- and FL- GFlowNets is lacking. 
      ]  
    ]
  )

  = Take-home message

  Our work provides a principled approach for analyzing GFlowNets through the perspective of flow imbalance (I), characterizing its potential failure modes under the light of the limited expressivity of GNNs (II), and diagnosing its learned distribution with a sound and computationally tractable metric that may help standardize evaluation and accelerate progress in the field (III).    
