  #import "typst-poster/poster.typ": *

  #let forestgreen = rgb("#0e380e")
  #let darkblue = rgb("00008B") 
  #let brickred = rgb("AC1616") 
  #let argmin = [argmin] 
  #let pf = text(fill: brickred)[$p_F (tau)$]
  #let pb = text(fill: forestgreen)[$p_B (tau|x)$]
  #let FCS = [FCS] 

  #show: poster.with(
    size: "24x36",
    title: "Analyzing GFlowNets: Stability, Expressiveness, and Assessment",
    authors: "Tiago da Silva, Eliezer da Silva, Rodrigo Alves, Luiz Max Carvalho, Amauri Souza, Samuel Kaski, Vikas Garg, Diego Mesquita", 
    departments: none,
    univ_logo: ("../logos/fgv.png", "../logos/aalto.svg", "../logos/ifce.png", "../logos/manchester.png"),
    footer_text: "SPIGM Workshop @ ICML 2024",
    footer_email_ids: "tiago.henrique@fgv.br",
    footer_color: "ebcfb2", 
    footer_url: "https://openreview.net/forum?id=B8KXmXFiFj", 
    univ_logo_column_size: (2in, 1.2in, 1.5in, 1.5in), 
    univ_logo_column_gutter: (-2in, -.05in, -.25in, -.05in, -.05in), 
    title_column_size: "16", 
    title_font_size: "64", 
    authors_font_size: "32", 
    num_columns: "2", 
    footer_text_font_size: 24, 
    footer_url_font_size: 24, 
    // Modifying the defaults
    keywords: ("GFlowNets", "GNNs", "generative models"),
  )

  #set text(size: 19.9pt) 

  #block(
    fill: rgb(0, 0, 155, 128),
    inset: 28pt,
    radius: 24pt,
    [
      #text(fill: white)[
      *TL;DR* 
      - we analyze GFlowNets from three fundamental perspectives: *stability*, *expressiveness*, and *assessment*,    
      - for stability, we quantify the impact of node-level balance violations over the learned distribution to derive a novel transition-decomposable learning objective, 
      - for expressiveness, we establish the distributional limits of GFlowNets parameterized by 1-WL GNNs and propose *LA-GFlowNets*, a provably more expressive extension of GFNs,  
      - for assessment, we develop a theoretically sound metric for measuring the accuracy of GFlowNets based on distributional errors on subsets of the state space.  
      ]
    ]
  )

= Background: GFlowNets 

GFlowNets are *amortized samplers* for distributions on a set $cal(X)$ of *compositional objects* (e.g., graphs) defined by three ingredients. 

1. a state graph $cal(G) = ({s_o} union cal(S) union cal(X), cal(E))$, which is a DAG on $cal(S) supset.eq cal(X)$,  
2. forward and backward policies, $p_F$ and $p_B$, on $cal(G)$, 
3. a flow function, $F$, representing the flow within each state. 

A GFlowNet is trained to ensure the marginal over $cal(X)$ induced by $p_F(dot | s_o)$ matches a *reward function* $R colon cal(X) arrow.r RR_(+)$. Often, this is done by 

#math.equation(block: true)[
  $
    p_F = argmin_(p_F) cal(L)_(D B)(p_F) = EE_(tau ~ p_F (dot | s_o)) [ 1/\#tau sum_((s, s') in tau) (log frac(F(s) p_F (s' | s), F(s') p_B (s | s')))^2]. 
  $
] <eq:obj> 

= Bounds on the TV of GFlowNets 

#block(
  fill: none,
  stroke: 2pt + darkblue,   
  inset: 24pt, 
  [ 
    #text(fill: darkblue)[ 
      [*Stability to local balance violations.*] Let $(cal(G), p_F, p_B, F)$ be a GFlowNet wrt $R$. Define $(cal(G), tilde(p)_F, p_B, tilde(F))$ by increasing the flow $F(s)$ in some $s$ by $delta$ and redirecting the extra flow to a child $s^star$ of $s$. Also, let $cal(D)_(s^star) subset cal(X)$ be the terminal descendants of $s^star$. Then,
      #math.equation(numbering: none, block: true)[
        $
        frac(delta, F(s_0) + delta) (1 - sum_(x in cal(D)_(s^star)) pi(x)) <=
        |tilde(p)_T - pi|_(T V) <= frac(delta, F(s_0) + delta) ( 1 - min_(x in cal(D)_(s^star)) pi(x) ).
        $ 

      ] 
    ]
  ] 
) 

The figure below illustrates the above result for a tree-structured state graph when the extra flow is added to the initial state. 

#figure(
  image("figures/treebalance.svg", width: 65%), 
  // caption: [An imbalanced state graph $cal(G)$.] 
) 

Our analysis suggest that errors at earlier nodes, associated to smaller $min_(x in cal(D)_(s^star)) pi(x)$, dominate the loss function of GFlowNets in (1).  

#block(
  fill: none,
  stroke: 2pt + forestgreen,   
  inset: 24pt, 
  [ 
    #text(fill: forestgreen)[ 
      [*Empirical illustration.*] @fig:eval confirms that, for the initial training epochs, the DB loss is mostly dominated by violations to the balance of shallow --- and rewardless --- states. 
    ]
  ] 
) 

#figure(
  image("figures/eval_traj_db.svg", width: 75%), 
  caption: [DB loss is dominated by imbalances at early states.]   
) <fig:eval> 

Experiments showed that, by weighting each transition in the DB loss proportionally to the inverse of the incoming node's number of terminal descendants, convergence is often significantly sped up. 

= Distributional limits of 1-WL GFlowNets 

Forward policies are frequently parameterized by 1-WL GNNs. We show that this parameterization *limits the expressivity* of GFlowNets. 

#block(
  stroke: 2pt + darkblue, 
  inset: 24pt, 
  [
    #text(fill: darkblue)[ 
        Let $cal(G)$ be a tree-structured state graph with reward $R$. Let $T(s) subset.eq cal(X)$ for $s in cal(S)$ be the terminal descendants of $s$. If there is a $s = (V, E) in cal(S)$ with $(a, b) != (b,c) in V^2 without E$ indistinguishable by 1-WL s.t.    
        $
        sum_(x in T(s^prime)) R(x) != sum_(x \in T(s^(prime prime))) R(x)
        $ 
        with $s^(prime) = (V, E union {(a,b)})$ and $s^{prime prime} = (V, E union {(c,d)})$, then there is no 1-WL GFlowNet capable of approximating $pi prop R$ with TV zero.
    ] 
  ] 
) 

On the positive side, we also prove that a GFlowNet equipped with a 1-WL GNN for $p_F$ can learn arbitrary distributions over trees. 

#block(
  inset: 24pt, 
  stroke: 2pt + darkblue, 
  [
    #text(fill: darkblue)[
      If $cal(S)$ is a set of trees such that $(s, s') in cal(E)$ implies that $s subset s'$ with $\# E(s^prime)=\# E(s)+1$, then there is a GFlowNet equipped with 1-WL GNNs that can approximate any distribution $pi$ over $cal(X) subset.eq cal(S)$.
    ]
  ]
)

To boost expressivity, we incorporate the embeddings of $s'$ when evaluating $p_F (s' | s)$. The resulting algorithm is called _LA-GFlowNet_. 

#block(
  inset: 24pt, 
  stroke: 2pt + forestgreen, 
  [
    #text(fill: forestgreen)[
      [*Empirical illustration*]. To exemplify the results above, we consider the task of learning a distribution over regular graphs by employing the conventional edge-adding generative process in a state graph satisfying (2). Results are laid out in @fig:regulargraphs. 
    ] 
  ] 
)

#figure(
  image("figures/test_reg_graphs.svg", width: 45%), 
  caption: [Cases in which, in contrast to #text(fill: teal)[LA-GFNs], GNN-based #text(fill: red)[GFNs] are unable to sample from the target due to limited expressivity.]   
) <fig:regulargraphs> 

= Diagnosing GFlowNets 

We design a sound and tractable metric for assessing the goodness-of-fit of a GFlowNet, called *flow-consistency in subnetworks*. 

#block(
  inset: 24pt, 
  stroke: 2pt + darkblue, 
  [
    #text(fill: darkblue)[
      [*FCS*]. For $S subset.eq cal(X)$, let $p_T^((S))$ and $pi^((S))$ be the restrictions of the learned and target measures to $S$. Also, let $P_S$ be a distribution over subsets of $cal(X)$ and $theta$ be the model's parameters. Then,  
      $
      FCS(p_T, pi) = EE_(S ~ P_S) [ e(S, theta) ] colon.eq EE_(S ~ P_S) [ frac(1, 2) sum_(x in S) |p_(T)^((S))(x ; theta) - pi^((S))(x)| ]. 
      $ 
    ]
  ]
)

For FCS, $p_T (x ; theta)$ can be easily computed by $sum_(tau arrow.r.squiggly x) p_F (tau | s_o)$ for a small subset $S$ of $x$. We also show that FCS and TV are roughly equivalent. 

#block(
  inset: 24pt, 
  stroke: 2pt + darkblue,
  [
    #text(fill: darkblue)[
      Let $P_S$ be a distribution over $(S subset.eq cal(X) colon \#S = beta)$ for a $beta >= 2$. Also, let $d_(T V) = e(cal(X), theta)$ be the TV between $p_(T)$ and $pi$ for a GFlowNet parameterized by $theta$. Then, $d_(T V) = 0$ if and only if $EE_(S ~ P_S)[e(S, theta)] = 0$. 
    ]   
  ] 
)

We consider the prototypical task of set generation to illustrate the correspondence between FCS and TV in the examples below. 

#grid(
  columns: (7.5in, 3.5in),
  [ 
  #block(
    inset: 24pt, 
    stroke: 2pt + forestgreen,
    [
      #text(fill: forestgreen)[
        [*Empirical illustration*]. Figure at side shows that TV and FCS are highly correlated for the task of set generation. Results correspond to 20 different runs with varying rewards and set sizes.  
      ]  
    ] 
  )
  ], 
  [
    #image("figures/fcs_vs_l1.svg", width: 65%)
  ] 
) 