#import "typst-poster/poster.typ": *

#let forestgreen = rgb("#0e380e")
#let darkblue = rgb("00008B")
#let brickred = rgb("AC1616")
#let mmapurple = rgb("#6a1b9a")
#let pf = text(fill: brickred)[$p_F$]
#let pb = text(fill: forestgreen)[$p_B$]
#let gam = text(fill: mmapurple)[$gamma$]

// Reusable tinted, rounded callout box.
#let cbox(body) = block(
  fill: rgb(0, 0, 139, 15),
  stroke: 2pt + darkblue,
  inset: 18pt,
  radius: 14pt,
  width: 100%,
  text(fill: darkblue)[#body],
)

#show: poster.with(
  size: "24x36",
  title: "Random Features for Discrete Amortized Samplers",
  authors: text[Tiago da Silva (+), Amauri H. Souza ($star$), Salem Lahlou (+)],
  departments: none,
  univ_logo: ("../logos/mbzuai.svg", "../logos/ifce.png", "../logos/avra.png"),
  footer_text: "LatinX in AI Workshop @ ICML 2026",
  footer_email_ids: "tiago.dasilva@mbzuai.ac.ae",
  footer_color: "ebcfb2",
  footer_url: text[(+) MBZUAI, ($star$) Federal Institute of Ceará, ($star$) Avra],
  univ_logo_column_size: (3.4in, 1.9in, 2.3in),
  univ_logo_column_gutter: (-1.4in, -.05in, .1in, -.05in),
  title_column_size: "13",
  title_font_size: "58",
  authors_font_size: "30",
  num_columns: "2",
  footer_text_font_size: 24,
  footer_url_font_size: 24,
  keywords: ("Discrete sampling", "GFlowNets"),
)

#set text(size: 27pt)

// ======================= COLUMN 1 =======================

#block(
  fill: rgb(0, 0, 155, 128),
  inset: 22pt,
  radius: 24pt,
  text(fill: white)[
    *TL;DR*
    - discrete amortized samplers (e.g., GFlowNets) suffer from *underdetermination*, *limited expressiveness*, and *inefficient exploration*,
    - *Mixture Model Augmentation (MMA)* injects *continuous random features* #gam into the policy, giving a *mixture of Markov processes* that *provably boosts expressivity*,
    - across three benchmarks, MMA *accelerates convergence* and *improves the fit*.
  ],
)

= Background

We sample $x tilde pi(x) = Z^(-1) R(x)$ over a discrete, *compositional* set $cal(X)$ with $Z = sum_x R(x)$ intractable; each $x$ is the sink of a *state-graph* DAG built from a seed $s_o$. A *forward policy* #pf matches the reward marginal,
$
  p_top (x) = sum_(tau : s_o arrow.r.squiggly x) #text(fill: brickred)[$p_F (s_o, tau)$] prop R(x),
$
tied to $R$ through a *backward policy* #pb by the *trajectory-balance (TB)* loss,
$
  cal(L)_(T B) = EE_(tau tilde p_E) ( log frac(Z thin #text(fill: brickred)[$p_F (s_o, tau)$], R(x) thin #text(fill: forestgreen)[$p_B (x, tau)$]) )^2.
$

= Three Challenges

#cbox[
  *C1. Underdetermination.* Infinitely many $(#text(fill: brickred)[$p_F$], #text(fill: forestgreen)[$p_B$])$ satisfy TB.

  *C2. Limited expressiveness.* A neural policy may represent *no* solution; e.g., 1-WL GNNs alias non-isomorphic states.

  *C3. Inefficient exploration.* A single policy induces *state aliasing*: near-identical embeddings for states that need different actions.
]

= Mixture Model Augmentation

Attach a *continuous random feature* #gam --- *constant along each trajectory* --- to every state, mirroring the momentum of Hamiltonian Monte Carlo.

#cbox[
  *Augmented state graph.* Lift $cal(G)$ to $overline(cal(G)) = (cal(S) times Gamma, cal(X) times Gamma)$ with $(s, gamma) arrow.r (s', gamma')$ iff $s arrow.r s'$ and $gamma = gamma'$.
]

Marginalizing #gam out gives a *convex mixture of Markov processes* --- distinct #gam index distinct policies, resolving *C1*:

#cbox[
  *Mixture semantics.* For $gamma_o tilde p(gamma_o)$,
  $
    p_top (s_t) = integral_Gamma p(gamma_t) thin overline(p)_top (s_t | gamma_t) thin d gamma_t.
  $
]

With $R(x, gam) = R(x) thin p(gam)$ (same $Z$), we target $overline(p)_top (x, gamma) prop R(x, gamma)$ via the *augmented TB loss* on the encoder $eta(s, gam) = phi(s) plus.o psi(gam)$, so MMA plugs into *any* existing sampler unchanged.

#colbreak()

// ======================= COLUMN 2 =======================

= Expressivity: RNI GNNs

Adding #gam is *random node initialization (RNI)*: an RNI GNN is a GNN *ensemble* and a universal approximator.

#figure(
  image("figures/regular_graphs_sg.svg", width: 86%),
  caption: [The 6-cycle $s_o$ has non-isomorphic children $x_1$ (chord $v_1 text("--") v_4$) and $x_2$ (chord $v_1 text("--") v_3$).],
)

A 1-WL #pf gives identical node embeddings, so it is *forced uniform* over $x_1, x_2$ and cannot fit $R_1 != R_2$. Random features $overline(bold(h)) = bold(h) plus.o gam$ break the symmetry,
$
  overline(p)_F ((s_o, gamma), (x_j, gamma)) prop exp{ phi(overline(bold(h))_(v_1) + overline(bold(h))_(v_j prime)) },
$
fitting *any* target and overcoming *C2*.

= Experiments

MMA *speeds up convergence* and *improves the fit* across Hypergrid, Lazy Random Walk, and Set Generation ($|cal(X)| tilde 10^9$).

#figure(
  grid(
    columns: (1fr, 1fr),
    column-gutter: 10pt,
    image("figures/hypergrids_corners.svg", width: 100%), image("figures/hypergrids_gaussian.svg", width: 100%),
  ),
  caption: [*Hypergrid* --- #smallcaps[Corners] (left) and #smallcaps[Gaussian] (right): MMA provides a better goodness-of-fit to the target.],
)

#figure(
  image("figures/discrete_diff.svg", width: 58%),
  caption: [*Lazy Random Walk*: MMA recovers the striped target the un-augmented sampler washes out.],
)

#figure(
  stack(
    spacing: 8pt,
    image("figures/full_sets_training_curves_tv_item.svg", width: 88%),
    image("figures/full_sets_training_curves_tv_size.svg", width: 88%),
  ),
  caption: [*Set Generation*: TV distance on item (top) and set-size (bottom) marginals vs. training step, $K in {12, 18, 24}$. MMA converges faster on both.],
)

#v(4pt)
*Conclusions.* MMA recasts a discrete sampler as a *mixture of Markovian samplers* via one auxiliary variable #gam --- *provably boosting expressivity* (*C2*) and *improving convergence* (*C1*, *C3*) at *no* extra trajectory length.
