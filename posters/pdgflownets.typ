#import "typst-poster/poster.typ": *

#let forestgreen = rgb("#228b22")
#let darkblue = rgb("00008B")
#let brickred = rgb("AC1616")
#let argmin = [argmin]
#let pf = text(fill: brickred)[$p_F (tau)$]
#let pb = text(fill: forestgreen)[$p_B (tau|x)$]
#let mkv = text(fill: brickred)[*Markovian*]
#let pd = text(fill: forestgreen)[*path-dependent*]

#let argmin = math.op("argmin", limits: true)

#show: poster.with(
  size: "48x36",
  title: "Path-dependent Discrete Amortized Inference",
  authors: text[Tiago da Silva (+), Esmeralda S. Whitammer ($star$), Salem Lahlou (+)],
  departments: none,
  univ_logo: ("../logos/mbzuai.svg", "../logos/uoe.png", "../logos/cifar.png"),
  footer_text: "International Conference on Machine Learning 2026",
  footer_url: "https://github.com/ML-FGV/nais",
  footer_email_ids: text[{tiago.dasilva, salem.lahlou}\@mbzuai.ac.ae, esmeralda.whitammer\@ed.ac.uk — (+) MBZUAI · ($star$) University of Edinburgh · ($star$) CIFAR],
  footer_color: "ebcfb2",
  univ_logo_column_size: (6in, 7in, 3.5in),
  univ_logo_column_gutter: (.8in, .4in, .4in),
  title_column_size: "26",
  title_font_size: "88",
  authors_font_size: "46",
  // Modifying the defaults
  keywords: ("GFlowNets", "State aliasing", "Recurrent policies"),
)

#set text(size: 24pt)

#block(
  fill: rgb(0, 0, 155, 128),
  inset: 32pt,
  radius: 24pt,
  [
    #text(fill: white)[
      *TL;DR*
      - *Markovian* amortized samplers (e.g., GFlowNets) suffer from *state aliasing*: distinct states look (nearly) the same to the policy but require *sharply different* action distributions,
      - we prove that Markovian forward policies are *provably less expressive* than their *path-dependent* counterparts,
      - we *lift* the state graph by attaching a *learned latent variable* $W_t$ to each state, keeping the joint process Markovian so that *all standard training objectives remain valid*,
      - $W_t$ evolves according to a learned dynamical system, which is parameterized by the newly introduced *Rotationary Self-Referential Weight Matrix* (R-SRWM),
      - across grids, set generation, and sequence design, path-dependence yields *faster, stabler convergence* and an *improved fit* to the target distribution.
    ]
  ],
)

#block(
  inset: 24pt,
  stroke: none,
  [
    = Background: Amortized Sampling with GFlowNets

    Many core problems reduce to *drawing samples* from a distribution known only up to a constant,
    $ pi(x) = 1/Z thin R(x), quad Z = sum_x R(x) "  intractable", $
    where $cal(X)$ is a *compositional* space: each $x$ is built one component at a time from a seed $s_o$ along a *trajectory* $tau : s_o -> s_1 -> dots.c -> s_T = x$.

    #figure(
      image("figures/liftedgraphs_p7-1.png", width: 64%),
      caption: [A *forward policy* #text(fill: brickred)[$p_F (s_t | s_(t-1))$] navigates the state graph, adding one component per step.],
    )

    A *forward policy* $#text(fill: brickred)[$p_F (s_t | s_(t-1))$]$ rolls out from $s_o$ to build $x$ with marginal $p_F^top (x) = sum_(tau : s_o arrow.squiggly x) product_t p_F (s_t | s_(t-1))$. Our goal is to learn $p_F$ so that $p_F^top (x) prop R(x)$.

    #v(12pt)

    Because $p_F^top$ is intractable, we introduce a *backward policy* $#text(fill: forestgreen)[$p_B (s_(t-1) | s_t)$]$ over the *parents* of a state and tie $p_F$ to the reward through the *trajectory balance (TB)* condition,
    $
      Z product_t #text(fill: brickred)[$p_F (s_t | s_(t-1))$] = R(x) product_t #text(fill: forestgreen)[$p_B (s_(t-1) | s_t)$].
    $
    Enforcing TB $=>$ $p_F^top (x) prop R(x)$. We learn $p_F, p_B, Z_theta$ by minimizing the expected *log-squared violation*,
    $
      cal(L)(tau ; p_F, p_B) = ( log [ Z_theta product_t #text(fill: brickred)[$p_F (s_t | s_(t-1))$] ] - log [ R(x) product_t #text(fill: forestgreen)[$p_B (s_(t-1) | s_t)$] ] )^2,
    $
    under an off-policy $p_E$ (commonly $epsilon$-greedy, $(1 - epsilon) p_F + epsilon p_U$), i.e.,
    $
      p_F^(star), p_B^(star) = argmin_(p_F, p_B) EE_(tau ~ p_E) [cal(L)(tau ; p_F, p_B)].
    $
    // .

    = The Problem: State Aliasing

    #block(
      fill: none,
      stroke: 2pt + darkblue,
      inset: 12pt,
      [
        #text(fill: darkblue)[
          *(Near) state aliasing*. Different states look (nearly) the *same* to the policy, but require *different* action distributions. Equivalently, from the agent's view the environment is *(almost) partially observable*---a plain network struggles to separate neighboring states when the optimal policy changes *sharply* between them (i.e., large Lipschitz constant).
        ]
      ],
    )

    A *Markovian* policy only sees the current state $s_t$. On the #smallcaps[Lines] environment (jump forward up to $M$ steps or stop), we use the linear policy
    $
      p_F (p_(i+k) | p_i) = "softmax"_k ( underbrace(W_i psi(p_i), "logits") dot.o underbrace(m(p_i), "mask") ),
    $
    with position embedding $psi(p_i)$ and mask $m(p_i) in {1, -infinity}^M$. It is *Markovian* when $W_i = W$ (same weights every step), and *path-dependent* when the weights *evolve along the trajectory*,
    $
      W_t = b thin a_t^top, quad a_t = a_(t-1) + (w_a^top psi(s_t)) thin v_a,
    $
    with $b$ fixed and $a_o, w_a, v_a$ learned.

    #figure(
      image("figures/liftedgraphs_p6-1.png", width: 60%),
      caption: [The #smallcaps[Lines] environment: from position $p_i$ the policy either *jumps forward* (up to $M$ steps) or *stops* at the terminal $q_i$.],
    )

    #block(
      fill: none,
      stroke: 2pt + darkblue,
      inset: 12pt,
      [
        #text(fill: darkblue)[
          *Proposition 3.1 (expressivity gap).* Let $cal(R)_"M"$ and $cal(R)_"PD"$ be the sets of distributions realizable by Markovian and path-dependent parameterizations. Then $#text[span ] cal(R)_"M"$ is a *strict subspace* of $#text[span ] cal(R)_"PD"$.
        ]
      ],
    )

    #figure(
      image("figures/liness-1.png", width: 60%),
      caption: [Target $R(q_i) = sum_j w_j e^(-|i - k_j|)$: #mkv collapses to one mode; #pd fits the target.],
    )
  ],
)

= State Aliasing in Practice

For a *sparse* target on #smallcaps[Lines] ($R(q_2) = R(q_20) = 1000 dot R(q_1)$), the optimal actions at $p_2$ and $p_3$ must differ *sharply*, even though the states are nearly indistinguishable.

#figure(
  grid(
    columns: (1fr, 1fr),
    column-gutter: 12pt,
    align: bottom,
    image("figures/linesaliasinggggmaxkl32-1.png", width: 96%),
    image("figures/linesaliasingggg-1.png", width: 100%),
  ),
  caption: [Separation of the learned policies at $p_2$ vs. $p_3$ (left) and convergence / goodness-of-fit (right): #mkv needs $tilde.op 100 times$ more steps to tell $p_2$ from $p_3$---slow and unstable.],
)

// == Graphs and the 1-WL barrier

On graph-building tasks a state is a graph $G$ and a move adds one edge, scored by a message-passing network $p_F (G^(+(m,n)) | G) prop e^(zeta(f(x_m; G) plus.o f(x_n; G)))$. Message passing aggregates a *multiset* of neighbors, so its power is capped by the *1-WL* color-refinement test: some *non-isomorphic* graphs receive *identical* embeddings, forcing the policy to be *uniform* over states it should distinguish.

#figure(
  image("figures/liftedgraphs_p5-1.png", width: 78%),
  caption: [Building a graph edge-by-edge. At $p$, the pairs $(a,b)$ and $(a,c)$ look *identical* to a message-passing policy, forcing it *uniform* over $q_1, q_2$; knowing *when* each edge appeared breaks the tie.],
)

#figure(
  image("figures/sg_tv_vs_alpha-1.png", width: 38%),
  caption: [Target $R(q_1) = alpha$, $R(q_2) = 1 - alpha$: #mkv is stuck at uniform (error $|alpha - 1/2|$); #pd $approx 0$.],
)

#block(
  fill: none,
  stroke: 2pt + darkblue,
  inset: 12pt,
  [
    #text(fill: darkblue)[
      *Proposition 3.2.* Path-dependence *strictly boosts expressivity* in graph-structured tasks---knowing *when* each edge appeared breaks ties that a Markovian graph network cannot.
    ]
  ],
)

= Path-dependent Amortized Sampling

We *lift* the state graph by attaching a latent *memory* $W_t in Omega$ to every state, updated recurrently as the object is built,
$
  W_t = W_(t-1) + phi(s_t, W_(t-1)),
$
and feed it to the action head: $p_F (dot | s_t) arrow.long p_F (dot | s_t, W_t)$. This mirrors *Hamiltonian Monte Carlo*---augmenting the state with a momentum-like variable that carries memory---but for *discrete* spaces.

#figure(
  image("figures/liftedgraphs_p8-1.png", width: 56%),
  caption: [The *lifted* state graph: a memory row $W_t$ is carried alongside each state. The *state-only* process may be non-Markovian, yet the *joint* $(s_t, W_t)$ stays Markovian.],
)

We instantiate $phi$ per task so that the memory *disambiguates* aliased states:
- #smallcaps[Lines:] $Omega = RR^d$, #h(0.3em) $phi(s, a) = (w_a^top psi(s)) v_a$---the linear path-dependent policy.
- #smallcaps[Graphs:] carry the information of *when* each node's it last changed---exactly the signal a plain graph network misses.

A single architecture works across tasks: a *self-referential weight matrix (SRWM)*,
$
  q_t, k_t, v_t, beta_t = W psi(s_t) + b, quad overline(v)_t = W_(t-1) zeta(k_t),
$
$
  W_t = W_(t-1) R + sigma(beta_t) (v_t - overline(v)_t) plus.o zeta(q_t),
$
with slow weights $W, b$ (SGD) and fast memory $W_(t-1)$ at self-set rate $sigma(beta_t)$. The rotation $R$ diversifies $W_t$'s eigenvectors, *reducing aliasing and improving exploration*; it beats similarly-sized LSTMs/GRUs.

#block(
  fill: none,
  stroke: 2pt + darkblue,
  inset: 12pt,
  [
    #text(fill: darkblue)[
      *Training objectives still apply.* Because the latent transitions are *deterministic* and the lifted graph remains *finite*, the *same* losses that guarantee $p_F^top (x) prop R(x)$ on the plain graph *still* guarantee it on the lifted graph (proved for all popular objectives, including TB).
    ]
  ],
)

= Empirical Results

We evaluate on grid exploration, the *lazy random walk*, set generation, and sequence design; across all tasks #pd yields a *closer fit* than its Markovian counterpart.

#figure(
  image("figures/hypergridss-1.png", width: 48%),
  caption: [*Grid World.* #pd recovers the multi-modal target; #mkv collapses onto a few modes.],
)

#figure(
  image("figures/rings-1.png", width: 50%),
  caption: [*Lazy random walk.* Learned densities: #mkv recovers only the inner ring, while #pd matches both rings of the target.],
)

On the *lazy random walk* (appendix), the *recurrent memory* closes the gap: the self-referential weight matrix (*R-SRWM*) *beats similarly-sized LSTMs and GRUs*, and both its *rotation matrix* and *recurrent link* are essential.

#grid(
  columns: (1fr, 1fr),
  column-gutter: 14pt,
  figure(
    image("figures/lazy_random_walk_gated_rnns.svg", width: 78%),
    caption: [*R-SRWM* beats PD-GRU / PD-LSTM and #mkv.],
  ),
  figure(
    image("figures/lazy_random_walk_ablations.svg", width: 78%),
    caption: [*Ablations:* rotation and recurrence both matter.],
  ),
)

= Take-aways

- *Markovian samplers alias states*---provably and empirically *less expressive*.
- Root cause: the environment becomes *(nearly) partially observable*.
- We fix this by attaching a *learned recurrent memory* to the state, keeping the joint process Markovian so *standard training carries over unchanged*.
- Result: *faster, stabler convergence* and a *better fit* to the target.
