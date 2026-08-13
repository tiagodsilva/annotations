#import "typst-poster/poster.typ": *

#let forestgreen = rgb("#228b22")
#let darkblue = rgb("00008B")
#let brickred = rgb("AC1616")
#let argmin = [argmin]
#let pf = text(fill: forestgreen)[$p_F (s_o, tau)$]
#let pb = text(fill: brickred)[$p_B (x, tau)$]
#let rr = text(fill: brickred)[$R(x)$]

#show: poster.with(
  size: "48x36",
  title: text[
    Particle GFlowNets:
    #v(-2cm)
    Rethinking Generative Marginalization Models],
  authors: text[Tiago da Silva $(*)$, Diego Mesquita $(+)$, Salem Lahlou $(*)$],
  departments: none,
  univ_logo: (
    "../logos/mbzuai.svg",
    "../logos/fgv.png",
  ),
  footer_text: "Conference on Uncertainty in Artificial Intelligence 2026",
  footer_url: "https://github.com/ML-FGV/pgflownets",
  footer_color: "ebcfb2",
  footer_email_ids: text[$(*)$ MBZUAI, $(+)$ FGV],
  univ_logo_column_size: (7in, 7in),
  univ_logo_column_gutter: (-2in, -.1in),
  title_column_size: "32",
  title_font_size: "88",
  authors_font_size: "49",
  header_font_size: 36pt,
  // Modifying the defaults
  keywords: ("GFlowNets", "Markov chain Monte Carlo"),
)

#set text(size: 36pt)
#show math.equation: set text(42pt)

#block(
  fill: rgb(0, 0, 155, 128),
  inset: 32pt,
  radius: 24pt,
  [
    #text(fill: white)[
      *TL;DR*
      - Generative Marginalization Models (MaMs) are fast autoregressive samplers for discrete distributions over combinatorial spaces;
      - we show MaMs can be seen as an instance of a GFlowNet with a newly defined learning objective;
      - we extend this objective to non-autoregressive discrete modelling, showcasing for the first time how to effectively train GFlowNets *without sampling entire trajectories* for each gradient step;
      - we demonstrate our algorithm---called *Particle GFlowNets*---significantly accelerates learning convergence in terms of wall-clock time.
    ]
  ],
)

#block(
  inset: 24pt,
  stroke: none,
)[
  = Background: Sampling from Discrete Distributions

  We have a finite set $cal(X)$ of _states_ and a positive function $R colon cal(X) arrow.r RR_(+)$.
  Our objective is to sample from
  $
    x ~ pi(x) prop R(x)
  $

  #block(
    fill: none,
    stroke: 2pt + darkblue,
    inset: 12pt,
    [
      #text(fill: darkblue)[
        *Assumption*. $cal(X)$ is a *combinatorial space*.
        This means each $x in cal(X)$ can be described as a *set of components* from a fixed set $cal(C)$.
      ]
    ],
  )

  Examples include the space of *graphs*, which can be described as a set of edges, and the space of *sets*.

  #grid(
    columns: (.9fr, 1fr),
    column-gutter: 1.5cm,
    text[
      #v(48pt)
      #figure(
        image("figures/multipage/stategraphs-page-3.svg", width: 89%),
        caption: [A GFlowNet *constructs* $x in cal(X)$ in proportion to the *target function* $rr$.],
      )],
    text[
      #v(48pt)

      A GFlowNet learns a *policy function* $p_F$ that builds each $x in cal(X)$ such that
      $
        sum_(tau colon s_o arrow.squiggly.r x) pf = rr / Z #h(64pt) // .
      $ <eq:marginalsa>
      while also estimating
      $
        Z = sum_(x in cal(X)) rr.
      $
      We paramterize #text(fill: forestgreen)[$p_F(s, dot.c)$] with a neural network.
    ],
  )

]

= MaMs as GFlowNets

A Generative Marginalization Model (MaM) assumes the generative process is *autoregressive*.

#v(24pt)

#grid(
  columns: (1fr, 1.2fr),
  column-gutter: 1cm,
  figure(
    image("figures/multipage/stategraphs-page-4.svg", width: 99%),
    caption: [Autoregressive.],
  ),
  figure(
    image("figures/multipage/stategraphs-page-5.svg", width: 89%),
    caption: [Non-autoregressive.],
  ),
)

#v(24pt)

We think of *$d$-sized sequences* with token set $cal(C)$, a combinatorial space on $cal(C) times [d]$.
A MaM's goal is to ensure

$
  F(s) #text(fill: forestgreen)[$p_F$] (s, s') = F(s') #text[ and ] F(x) = rr
$ <eq:consistencysa>

for $x in cal(X)$ and *adjacent* states $s, s'$, i.e., for which $|s' \\ s| = 1$.
Example: $s = {(c, 1)}$ and $s' = {(c, 1), (c', 2)}$.

#v(24pt)

#figure(
  grid(
    columns: (1fr, 1fr),
    column-gutter: 1cm,
    image("figures/multipage/stategraphs-page-1.svg", width: 95%),
    image("figures/multipage/stategraphs-page-2.svg", width: 95%),
  ),
  caption: [A Permutation ($sigma$) Conditioned generative process (PCGP): elements added according to $sigma$.],
) <fig:permutationssa>

#v(24pt)

MaMs maintain $B$ *Markov chains* $(s_(b, t))_(t >= 1)$ for $b in {1, dots, B}$ via *Markov kernels* compatible with any PCGP of the kind described in @fig:permutationssa and minimize

$
  cal(L)(p_F, F, t) = sum_(1 <= b <= B) ( log frac(F (s_(b, t)) #text(fill: forestgreen)[$p_F$] (s_(b, t), s_(b, t + 1)), F(s_(b, t + 1))) )^2.
$

// .

#v(24pt)

#block(
  fill: none,
  stroke: 2pt + forestgreen,
  inset: 12pt,
  [
    #text(fill: forestgreen)[
      *Proposition*.
      (MaMs are *PC GFlowNets*).
      For each MaM abiding by @eq:consistencysa and each PCGP in @fig:permutationssa, there is a unique GFlowNet satisfying @eq:marginalsa.
      We call *PC GFlowNets* the set of all such GFlowNets.
    ]
  ],
)


= Particle GFlowNets: \ Learning GFlowNets from Partial Trajectories

We extend MaMs to *non-autoregressive generative processes* by simply introducing a *backward policy*, i.e., // .

#math.equation(block: true, numbering: none)[
  $cal(L)_P (p_F, p_B, F, t) = sum_(1 <= b <= B) ( log frac(F (s_(b, t)) #text(fill: forestgreen)[$p_F$] (s_(b, t), s_(b, t + 1)), F(s_(b, t + 1)) #text(fill: brickred)[$p_B$] (s_(b, t + 1), s_(b, t))) )^2.$
]

The Markov chains $(s_(b, t))_(t >= 1)$ are constructed by *backward-forward kernels*, as described in the next figure.

#v(24pt)

#grid(
  columns: (1fr, 1fr),
  column-gutter: 1.5cm,
  figure(
    image("figures/multipage/stategraphs-page-6.svg", width: 99%),
    caption: [Markov chain for *Particle GFlowNets*.],
  ),
  text[
    The main process lives on $cal(X)$: $(x_(b, t))_(t >= 1)$.
    We sample $k ~ #text[Poisson] (log L)$, and
    #math.equation(block: true, numbering: none)[
      $s_(b, t) ~ #text(fill: brickred)[$p_B^((k))$] (x_(b, t), dot.c) \
      s_(b, t + 1) ~ #text(fill: forestgreen)[$p_F$] (s_(b, t), dot.c).$
    ]
    ($L$ is the maximum trajectory size). We then sample
    #math.equation(block: true, numbering: none)[
      $x_(b, t + 1) ~ #text(fill: forestgreen)[$p_F^((k - 1))$] (s_(b, t + 1), dot.c).$
    ]
  ],
)

#v(24pt)

We evaluate $cal(L)_P$ for both transitions, $x_(b, t) arrow.r x_(b, t + 1)$ and $s_(b, t) arrow.r s_(b, t + 1)$.
Our model enables both fast, correlated and expensive, independent sampling from $rr$.

#v(24pt)


#block(
  fill: none,
  stroke: 2pt + forestgreen,
  inset: 12pt,
  [
    #text(fill: forestgreen)[
      *Proposition*.
      (Sampling correctness).
      When $cal(L)_P$ is *minimized*, the GFlowNet satisfies @eq:marginalsa and the *Markov chain* $(x_(b, t))_(t >= 1)$ is *ergodic* relatively to $R(x)$.
      #v(12pt)
    ]
  ],
)

// #lorem(150)

= Experiments

#figure(
  grid(
    columns: 1fr,
    rows: (auto, -3cm, auto),
    row-gutter: 1.5cm,
    image("figures/sets_fact.svg", width: 100%),
    [],
    align(right, image("figures/legend_1x2_pgflownets.svg", width: 40%))
  ),
  caption: [Learning curve set generation ($d$ = set size).],
)

// #lorem(150)
