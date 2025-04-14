#let template(
  title: none,
  abstract: [], 
  doc,  
) = {

  set page(
    paper: "a4",
    header: align(right)[
      #title 
    ], 
    margin: (x: 1.8cm, y: 1.5cm),
    numbering: "(1/1)"
  )
  set math.equation(numbering: "(1)") 

  set text(
    font: "Comfortaa", 
    size: 11pt 
  )
  set par(justify: true) 

  set heading(numbering: "1.") 
  show heading.where(level: 1): it => [
    #set text(12pt, weight: "bold")
    #if it.numbering != none [
      #counter(heading).display(
        it.numbering 
      )  
    ]
    #smallcaps(it.body) 
  ]

  
  set align(center) 
  text(17pt, title) 

  par(justify: false)[
    #set align(left) 
    *Abstract* \ 
    #abstract 
  ]

  set align(left)
  columns(2, doc) 
}


Estatística e teoria da decisão 

Livros: 
  Casella e Berger (Statistical Inference) 
  McCullagh (GLMs) 
  Hardin (GLMs)
  Kutner (Linear Models)
  Keener (Theoretical Statistics; mostly for Hypothesis Testing)
  Robert (The Bayesian Choice)