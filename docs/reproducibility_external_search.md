# External Reproducibility Search — Deep Reinforcement Learning for Trading

This note records the external search result for public reproductions of:

- Zhang, Zohren, Roberts (2019/2020)
- *Deep Reinforcement Learning for Trading*
- arXiv:1911.10107

## Question

Can we find a credible public blog / tutorial / codebase that fully reproduces the paper, especially:

- the 50 continuous futures setup
- the Table 2 / Table 3 relationship
- the portfolio-level volatility scaling bridge
- the final `MDD` / `Calmar` reporting path

## What Was Found

### Paper mirrors / indexes

- [ResearchGate paper page](https://www.researchgate.net/publication/337484696_Deep_Reinforcement_Learning_for_Trading)
- [IDEAS / RePEc entry](https://ideas.repec.org/p/arx/papers/1911.10107.html)
- [CatalyzeX entry](https://www.catalyzex.com/paper/deep-reinforcement-learning-for-trading)
- [Oxford seminar page](https://www.maths.ox.ac.uk/node/34415)

These confirm the paper exists and was discussed publicly, but they do **not** provide a clean reproducibility recipe.

### Code aggregation sites

- [Papers With Code entry](https://paperswithcode.com/paper/deep-reinforcement-learning-for-trading)

Important observation:

- Papers With Code currently shows **“No code implementations yet”** for this paper.

This is not proof of irreproducibility by itself, but it is notable negative evidence: there is no widely recognized public implementation registered there.

### Blog-style articles

- [Leo Mercanti Medium article](https://leomercanti.medium.com/deep-reinforcement-learning-for-trading-9a1627cbef23)

This is a general DRL-trading explainer, **not** a faithful reproduction of the Zhang/Zohren/Roberts continuous-futures experiment.

### Related survey

- [Deep Reinforcement Learning for Trading—A Critical Survey](https://www.mdpi.com/2306-5729/6/11/119?type=check_update&version=2)

This survey does not serve as a reproduction of the paper, but it supports the broader point that DRL trading papers often suffer from:

- insufficient environment specification
- dataset ambiguity
- unclear evaluation pipelines
- reproducibility difficulties

## What Was *Not* Found

No credible public source was found that clearly and completely reproduces all of the following:

1. the exact 50-futures universe
2. the exact continuous-futures data construction
3. the exact Table 2 portfolio-level scaling rule
4. the exact reporting-path definition behind `MDD` / `Calmar`

In particular, no blog / repo / tutorial was found that cleanly explains:

- why Table 2 differs from Table 3 the way it does
- how the portfolio-level volatility scaling is implemented
- how `MDD` and `Calmar` are computed consistently after that scaling

## Current External Conclusion

Based on public search results, the most defensible statement is:

- there is **no obvious public exact reproduction** of this paper
- there is **no widely recognized reference implementation**
- and there is **no public write-up that resolves the Table 2 / Table 3 bridge ambiguities**

This does **not** prove the paper is impossible to reproduce.

However, it is fair to say that:

- the paper appears **poorly externally reproducible**
- and our local difficulties are **not contradicted by public evidence**

## Practical Implication For This Repo

Because no external exact reproduction was found, this repo should continue to treat the following as open specification problems rather than “implementation bugs until proven otherwise”:

- continuous futures source construction
- Table 2 portfolio-level volatility scaling
- reporting-path construction for `MDD`
- reporting-path construction for `Calmar`

The strongest current interpretation is:

- the paper likely intended an additional portfolio-level volatility scaling layer for Table 2
- but the public paper + references do not specify enough detail to recover a unique implementation
