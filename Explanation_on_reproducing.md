# Why *Deep Reinforcement Learning for Trading* Is Not Fully Reproducible

## Summary

This note explains why I do **not** believe the paper *Deep Reinforcement Learning for Trading* can be fully and faithfully reproduced from the published text alone.

The issue is not simply that the implementation is difficult. Rather, the paper contains a combination of:

1. **hard internal inconsistencies**
2. **underspecified metric definitions**
3. **evaluation choices that are not sufficiently pinned down for replication**

In my view, this means that partial reproduction is possible, but **full table-level reproduction across all asset classes is not realistically achievable from the paper alone**. 

---

## 1. The paper is internally inconsistent on key evaluation metrics

### 1.1 Calmar ratio is the clearest contradiction

The paper defines Calmar as follows:

> “the Calmar ratio compares the expected annual rate of return with maximum drawdown” 

The natural reading of this statement is:

**Calmar = E(R) / MDD**

However, the values reported in Tables 2 and 3 do **not** follow that formula.

For example, in **Table 2**, for **Equity Index / Long**:

* `E(R) = 0.668`
* `MDD = 0.132`
* reported `Calmar = 0.509` 

But if Calmar were truly computed as `E(R) / MDD`, then:

`0.668 / 0.132 ≈ 5.06`

which is nowhere near `0.509`.

Likewise, in **Table 3**, for **Equity Indexes / Long**:

* `E(R) = 0.504`
* `MDD = 0.127`
* reported `Calmar = 0.466` 

But:

`0.504 / 0.127 ≈ 3.97`

Again, this is completely inconsistent with the reported value.

### Conclusion

This is not a minor ambiguity. It is a **direct contradiction between the paper’s written definition and the paper’s reported results**.

Therefore, **Calmar cannot be treated as a faithfully reproducible metric from the paper text**.  

---

## 2. The paper is under-specified on downside deviation and Sortino

The paper defines downside deviation as:

> “annualised standard deviation of trade returns that are negative” 

and then defines Sortino as:

**Sortino = E(R) / Downside Deviation**

The problem is that the phrase “standard deviation of trade returns that are negative” is **not statistically precise enough** for exact reproduction.

There are at least two plausible interpretations:

### Interpretation A: standard deviation of the negative-return subsample

`DD = sqrt(252) * std(R_t | R_t < 0)`

### Interpretation B: zero-target downside deviation / lower partial moment

`DD = sqrt(252) * sqrt((1/T) * Σ min(R_t, 0)^2)`

These are **not the same quantity**.

What makes this especially problematic is that the paper’s **Sortino values do appear numerically consistent with the reported DD column**, which means Sortino is internally consistent **conditional on DD**. But DD itself is not defined rigorously enough to guarantee that a third party will compute the same thing. 

### Conclusion

* **Sortino is not as broken as Calmar**, because its ratio structure is internally coherent.
* But **Sortino inherits the ambiguity of DD**.
* Therefore, Sortino is only **partially reproducible**, not fully pinned down.

---

## 3. The paper uses additive profits, but later evaluates path-dependent ratio metrics without fully specifying the wealth construction

In the reward-function section, the paper explicitly states:

> “We define `r_t = p_t - p_{t-1}` and this expression represents additive profits.” 

It further contrasts this with multiplicative returns:

> “If we want to trade a fraction of our accumulated wealth at each time, multiplicative profits should be used and `r_t = p_t / p_{t-1} - 1`.” 

This is an important choice. It means the paper’s primitive reward object is **additive PnL**, not a percentage return.

However, later the paper evaluates metrics such as:

* maximum drawdown
* Calmar ratio
* cumulative trade returns
* portfolio-level volatility scaling 

These quantities depend on a **wealth path** or at least on a path-normalization convention. But the paper never fully specifies:

1. how initial wealth is set
2. how additive trade returns are converted into a drawdown path
3. whether drawdown is computed on additive wealth, normalized NAV, or some scaled portfolio process
4. whether MDD is computed on the raw portfolio return stream or on the post-scaled stream from Table 2

### Why this matters

If returns are additive, then different choices such as:

* `W_t = W_0 + Σ R_s`
* `W_t = N * W_0 + Σ R_s`
* a normalized NAV
* or some post-vol-targeted equity curve

can all produce different MDD and therefore different Calmar values.

### Conclusion

The paper’s use of **additive profits** is explicit, but the corresponding **wealth-path construction for evaluation is not**.

That leaves too much freedom in reproducing MDD-related metrics.  

---

## 4. The paper mixes multiple layers of volatility scaling without fully operationalizing them

In the reward definition, the paper says that positions are volatility scaled by `σ_tgt / σ_t`, and that this helps normalize rewards across contracts with different price scales. 

Later, in the experimental section, the paper says:

> “We present our results in Table 2 where an additional layer of portfolio-level volatility scaling is applied for each model.”
> “We also include the results without this volatility scaling for reference in Table 3.” 

So there are at least **two distinct layers of scaling**:

1. **contract-level scaling inside the reward function**
2. **portfolio-level scaling for Table 2**

But the paper does not fully specify, at a practical implementation level:

* how the portfolio-level scaling is estimated
* whether it uses rolling realized volatility, EWM volatility, or another estimator
* what target volatility is used
* whether the same procedure is applied identically to all baselines and all RL models
* exactly how this scaling interacts with the already volatility-scaled reward stream

### Why this matters

This affects:

* `E(R)`
* `std(R)`
* `MDD`
* `Calmar`
* cross-asset comparability

It also means that a replication can look “close” for one asset class and fail badly for another, even if the core signal logic is similar.

### Conclusion

The paper clearly states that **an extra portfolio-level scaling layer exists**, but it does not specify it in enough detail to make third-party reproduction unique. 

---

## 5. The portfolio construction is stated, but not enough detail is given to uniquely reconstruct the reported tables

The paper states that portfolio returns are formed as an equal-weight average of per-contract trade returns:

`R_t^port = (1/N) * Σ R_t^i`



That part is clear.

However, several practical details remain unresolved for faithful reproduction:

* exact treatment of missing data or differing contract histories
* exact handling of rolling/continuous futures construction
* precise ratio-adjustment details
* exact data version from the Pinnacle CLC database
* whether any cleaning or exclusions were applied before final reporting
* whether portfolio scaling is done before or after averaging in every case

This matters because the paper spans:

* commodities
* equity indices
* fixed income
* FX 

and empirical sensitivity to data construction can be very different across these asset classes.

### Conclusion

Even if one accepts the high-level portfolio formula, the paper still leaves enough data-engineering freedom that **matching one asset class does not imply the others can be matched**.

---

## 6. Why one asset class can look “roughly right” while the others fail

This is exactly what I observed in reproduction attempts.

It is possible to obtain a **reasonably close match for Equity Index**, especially for simpler baselines such as Long Only, while still failing badly on:

* commodities
* fixed income
* FX

This does **not** necessarily mean the reproduction is wrong. It may simply reflect that:

1. Equity index behavior in the sample period is easier to approximate with coarse implementation choices
2. scaling and data-version sensitivities are smaller there
3. the paper’s under-specification is less damaging in that subset

Once the same code is applied across the other asset classes, the unresolved choices in:

* return definition
* scaling
* cost treatment
* evaluation path construction

start to dominate.

### Conclusion

A replication that “kind of works” in Equity Index but breaks elsewhere is **fully consistent with the paper being under-specified**, not necessarily with the replication being careless.

---

## 7. What is actually reproducible, and what is not

### Relatively reproducible

These parts are comparatively well specified:

* the broad RL framing
* the action spaces
* the use of LSTMs
* equal-weight portfolio construction
* Sharpe as `E(R) / std(R)`
* the existence of contract-level and portfolio-level volatility scaling  

### Only partially reproducible

These parts can be approximated, but not uniquely reproduced:

* downside deviation
* Sortino
* maximum drawdown
* the exact realized `E(R)` across all asset classes
* the Table 2 vs. Table 3 scaling pipeline

### Not faithfully reproducible from the paper text alone

The strongest case is:

* **Calmar ratio as reported in the tables**, because the numerical values contradict the written definition.  

---

## 8. Final assessment

My conclusion is **not** that the paper is worthless. The paper may still contain a valid high-level research idea. But as a reproduction target, it has serious problems.

### Strong claim

The paper contains at least one **hard internal inconsistency**:

* the reported **Calmar values do not match the paper’s own written formula**.  

### Moderate claim

The paper also contains several **major under-specifications**:

* downside deviation
* Sortino input definition
* wealth-path construction for MDD
* interaction between additive profits and drawdown-based metrics
* the exact implementation of portfolio-level volatility targeting.  

### Practical conclusion

Therefore, I believe the correct academic position is:

> This paper can be **partially reproduced in spirit**, but it **cannot be fully reproduced at the table level across all asset classes from the published text alone**.

That is not merely a limitation of my implementation. It is a direct consequence of the paper’s own inconsistencies and under-specified evaluation methodology.

---

## 9. One-sentence version for discussion with my advisor

> The main obstacle is not just implementation difficulty; the paper itself combines inconsistent metric definitions, under-specified evaluation choices, and insufficient operational detail, so full reproduction of the reported tables is not possible from the publication alone.
