# IEOR 4733: Algorithmic Trading – Course Project Guidelines

## Introduction

This course project is divided into two major components:
* Project Proposal & Introduction Deck (Mid-Term Take Home Assessment)
* Final Implementation / Research Presentation (Final Project Presentation)

The goal is to combine:
* Academic rigor (paper-level thinking)
* Quantitative validation
* Production-level system thinking

## Logistical Information

* The project selection must be done prior to the Mid-Term Examination – The guidelines for the same will be shared soon.
* You can form Teams of 2-4. Be wary that the project complexity must be proportional to the number of members in the team.

---

## Proposal Deck

Students must:
* Select a research paper (from provided list OR self-selected) OR propose a new research direction
* Present a structured introduction deck

### Approval Requirement

If students:
* Choose a paper NOT on the provided list
* OR propose a new research idea

They must receive approval from the Professor before proceeding.

### Deck Structure

The deck that will be submitted in the mid-term should be approximately 4 slides and should answer the following questions and points:
* What is the topic?
* Brief Introduction of the topic.
* Overview of Literature and the Methods.
* Strategy Overview – This can consist of Data, Methodology, and Evaluation Metrics
* Identified Weakness in existing Methods
* Tentative Proposed Timeline of the Project or a Planned Extension of the Project – This can consist of what technology is being used or what strategy is being changed?

---

## Project Approaches

There are two approaches for the project – Reproduce / Extend an Existing Paper, or Propose a New Research Area.

### Option A – Reproduce / Extend an Existing Project

Students may:
* Reproduce results
* Improve methodology
* Stress-test assumptions
* Extend to new markets / data

If reproducing a paper - They must deliver a fully working trading system pipeline, not just a notebook.

#### Required Deliverables

**Research Presentation – Must include:**
* Summary of original paper
* Reproduction results
* Differences vs original
* Robustness checks
* Regime sensitivity
* Risk diagnostics

**Deployed Application - Must include:**
* Clean data pipeline
* Backtest engine
* Transaction cost modeling
* Performance dashboard
* Risk metrics
* Ability to run new simulations

**Acceptable formats:**
* Web app (Streamlit / React + backend)
* Interactive dashboard
* API-based system
* Modular Python framework

**The system must demonstrate:**
* No lookahead bias
* Realistic execution assumptions
* Reproducibility
* Clear separation of training vs testing
* Sensitivity analysis

---

### Option B – Propose a New Research Area

Students proposing something new must include:
* Clear alpha hypothesis
* Economic intuition
* Testable research question
* Data availability
* Feasibility assessment

If proposing a new idea: They must write a structured research paper.

#### Paper Format (Max 6 Pages)

* Abstract
* Introduction
* Literature Review
* Methodology
* Experimental Design
* Results
* Risk Analysis
* Limitations
* Conclusion

**Must include:**
* Clear mathematical formulation
* Proper backtest design
* Economic interpretation
* Robustness tests
* Discussion of market impact and transaction costs

Code must still be functional and reproducible.

---

## What is not acceptable?

* Pure ML classification project without trading context
* No transaction cost modeling
* No out-of-sample validation
* Notebook-only submission
* Strategy with unrealistic turnover assumptions
* Vague alpha hypothesis