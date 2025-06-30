---
title: "Reinforcement Learning Notes: From Basics to Planning"
date: "2025-06-28"
tags: ["reinforcement-learning", "ai", "class-note"]
excerpt: "An introduction to basic RL concept. Based on *Machine Learning* instructed by Prof. Mingsheng Long from School of Software, Tsinghua University. Refined by ChatGPT."
---

# Reinforcement Learning Notes: From Basics to Planning

## Table of Contents

- [Reinforcement Learning Notes: From Basics to Planning](#reinforcement-learning-notes-from-basics-to-planning)
  - [Table of Contents](#table-of-contents)
  - [What is Reinforcement Learning?](#what-is-reinforcement-learning)
  - [The Agent-Environment Loop](#the-agent-environment-loop)
  - [States, Observations, and Rewards](#states-observations-and-rewards)
  - [Understanding Markov Reward Processes (MRPs)](#understanding-markov-reward-processes-mrps)
  - [Solving MRPs with Bellman Equations](#solving-mrps-with-bellman-equations)
  - [Moving to Markov Decision Processes (MDPs)](#moving-to-markov-decision-processes-mdps)
  - [Value Functions and What They Tell Us](#value-functions-and-what-they-tell-us)
  - [Learning via Bellman Equations in MDPs](#learning-via-bellman-equations-in-mdps)
  - [What Makes a Policy Optimal?](#what-makes-a-policy-optimal)
  - [Planning in Known Environments](#planning-in-known-environments)
    - [Two Core Tasks](#two-core-tasks)
    - [Policy Evaluation via Iteration](#policy-evaluation-via-iteration)
    - [Policy Improvement](#policy-improvement)
    - [Policy Iteration: Putting It Together](#policy-iteration-putting-it-together)

---

## What is Reinforcement Learning?

Reinforcement Learning (RL) is a framework where an agent interacts with an environment to learn behaviors that maximize cumulative rewards over time. Think of it as trial-and-error learning driven by feedback signals (rewards).

---

## The Agent-Environment Loop

At each timestep:

* The **agent** receives an observation $O_t$ and a reward $R_t$, and chooses an action $A_t$ based on its policy.
* The **environment** responds by transitioning to a new state $S_{t+1}$, returning a new observation $O_{t+1}$ and reward $R_{t+1}$.

This forms a feedback loop fundamental to learning.

---

## States, Observations, and Rewards

* The **history** $H_t$ contains everything the agent has seen so far.
* The **state** $S_t$ is often a function of the history ($S_t = f(H_t)$) and serves as a sufficient summary.
* **Rewards** are scalar signals that indicate the desirability of the outcomes.
* If observations perfectly represent states, the process is a **Markov Decision Process (MDP)**. Otherwise, it's a **Partially Observable MDP (POMDP)**.

---

## Understanding Markov Reward Processes (MRPs)

An MRP models the environment without actions:

* States $\mathcal{S}$
* Transition probabilities $\mathcal{P}_{ss'} = \mathbb{P}(S_{t+1}=s'|S_t=s)$
* Rewards $\mathcal{R}_s = \mathbb{E}[R_{t+1}|S_t=s]$
* Discount factor $\gamma \in [0,1]$

**Goal**: Learn the **value function**:
$v(s) = \mathbb{E}[G_t | S_t = s], \text{ where } G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$

---

## Solving MRPs with Bellman Equations

The value function satisfies:
$v(s) = \mathcal{R}_s + \gamma \sum_{s'} \mathcal{P}_{ss'} v(s')$

In matrix form:
$\boldsymbol{v} = (\boldsymbol{I} - \gamma \boldsymbol{P})^{-1} \boldsymbol{\mathcal{R}}$

This gives a way to compute $v$ analytically.

---

## Moving to Markov Decision Processes (MDPs)

MDPs add the agent's actions:

* Action set $\mathcal{A}$
* Transition probability $\mathcal{P}_{ss'}^a$
* Reward function $\mathcal{R}_s^a$
* Policy $\pi(a|s)$: probability of choosing action $a$ in state $s$

---

## Value Functions and What They Tell Us

To evaluate how good a policy $\pi$ is:

* **State-value**:
  $v_\pi(s) = \mathbb{E}_\pi[G_t | S_t = s] = \sum_a \pi(a|s) q_\pi(s,a)$
* **Action-value**:
  $q_\pi(s,a) = \mathcal{R}_s^a + \gamma \sum_{s'} \mathcal{P}_{ss'}^a v_\pi(s')$

These functions guide better decision-making.

---

## Learning via Bellman Equations in MDPs

Bellman equations describe the recursive structure of the value functions:

* State-value:
  $v_\pi(s) = \sum_a \pi(a|s)(\mathcal{R}_s^a + \gamma \sum_{s'} \mathcal{P}_{ss'}^a v_\pi(s'))$
* Action-value:
  $q_\pi(s,a) = \mathcal{R}_s^a + \gamma \sum_{s'} \mathcal{P}_{ss'}^a \sum_{a'} \pi(a'|s') q_\pi(s',a')$

---

## What Makes a Policy Optimal?

We aim to find a policy $\pi_*$ such that:
$v_*(s) = \max_\pi v_\pi(s), \quad q_*(s,a) = \max_\pi q_\pi(s,a)$

The Bellman optimality equations are:

$$
\begin{aligned}
q_*(s,a) &= \mathcal{R}_s^a + \gamma \sum_{s'} \mathcal{P}_{ss'}^a \max_{a'} q_*(s',a') \\
v_*(s) &= \max_a q_*(s,a)
\end{aligned}
$$

A deterministic optimal policy is:
$\pi_*(s) = \arg\max_a q_*(s,a)$

---

## Planning in Known Environments

If the MDP model is fully known, we can plan ahead using **Dynamic Programming**.

### Two Core Tasks

1. **Policy Evaluation**: Compute $v_\pi(s)$
2. **Policy Control**: Find $\pi_*$

### Policy Evaluation via Iteration

$v_{k+1}(s) = \sum_a \pi(a|s)(\mathcal{R}_s^a + \gamma \sum_{s'} \mathcal{P}_{ss'}^a v_k(s'))$

### Policy Improvement

$\pi'(s) = \arg\max_a q_\pi(s,a)$

### Policy Iteration: Putting It Together

Alternate between evaluation and improvement until the policy stops changing. This gives an optimal policy.

**Policy Improvement Theorem**:
If $q_\pi(s, \pi'(s)) \ge v_\pi(s)$ for all $s$, then $v_{\pi'}(s) \ge v_\pi(s)$.

---

This concludes the foundational tutorial on reinforcement learning. Future sections may include topics like temporal difference learning, actor-critic methods, or deep RL extensions.
