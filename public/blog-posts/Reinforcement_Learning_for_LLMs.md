---
title: "Reinforcement Learning for LLMs"
tags: ["reinforcement-learning", "llm", "tutorial"]
excerpt: "Basic knowledge for RL in LLMs' scenario. Start from multi-armed bandit model to PPO with detailed explanation and sample codes." Based on slides from Cornell Tech's CS 5740 Natural Language Processing. Refined by ChatGPT.
---
# Reinforcement Learning for LLMs

## Table of Contents
- [Reinforcement Learning for LLMs](#reinforcement-learning-for-llms)
  - [Table of Contents](#table-of-contents)
  - [Multi-Armed Bandit](#multi-armed-bandit)
    - [Settings](#settings)
    - [Formalization](#formalization)
    - [Policy Gradient in Multi-Armed Bandit](#policy-gradient-in-multi-armed-bandit)
    - [Log-derivative Trick](#log-derivative-trick)
    - [Code Simulation](#code-simulation)
      - [Problem Setup](#problem-setup)
      - [Pseudocode](#pseudocode)
      - [Python Implementation](#python-implementation)
  - [From Bandit to Contextual Bandit](#from-bandit-to-contextual-bandit)
    - [Model the LLMs](#model-the-llms)
    - [Formalization of Contextual Bandit](#formalization-of-contextual-bandit)
    - [Policy Gradient for Contextual Bandit](#policy-gradient-for-contextual-bandit)
  - [Proximal Policy Optimization (PPO): From Bandit to Full RL](#proximal-policy-optimization-ppo-from-bandit-to-full-rl)
    - [**Motivation: Beyond Bandits**](#motivation-beyond-bandits)
    - [Markov Decision Process (MDP)](#markov-decision-process-mdp)
    - [Policy Gradient: Trajectory Form](#policy-gradient-trajectory-form)
    - [Advantage Function](#advantage-function)
    - [**Proximal Policy Optimization (PPO)**](#proximal-policy-optimization-ppo)
    - [Minimal PPO Simulation](#minimal-ppo-simulation)
      - [Environment](#environment)
      - [PPO Objective in Bandit Setting](#ppo-objective-in-bandit-setting)
      - [Pseudocode](#pseudocode-1)
      - [Python Code](#python-code)
  - [Reward Modeling via Preference](#reward-modeling-via-preference)
    - [**Bradley-Terry Model**](#bradley-terry-model)
    - [Loss Function](#loss-function)
    - [Usage in PPO](#usage-in-ppo)
    - [Code Simulation](#code-simulation-1)

---


## Multi-Armed Bandit

### Settings

This is a very simple reinforcement learning setting, and we will start from here to gradually build the knowledge system for RL in LLMs.

Imagine there are multiple bandit in front of you, saying $N$. Every bandit has a chance to bring you a reward $r_i$ obeying an specific but unknown distribution $r_i\sim\mathcal D_i$. We only have limited chance to draw the bandit, so the question is how can we maximize our total reward in this limited budget. 

Intuitively, we just need to find out which bandit produces the highest reward, and keep playing with it. However, in this limited-budget seneria, if we decide to explore a new bandit, we will lose a chance to play with the current bandit, and we don't know whether the reward from another bandit will increase or not, which means, there will be risk. Therefore, we need to balance the **Exploration** and **Exploitation**.

### Formalization

Let's formalize this problem into a RL style. In RL, there are five key component: state $S$, initail state $s_0$, action $a$, transition $T$ and reward $r$.

- State

  - In the multi-armed bandit setting, since there is only single round of decision, there is no state, or you can say, only a initial state $s_0$.

- Action

  - Every round of playing, we only draw one action $a\in\mathcal A$ from a policy $a\sim\pi(\mathcal A)$. Here, $\pi$ is a distribution over the action set.

- Transition

  - No states, no transition, or you can say, only one transition from start to end.

- Reward

  - Every action $a$ on a bandit will bring a reward $r\sim\mathbb P(r|a)$.

- Goal

  - Maximize the expectaion of reward, which is
    $$
    J(\theta)=\mathbb E_{a\sim\pi_\theta}[r(a)]
    $$

> Recall the definition of **value function**, which is used to describe the value of a certain state $s$.
> $$
> v_{\pi_\theta}(s) = \mathbb E_{\pi_\theta}[G_t|s]
> $$
> $G_t$ is the return (accumulated reward on a trajetory) from a state $s$. In the bandit scenario, $G_t = r$, since the trajetory length is 1, and $s=s_0$. Therefore
> $$
> J(\theta) =\mathbb E_{a\sim\pi_\theta}[r(a)]=\mathbb E_{a\sim\pi_\theta}[r(a)|s_0] = \mathbb E_{\pi_\theta}[G_t|s_0]=v_{\pi_\theta}(s_0)
> $$
> Our goal is equivalent to maximize the value of initail state.

How can we maxized the function? Policy gradient is a powerful tool.

### Policy Gradient in Multi-Armed Bandit

Policy gradient is one of the most common methods to perform reinforcement learning, and it's the foundation of numerous advanced learning algorithm, such as PPO, DPO and GRPO.

In bandit settings, we want to perform following optimization
$$
\max_\theta J(\theta)=\mathbb E_{a\sim\pi_\theta}[r(a)] = \sum_{a}\pi_\theta(a)r(a)\\
\text{s.t. $\pi_\theta$ is a distribution over $\mathcal A$}
$$
We can use gradient ascent to $\theta$, so we calculate the derivative
$$
\nabla_\theta J(\theta) = \nabla_\theta \sum_a \pi_\theta(a) r(a)= \sum_a \nabla_\theta\pi_\theta(a)\cdot r(a)
$$
This term is intuitively hard to estimate (need to sum over all possible $a$), but we can use some tricks to turn it into a expectation, and then we can use sampling to estiname it.

### Log-derivative Trick

In single variable calculus, we have
$$
\frac{d(\log(f(x)))}{dx} = \frac{d(\log(f(x)))}{d(f(x))}\cdot \frac{d(f(x))}{dx} = \frac{1}{f(x)}\cdot\frac{d(f(x))}{dx}
$$
which means
$$
\frac{d(f(x))}{dx} = f(x)\frac{d(\log(f(x)))}{dx}
$$
Similarily, we have
$$
\nabla_\theta\pi_\theta(a)=\pi_\theta(a)\nabla_\theta\log(\pi_\theta(a))
$$
Therefore, we can rewrite the target function
$$
\begin{aligned}
\nabla_\theta J(\theta) &= \sum_a \nabla_\theta\pi_\theta(a)\cdot r(a)\\
&=\sum_a\pi_\theta(a)(\nabla_\theta\log(\pi_\theta(a))\cdot r(a))\\
&=\mathbb E_{a\sim\pi_\theta}[\nabla_\theta\log(\pi_\theta(a))\cdot r(a))]
\end{aligned}
$$
Great! Now we can use sampling to estimate the expectation. Namely, the loss for learning can be defined as 
$$
\mathcal L(\theta) = -\log(\pi_\theta(a))r(a)
$$

> Here, what we did is moving the $\nabla$ from the outside of $\mathbb E$ to inside of $\mathbb E$:
> $$
> \nabla_\theta J(\theta) =\nabla_\theta\mathbb E_{a\sim\pi_\theta}[...]=\mathbb E_{a\sim\pi_\theta}[\nabla_\theta...]
> $$
> The latter one can use sampling while the former one cannot.

### Code Simulation

To solidify our understanding, we now simulate a simple multi-armed bandit setting with N machines. Each machine has a fixed but unknown reward probability distribution. We aim to train a stochastic policy that learns to prefer arms with higher expected reward.

#### Problem Setup

- We define N arms (bandits).
- Each arm has a fixed reward distribution: $r_i \sim \mathcal{D}_i = \mathcal{N}(\mu_i, \sigma^2)$
- Our policy is a softmax distribution over actions $a \in \{0, …, N-1\}$
- Objective: Maximize expected reward using gradient ascent.

#### Pseudocode

```
Initialize policy parameters θ (size N)
For each iteration:
    Sample a batch of actions a_i ~ π_θ
    Pull the bandit arm a_i → observe reward r_i
    Compute log-prob gradients: ∇θ log π_θ(a_i)
    Multiply by reward r_i
    Average gradients and update θ by gradient ascent
```

#### Python Implementation

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---- Config ----
num_arms = 10
num_steps = 1000
batch_size = 32
lr = 0.1
torch.manual_seed(42)

# ---- Environment ----
true_means = torch.randn(num_arms)  # True reward means for each arm
true_stds = torch.ones(num_arms)

def pull_bandit(arm):
    # Return stochastic reward from N(mean, 1)
    return (torch.randn(1) * true_stds[arm]).item() + true_means[arm].item()

# ---- Policy Model ----
# Simple softmax over learnable logits
class BanditPolicy(nn.Module):
    def __init__(self, num_arms):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_arms))

    def forward(self):
        return torch.softmax(self.logits, dim=0)

# ---- Training Loop ----
policy = BanditPolicy(num_arms)
optimizer = optim.Adam(policy.parameters(), lr=lr)

reward_history = []

for step in range(num_steps):
    probs = policy()
    dist = torch.distributions.Categorical(probs)
    actions = dist.sample((batch_size,))
    rewards = torch.tensor([pull_bandit(a) for a in actions])

    log_probs = dist.log_prob(actions)
    loss = -(log_probs * rewards).mean()  # Policy gradient loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    reward_history.append(rewards.mean().item())

    if (step + 1) % 100 == 0:
        print(f"Step {step+1}, Avg Reward: {np.mean(reward_history[-100:]):.3f}")

# ---- Visualization (optional) ----
import matplotlib.pyplot as plt
plt.plot(reward_history)
plt.xlabel("Step")
plt.ylabel("Average Reward (per batch)")
plt.title("Learning Curve: Policy Gradient on Multi-Armed Bandit")
plt.show()
```

For this simple example, 100 steps lead to converge.

![learning curve](blog-posts/imgs/learning%20curve.png)

![arm freq](blog-posts/imgs/arm%20freq.png)

![arm reward](blog-posts/imgs/arm%20reward.png)

## From Bandit to Contextual Bandit

The concepts this session are improtant, because they bridge the gap between RL and LLMs.

### Model the LLMs

For next-token-prediction style LLMs, the procedure of token generation can be regarded as a sequence decesion, thus it's a standard RL problem. The current state is all the prefix, the action is generating a new token, the policy is the probability distrubution on the vocab and the transition is just concatenation. 

Indeedly, we can define the reward on token level, but this makes the reward hard to define. Therefore, we regard the entire sequence generated as single action, which means, LLMs are now converted into bandits, and every time we pull the arm, the bandit will produce an entire sequence. This makes it easier to tell which action is better.

> E.g.
>
> Q: What is on the road?
>
> A1: There are <u>some</u> people.
>
> A2: There are <u>a</u> bunch of people.
>
> It's hard to tell "some" and "a" which is better, but you can easily define your preference on A1 and A2.

### Formalization of Contextual Bandit

In contextual bandits, each decision is conditioned on a context $x$ (e.g., a prompt), and the agent selects an action $y$ (e.g., a response). The reward is then received based on the tuple $(x, y)$. Formally:

- **Context**: $x \in \mathcal{X}$

- **Action**: $y \in \mathcal{Y}$ drawn from policy $\pi_\theta(y|x)$

- **Reward**: $r \sim \mathbb{P}(r|x, y)$

- **Objective**:
  $$
  J(\theta)=\mathbb E_{x\sim\mathcal D}[\mathbb E_{y\sim\pi_\theta(\cdot|x)}[r(x,y)]]
  $$

This formulation matches well with LLM learning from preference data, where $x$ is the prompt and $y$ is the generated response.

### Policy Gradient for Contextual Bandit

We apply the log-derivative trick again:
$$
\begin{aligned}
\nabla_\theta J(\theta)&=\mathbb E_{x\sim\mathcal D}[\mathbb E_{y\sim\pi_\theta(\cdot|x)}[r(x,y)]] \\
&=\mathbb E_{x\sim\mathcal D}[\sum_{y}\pi(y|x)\nabla_\theta\log\pi_\theta(y|x)\cdot r(x, y)]\\
&=\mathbb E_{x\sim\mathcal D}[\mathbb E_{y\sim\pi_\theta(\cdot|x)}[\nabla_\theta\log\pi_\theta(y|x)r(x,y)]]
\end{aligned}
$$
In practice, this means:

1. Sample $x$ from the dataset.
2. Sample $y \sim \pi_\theta(\cdot|x)$.
3. Evaluate reward $r(x, y)$.
4. Compute gradient $\nabla_\theta \log \pi_\theta(y|x) \cdot r(x, y)$.

This forms the basis for reinforcement learning algorithms such as REINFORCE, PPO, and DPO applied to LLMs.



## Proximal Policy Optimization (PPO): From Bandit to Full RL

We’ve seen how LLMs can be modeled as contextual bandits. However, in many realistic tasks, token-by-token reward signals, temporal credit assignment, or exploration over multi-turn dialogs require us to go beyond bandits — into the full realm of reinforcement learning (RL). This is where Proximal Policy Optimization (PPO) comes into play.

### **Motivation: Beyond Bandits**

In contextual bandits, the response is one-shot: we generate the full response and get a scalar reward. But what if we want to:

- Assign partial credit to good or bad segments of a long generation?
- Penalize hallucinations mid-sentence?
- Reward helpful clarifications near the end?

This requires **stepwise feedback** — a setting where **state**, **action**, and **reward** evolve over time.

This naturally leads us to **Markov Decision Processes (MDP)**.

### Markov Decision Process (MDP)

An MDP consists of:

- States $s_t$: current context (e.g., prefix tokens).
- Actions $a_t$: next token to generate.
- Rewards $r_t$: feedback after action.
- Transitions $s_{t+1} = s_t \cup \{a_t\}$: next context is previous context + new token.
- Policy $\pi_\theta(a_t | s_t)$: a distribution over tokens given context.

In this formulation, LLM generation is modeled as a trajectory:
$$
\tau = (s_0, a_0, r_0), (s_1, a_1, r_1), …, (s_T, a_T, r_T)
$$
The objective is to **maximize expected return**:
$$
J(\theta) = \mathbb{E}{\tau \sim \pi\theta} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]
$$

### Policy Gradient: Trajectory Form

Using the policy gradient theorem:
$$
\nabla_\theta J(\theta) = \mathbb{E}{\tau \sim \pi\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t \right]
$$
Here, $G_t = \sum_{t’=t}^T \gamma^{t’-t} r_{t’}$ is the return from step t.

However, this estimator has high variance. That’s where **Advantage** comes in.

### Advantage Function

The **advantage function** compares how much better (or worse) an action is, compared to the average action at that state:
$$
A_t = G_t - V(s_t)
$$

- $G_t$: actual return from trajectory.
- $V(s_t)$: estimated value function — expected return from that state under policy.

Using $A_t$, we rewrite the gradient:
$$
\nabla_\theta J(\theta) = \mathbb{E} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot A_t \right]
$$
This is the core idea of **actor-critic methods**:

- **Actor**: $\pi_\theta(a|s)$ is trained with policy gradient.
- **Critic**: $V(s)$ is trained with TD/MSE loss to estimate return.

> Why subtracting a baseline doesn't change the gradient?
>
>  Given the original policy gradient:
> $$
> \nabla_\theta J(\theta) = \mathbb{E} [ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t ]
> $$
> we define a baseline $b(s_t)$ which is independent of action $a_t$. We can have
> $$
> \begin{aligned}
> \nabla_\theta J(\theta)& = \mathbb{E} [ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b(s_t)) ]\\
> &= \mathbb{E} [ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t ]-\mathbb{E} [ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot b(s_t) ]
> \end{aligned}
> $$
> For the second term, we have
> $$
> \mathbb{E}[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot b(s_t)] = b(s_t)\cdot\nabla_\theta\sum_{a_t}\pi_\theta(a_t|s_y)=b(s_t)\nabla_\theta 1 = 0
> $$

### **Proximal Policy Optimization (PPO)**

While the above gradient works, naïvely updating the policy with large steps may destabilize training. PPO introduces a **conservative surrogate objective**:
$$
L^{\text{PPO}}(\theta) = \mathbb{E}_t[ \min \left( r_t(\theta) A_t, \ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right)]
$$
Where:

- $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$ is the likelihood ratio.
- $\epsilon$ is a hyperparameter (e.g. 0.2).

This encourages updates **only if** the new policy doesn’t deviate too far from the old one.

### Minimal PPO Simulation

To consolidate our understanding of PPO, we simulate a minimal PPO loop in a simple setting. The environment is a **contextual bandit** with fixed reward function. This avoids the complexity of fitting a reward model while allowing us to learn a policy via PPO.

#### Environment

We consider a contextual bandit setting:

- Each state is a context vector $x \in \mathbb{R}^d$.

- There are $K$ actions, each associated with a fixed linear weight vector $w_a \in \mathbb{R}^d$.

- The reward for taking action a in context $x$ is:
  $$
  r(x, a) = w_a^T x + \epsilon,\quad \epsilon \sim \mathcal{N}(0, 0.1)
  $$

- The goal is to learn a stochastic policy $\pi_\theta(a|x)$ to maximize the expected reward.

#### PPO Objective in Bandit Setting

Since the environment has no temporal dynamics (i.e., one-step episodes), we can estimate the advantage as:
$$
A_t = r(x_t, a_t)
$$
We don't substract baseline first, or you can consider the baseline is $0$ for initail state.

The PPO clipped objective becomes:
$$
L^{\text{clip}}(\theta) = \mathbb{E}_t \left[ \min\left(r_t(\theta) A_t,\ \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) A_t\right) \right]
$$
where:

- $r_t(\theta) = \frac{\pi_\theta(a_t|x_t)}{\pi_{\theta_{\text{old}}}(a_t|x_t)}$

#### Pseudocode

```text
Initialize:
    - Context dimension d
    - Number of actions K
    - Policy network π_θ(a | s) with parameters θ
    - Learning rate η, clipping range ε
    - Reward environment with fixed reward weights for each arm

Repeat for each training step:
    1. Sample a batch of contexts s_i ~ N(0, I) for i = 1 ... B
    2. Compute action probabilities π_θ(a | s_i) for each context
    3. Sample actions a_i ~ π_θ(· | s_i)
    4. Get rewards r_i = reward(s_i, a_i)
    5. Store old log-probs: log π_θ(a_i | s_i)
    6. Estimate advantages A_i ← r_i (no baseline for simplicity)

    For each PPO epoch (e.g. 4 times):
        a. Recompute action probabilities π_θ′(a | s_i)
        b. Compute new log-probs log π_θ′(a_i | s_i)
        c. Compute ratio: ρ_i = exp(log π_θ′ - log π_θ)
        d. Compute clipped objective:
            L_i = min(ρ_i * A_i, clip(ρ_i, 1 - ε, 1 + ε) * A_i)
        e. Update θ by maximizing mean L_i over batch

    Log average reward and optionally visualize learning curve
```

> Why there are two loops?
>
> PPO training consists of two distinct loops:
>
> 1. **Outer Loop: Trajectory Collection**
>
> This loop collects data by interacting with the environment:
>
> - Sample contexts (states), actions, rewards, and log-probabilities using the current policy.
> - The collected data is **frozen** (not updated during optimization), serving as the *reference behavior*.
> - This data is considered **on-policy**, but only for a short time—hence PPO is often called *“almost on-policy.”*
>
> Once a batch of data is collected, we reuse it for multiple updates:
>
> 2. **Inner Loop: Multiple Epochs of Optimization**
>
> - We update the policy **several times** on the same batch to improve sample efficiency.
> - To prevent excessive deviation from the original policy used during sampling, PPO applies a **clipped objective** to ensure stability.
> - This design balances learning efficiency with policy stability.

#### Python Code

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ---- Config ----
context_dim = 5
num_actions = 10
num_steps = 100
batch_size = 64
ppo_clip = 0.2
lr = 1e-2
torch.manual_seed(0)

# ---- Environment ----
class ContextualBanditEnv:
    def __init__(self, context_dim, num_actions):
        self.weights = torch.randn(num_actions, context_dim)  # reward weights

    def get_batch(self, batch_size):
        context = torch.randn(batch_size, context_dim)
        return context

    def get_reward(self, context, actions):
        w = self.weights[actions]  # (B, d)
        reward = (context * w).sum(dim=1) + 0.1 * torch.randn_like(context[:, 0])
        return reward

env = ContextualBanditEnv(context_dim, num_actions)

# ---- Policy Network ----
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions)
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=lr)

# ---- Training Loop ----
reward_trace = []
rewards_0 = []
actions_0 = []
for step in range(num_steps):
    with torch.no_grad():
        context = env.get_batch(batch_size)
        probs = policy(context)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        old_log_probs = dist.log_prob(actions)
        rewards = env.get_reward(context, actions)
        advantages = rewards  # no baseline
        rewards_0.append(env.weights @ context[0])
        actions_0.append(actions[0].item())

    # PPO update
    for _ in range(4):  # multiple epochs
        probs = policy(context)
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions)

        ratio = torch.exp(new_log_probs - old_log_probs)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip) * advantages
        loss = -torch.mean(torch.min(unclipped, clipped))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    reward_trace.append(rewards.mean().item())

    if (step + 1) % 50 == 0:
        print(f"Step {step+1}, Avg Reward: {reward_trace[-1]:.3f}")
        print("Policy probs:", probs[0].detach().numpy())

# ---- Visualization ----
plt.figure(figsize=(8, 4))

# 1. Learning Curve
plt.subplot(1, 2, 1)
plt.plot(reward_trace)
plt.xlabel("Training Step")
plt.ylabel("Avg Reward")
plt.title("PPO Learning Curve")

max_reward_action = [reward.argmax() for reward in rewards_0]
# 2. Action Comparison over Time
plt.subplot(1, 2, 2)
plt.plot(max_reward_action, label='Best Action (Reward)', linestyle='--')
plt.plot(actions_0, label='Chosen Action', alpha=0.7)
plt.xlabel("Training Step")
plt.ylabel("Action Index")
plt.title("Chosen vs Best Action Over Time")
plt.legend()


plt.tight_layout()
plt.show()
```

![ppo](blog-posts/imgs/ppo.png)

## Reward Modeling via Preference

To estimate the advantage in RLHF, we need a way to obtain the reward signal. But for open-ended generation tasks, assigning a scalar reward to each token is challenging. One practical alternative is to use preference data.

### **Bradley-Terry Model**

The **Bradley-Terry model** offers a probabilistic way to connect pairwise preferences to scalar scores. It assumes:
$$
p(a > b) = \sigma(s(a) - s(b))
$$
where:

- $s(\cdot)$ is the score (or reward) assigned to an output,
- $\sigma$ is the sigmoid function.

This gives us a differentiable objective that allows us to learn $s(\cdot)$ from pairwise preference data.

### Loss Function

We minimize the negative log-likelihood over a dataset $\mathcal{D}$ of preference tuples:
$$
\mathcal{L}_r(\psi, \mathcal{D}) = - \mathbb{E}_{(\bar{x}, \bar{y}_w, \bar{y}_l) \sim \mathcal{D}} \left[ \log \sigma\left(r_\psi(\bar{x}, \bar{y}_w) - r_\psi(\bar{x}, \bar{y}_l)\right) \right]
$$
Here:

- $\bar{x}$ is the prompt,
- $\bar{y}_w, \bar{y}_l$ are the **preferred** and **less preferred** responses,
- $r_\psi$ is a reward model (e.g., a transformer) with learnable parameters $\psi$.

> Intuition:
>
> This objective pushes the score of the better response higher than that of the worse one. If the model outputs very close scores, the loss is large; if the difference matches the observed preference, the loss is small.

### Usage in PPO

Once the reward model is trained, we can use it to compute $r(\bar{x}, \bar{y})$, and then:

- Compute advantage using reward $r$ and value baseline $V$
- Proceed with PPO updates

This connects **preference modeling** and **policy optimization** in RLHF pipelines.

### Code Simulation

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Simple tokenizer
def encode(x, y):
    return torch.tensor([hash(w) % 10000 for w in (x + " " + y).split()])

class RewardModel(nn.Module):
    def __init__(self, vocab_size=10000, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch_pairs):
        scores = []
        for x, y in batch_pairs:
            input_ids = encode(x, y)
            embeds = self.embedding(input_ids)
            pooled = embeds.mean(dim=0)
            score = self.scorer(pooled)
            scores.append(score)
        return torch.cat(scores, dim=0)

# Simulated dataset
pairs = [("sky is", "blue", "green"),
         ("apple is", "red", "blue"),
         ("banana is", "yellow", "purple")]

model = RewardModel()
optimizer = optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(50):
    batch = [(x, y_w) for x, y_w, _ in pairs] + [(x, y_l) for x, _, y_l in pairs]
    labels = torch.tensor([1.0] * len(pairs) + [0.0] * len(pairs))
    scores_w = model([(x, y_w) for x, y_w, _ in pairs])
    scores_l = model([(x, y_l) for x, _, y_l in pairs])
    logits = scores_w - scores_l
    loss = loss_fn(logits, torch.ones_like(logits))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, loss={loss.item():.4f}")
```

