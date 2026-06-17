---
title: "From DDPM to Flow Matching"
date: "2026-06-17"
tags: ["diffusion", "flow matching", "tutorial"]
excerpt: Inspect diffusion models (DDPM/DDIM) and flow matching from a low-level *geometric* view. This intuitive level helps build a feel for the shared core and the different implementations of these algorithms. Refined by Claude.
---

> Extended & annotated version of [Diffusion Meets Flow Matching: Two Sides of the Same Coin](https://diffusionflow.github.io/).
>
> In this blog we inspect diffusion models (DDPM/DDIM) and flow matching from a low-level *geometric* view. This intuitive level helps build a feel for the shared core and the different implementations of these algorithms. More involved mathematics will be introduced in later posts.

## Recap

**Notation.**
- For diffusion we use a discrete time $t \in \{0, 1, \dots, T\}$, where $t=0$ is the clean image and $t=T$ is pure noise. For flow matching we use a continuous time $t \in [0,1]$.
- $\alpha_t$ and $\sigma_t$ are **cumulative** coefficients (the per-step retention rate is $\sqrt{1-\beta_t} = \alpha_t / \alpha_{t-1}$). We use $\alpha_t, \sigma_t$ purely to simplify notation.

### Forward Process in DDPM & DDIM

Recall the basic setup of DDPM. It is a Markov process where, at each step $t$, a noise $\boldsymbol{\epsilon}_t$ is added to corrupt the original observation $\mathbf{x}_0$:

$$
\mathbf{x}_t = \sqrt{1-\beta_t}\,\mathbf{x}_{t-1} + \sqrt{\beta_t}\,\boldsymbol{\epsilon}_t, \quad \boldsymbol{\epsilon}_t \sim \mathcal{N}(0, \mathbf{I}).
$$

> Using $\sqrt{\beta_t}$ and $\sqrt{1-\beta_t}$ is called **Variance-Preserving (VP)**. Intuitively it has several benefits, such as keeping the "energy" of the image unchanged. It also has a deep correspondence with the discrete Ornstein–Uhlenbeck process (a *mean-reverting* process).

By the Markov property, this is equivalent to one-step noising:

$$
\mathbf{x}_t = \alpha_t\,\mathbf{x}_0 + \sigma_t\,\boldsymbol{\epsilon}_t, \quad \boldsymbol{\epsilon}_t \sim \mathcal{N}(0, \mathbf{I}),\ \alpha_t^2 + \sigma_t^2 = 1,
$$

where $\alpha_t$ and $\sigma_t$ are determined by the $\beta_i$'s.

So if we train a model to predict either $\boldsymbol{\epsilon}$ or $\mathbf{x}_0$, we can reverse the process to predict a less-noisy image. Usually the model predicts the noise $\hat{\boldsymbol{\epsilon}}_t$, so

$$
\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t - \sigma_t\,\hat{\boldsymbol{\epsilon}}_t}{\alpha_t}.
$$

There are two ways to obtain a less-noisy image:
1. **Re-noise:** recover the clean image, then project it back to a lower noise level.
2. **Interpolation:** compute a "direction" and transform directly from a higher noise level to a lower one.

### Re-noise

Typical DDPM and DDIM implementations use the re-noise method. We solve for $\hat{\mathbf{x}}_0$ and re-noise to $\hat{\mathbf{x}}_{t-1}$:

$$
\begin{aligned}
\hat{\boldsymbol{\epsilon}}_t &= p_\theta(\mathbf{x}_t, t), \quad \hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t - \sigma_t\,\hat{\boldsymbol{\epsilon}}_t}{\alpha_t}, \\
\hat{\mathbf{x}}_{t-1} &= \alpha_{t-1}\,\hat{\mathbf{x}}_0 + \sigma_{t-1}\,\boldsymbol{\epsilon}_{t-1}.
\end{aligned}
$$

Here we need to design the new noise $\boldsymbol{\epsilon}_{t-1}$. A very naive choice is to draw a fresh $\boldsymbol{\epsilon}_{t-1} \sim \mathcal{N}(0, \mathbf{I})$ — let's call this ***"He's sampling"*** (name after myself; it is just "resample everything from scratch"). This is a valid operation, but it is unstable and throws away the information contained in $\hat{\boldsymbol{\epsilon}}_t$.

We can do better by using $\hat{\boldsymbol{\epsilon}}_t$ to construct $\boldsymbol{\epsilon}_{t-1}$. The connection between $\boldsymbol{\epsilon}_{t-1}$ and $\boldsymbol{\epsilon}_t$ is intuitive: $\boldsymbol{\epsilon}_t$ is obtained by adding *more* noise on top of $\boldsymbol{\epsilon}_{t-1}$.

$$
\begin{aligned}
\mathbf{x}_{t-1} &= \alpha_{t-1}\,\mathbf{x}_0 + \sigma_{t-1}\,\boldsymbol{\epsilon}_{t-1}, \\
\mathbf{x}_t &= \sqrt{1-\beta_t}\,\mathbf{x}_{t-1} + \sqrt{\beta_t}\,\boldsymbol{\epsilon}', \quad \boldsymbol{\epsilon}' \sim \mathcal{N}(0, \mathbf{I}) \\
&= \alpha_{t-1}\sqrt{1-\beta_t}\,\mathbf{x}_0 + \underbrace{\sigma_{t-1}\sqrt{1-\beta_t}}_{c}\,\boldsymbol{\epsilon}_{t-1} + \underbrace{\sqrt{\beta_t}}_{d}\,\boldsymbol{\epsilon}' \\
&= \alpha_t\,\mathbf{x}_0 + c\,\boldsymbol{\epsilon}_{t-1} + d\,\boldsymbol{\epsilon}'.
\end{aligned}
$$

Comparing with $\mathbf{x}_t = \alpha_t\,\mathbf{x}_0 + \sigma_t\,\boldsymbol{\epsilon}_t$, we get

$$
\sigma_t\,\boldsymbol{\epsilon}_t = c\,\boldsymbol{\epsilon}_{t-1} + d\,\boldsymbol{\epsilon}', \quad \boldsymbol{\epsilon}' \sim \mathcal{N}(0, \mathbf{I}),\ c^2 + d^2 = \sigma_t^2 \ \text{(easily verified)}.
$$

So the question becomes: **given $\hat{\boldsymbol{\epsilon}}_t$, what is a good estimate of $\boldsymbol{\epsilon}_{t-1}$?** This is a Gaussian inference problem.

> **A recap from probability theory.**
> Given $x, y \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0,1)$ and their linear combination $S = ax + by$, we want the conditional distribution $p(x \mid S)$. By Bayes' rule,
> $$ p(x \mid S) \propto p(x)\,p(S \mid x), \qquad S \mid x \sim \mathcal{N}(ax, b^2). $$
> Expanding the Gaussians,
> $$\begin{aligned}
   p(x \mid S) &\propto \exp\!\left(-\tfrac{x^2}{2}\right)\exp\!\left(-\tfrac{(S-ax)^2}{2b^2}\right) \\
   &\propto \exp\!\left(-\tfrac{1}{2}\left(\tfrac{a^2+b^2}{b^2}x^2 - \tfrac{2aS}{b^2}x + C\right)\right).
   \end{aligned}$$
> Completing the square gives
> $$ \mu = \frac{aS}{a^2+b^2}, \qquad \sigma^2 = \frac{b^2}{a^2+b^2}. $$
>
> ![bayes](blog-posts/imgs/bayes.png)
>
> Geometrically: $x, y$ form an isotropic Gaussian (circular contours). Observing $S$ restricts us to a line perpendicular to $(a,b)$; the conditional mean is the projection of the closest point onto the $x$-axis, and the conditional variance is the leftover spread along that line.

Applying this with $a = c$, $b = d$, and observed value $S = \sigma_t\,\hat{\boldsymbol{\epsilon}}_t$ (so $a^2+b^2 = \sigma_t^2$):

$$
\boldsymbol{\epsilon}_{t-1} \mid \hat{\boldsymbol{\epsilon}}_t \sim \mathcal{N}\!\left(\frac{c\,\sigma_t\,\hat{\boldsymbol{\epsilon}}_t}{\sigma_t^2},\ \frac{d^2}{\sigma_t^2}\right) = \mathcal{N}\!\left(\frac{c}{\sigma_t}\,\hat{\boldsymbol{\epsilon}}_t,\ \frac{d^2}{\sigma_t^2}\right).
$$

Sampling from this posterior (mean plus standard deviation times a fresh noise $z$):

$$
\hat{\boldsymbol{\epsilon}}_{t-1} = \frac{c}{\sigma_t}\,\hat{\boldsymbol{\epsilon}}_t + \frac{d}{\sigma_t}\,z, \quad z \sim \mathcal{N}(0, \mathbf{I}).
$$

Note that $z$ is a **freshly sampled** standard normal — it is *not* the forward noise $\boldsymbol{\epsilon}'$ (which we cannot observe); it represents the irreducible part of $\boldsymbol{\epsilon}_{t-1}$ that $\hat{\boldsymbol{\epsilon}}_t$ cannot recover. The re-noised result is

$$
\hat{\mathbf{x}}_{t-1} = \alpha_{t-1}\,\hat{\mathbf{x}}_0 + \sigma_{t-1}\!\left(\frac{c}{\sigma_t}\,\hat{\boldsymbol{\epsilon}}_t + \frac{d}{\sigma_t}\,z\right).
$$

All of $\alpha, \sigma, c, d$ are determined by the $\beta_i$'s. This is the standard DDPM procedure. Because we introduce a fresh $z$, sampling is **stochastic**.

We can generalize the split between "reuse $\hat{\boldsymbol{\epsilon}}_t$" and "inject fresh noise" with a controller $\eta$:

$$
\hat{\boldsymbol{\epsilon}}_{t-1} = \sqrt{1-\eta^2}\,\hat{\boldsymbol{\epsilon}}_t + \eta\,z, \quad \eta \in [0, 1].
$$

- $\eta = 1$: everything is resampled — this is **He's sampling**, fully stochastic.
- $\eta = 0$: no fresh noise — this is **deterministic DDIM**:
$$
\hat{\mathbf{x}}_{t-1} = \alpha_{t-1}\,\hat{\mathbf{x}}_0 + \sigma_{t-1}\,\hat{\boldsymbol{\epsilon}}_t.
$$
- **DDPM** is *not* an arbitrary $\eta$: it is the one specific value matching the posterior we derived above, namely $\eta = d/\sigma_t = \sqrt{\beta_t}/\sigma_t$. In other words, DDPM is the "honest" choice that injects exactly the irreducible residual and reuses everything recoverable.

The benefit of the deterministic procedure is that, in principle, we can jump to **any** time step $s < t$:

$$
\hat{\mathbf{x}}_s = \alpha_s\,\hat{\mathbf{x}}_0 + \sigma_s\,\hat{\boldsymbol{\epsilon}}_t.
$$

This already looks a lot like an interpolation with a step parameter $s$.

### Interpolation

Here we only discuss the deterministic procedure (DDIM).

Given a ground-truth image $\mathbf{x}_0$ and a Gaussian noise $\boldsymbol{\epsilon}$, there is a variance-preserving path $p$ connecting them. Reusing our notation,

$$
\mathbf{x}_t = \alpha_t\,\mathbf{x}_0 + \sigma_t\,\boldsymbol{\epsilon}, \quad \alpha_t^2 + \sigma_t^2 = 1,\ \alpha_t = \cos\tfrac{\pi t}{2},\ \sigma_t = \sin\tfrac{\pi t}{2},\ t \in [0,1],
$$

so $t=0$ gives $\mathbf{x}_0$ (clean) and $t=1$ gives $\boldsymbol{\epsilon}$ (pure noise).

**This path is, in general, an arc of an *ellipse* — and it is curved.** It reduces to a *circular* arc only when $\mathbf{x}_0$ and $\boldsymbol{\epsilon}$ happen to be orthogonal and of equal length (e.g. in the abstract basis where we treat them as the coordinate axes).

> **Why an ellipse?** Write the curve as $\mathbf{p}(\theta) = \cos\theta\,\mathbf{x}_0 + \sin\theta\,\boldsymbol{\epsilon}$ and stack the two vectors into a matrix $A = [\,\mathbf{x}_0\ \ \boldsymbol{\epsilon}\,]$, so $\mathbf{p} = A(\cos\theta, \sin\theta)^\top$. Then $(\cos\theta, \sin\theta)^\top = A^{-1}\mathbf{p}$, and substituting into $\cos^2\theta + \sin^2\theta = 1$ gives
> 
> $$\mathbf{p}^\top \underbrace{(A^{-\top} A^{-1})}_{M}\,\mathbf{p} = 1.$$
> $M$ is symmetric positive-definite, and $\mathbf{p}^\top M \mathbf{p} = 1$ is exactly the equation of an origin-centered ellipse. It degenerates to a circle iff $A^\top A \propto I$, i.e. $\mathbf{x}_0 \perp \boldsymbol{\epsilon}$ with equal norm.

![path](blog-posts/imgs/path_sample.png)

This representation is literally identical to the forward process of diffusion:

$$
\mathbf{x}_t = \alpha_t\,\mathbf{x}_0 + \sigma_t\,\boldsymbol{\epsilon}_t, \quad \boldsymbol{\epsilon}_t \sim \mathcal{N}(0, \mathbf{I}),\ \alpha_t^2 + \sigma_t^2 = 1.
$$

So after predicting $\hat{\boldsymbol{\epsilon}}_t$ from $\mathbf{x}_t$, we are effectively predicting a whole path $\hat{p}_t$ that passes through $\mathbf{x}_t$ with clean end $\hat{\mathbf{x}}_0$ and noise direction $\hat{\boldsymbol{\epsilon}}_t$:

$$
\hat{p}_t \coloneqq \{\, \mathbf{x}_\tau = \alpha_\tau\,\hat{\mathbf{x}}_0 + \sigma_\tau\,\hat{\boldsymbol{\epsilon}}_t \mid \tau \in [0, 1] \,\}.
$$

To obtain a data point at any noise level $s$, we interpolate along this path:

$$
\hat{\mathbf{x}}_s = \alpha_s\,\hat{\mathbf{x}}_0 + \sigma_s\,\hat{\boldsymbol{\epsilon}}_t.
$$

Note that $\hat{p}_t$ depends on $t$ (through the predicted $\hat{\boldsymbol{\epsilon}}_t$), so interpolating far from $\mathbf{x}_t$ becomes inaccurate. This is why DDIM cannot skip too many steps.

#### Direction & v-prediction

Intuitively, the direction from $\mathbf{x}_0$ to $\boldsymbol{\epsilon}$ should be $\boldsymbol{\epsilon} - \mathbf{x}_0$. But since the path is curved (to preserve variance), the direction at $\mathbf{x}_t$ is not the chord — it is the **tangent** of the path, $d\mathbf{x}_t/dt$:

$$
\hat{\mathbf{v}}_t = \frac{d\mathbf{x}_t}{dt} = \frac{d\alpha_t}{dt}\,\hat{\mathbf{x}}_0 + \frac{d\sigma_t}{dt}\,\hat{\boldsymbol{\epsilon}}_t.
$$

Using $\alpha_t = \cos\theta_t,\ \sigma_t = \sin\theta_t$:

$$
\begin{aligned}
\hat{\mathbf{v}}_t
&= \frac{d\theta_t}{dt}\left(-\sin\theta_t\,\hat{\mathbf{x}}_0 + \cos\theta_t\,\hat{\boldsymbol{\epsilon}}_t\right) \\
&= \frac{d\theta_t}{dt}\left(\alpha_t\,\hat{\boldsymbol{\epsilon}}_t - \sigma_t\,\hat{\mathbf{x}}_0\right).
\end{aligned}
$$

Since $d\theta_t/dt$ is a scalar set by the $\beta$-schedule, the *direction* of the velocity is determined solely by $\alpha_t\,\hat{\boldsymbol{\epsilon}}_t - \sigma_t\,\hat{\mathbf{x}}_0$. Training the network to match this velocity is exactly **v-prediction**, proposed by Salimans & Ho in *Progressive Distillation for Fast Sampling of Diffusion Models*. So $v$-prediction is just "predict the tangent of the curved VP path." 

We can now summarize training and sampling in DDIM from this interpolation viewpoint.

**Training.**
1. Draw $\mathbf{x}_0 \sim p_{\text{data}}$ and sample $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$. This pair defines a true path $s$: $\mathbf{x}_t = \alpha_t\,\mathbf{x}_0 + \sigma_t\,\boldsymbol{\epsilon}$, where $t$ is the position along the path and the $\beta$-schedule sets how the $t$ values are distributed along it.
2. Pick a $t$ and obtain the point $\mathbf{x}_t$ on the path.
3. The network looks at $\mathbf{x}_t$ and predicts the noise $\hat{\boldsymbol{\epsilon}}_t$, equivalently guessing a clean endpoint $\hat{\mathbf{x}}_0 = (\mathbf{x}_t - \sigma_t\,\hat{\boldsymbol{\epsilon}}_t)/\alpha_t$. This defines an estimated path $s'$ through $\mathbf{x}_t$ whose clean end is $\hat{\mathbf{x}}_0$.
4. The loss pulls $\hat{\mathbf{x}}_0 \to \mathbf{x}_0$ (equivalently $\hat{\boldsymbol{\epsilon}}_t \to \boldsymbol{\epsilon}$), i.e. it bends the estimated path $s'$ toward the true path $s$.

**Inference (step $t$).**
1. The network predicts $\hat{\boldsymbol{\epsilon}}_t$, giving the estimated path $s'$ and its clean end $\hat{\mathbf{x}}_0$.
2. Move along $s'$ toward the clean end, taking the point at noise level $\tau < t$: $\hat{\mathbf{x}}_\tau = \alpha_\tau\,\hat{\mathbf{x}}_0 + \sigma_\tau\,\hat{\boldsymbol{\epsilon}}_t$. (Deterministic DDIM lands *exactly* on $s'$ via a single analytic step — not a chord approximation in $\mathbf{x}$-space.)
3. Once at $\hat{\mathbf{x}}_\tau$, predict $\hat{\boldsymbol{\epsilon}}_\tau$ again, update the estimated path, and continue.

![diff_to_flow](blog-posts/imgs/DDIM.png)

The error of DDIM comes entirely from the gap between the estimated path $s'$ and the true path $s$. The larger the step, the more $\hat{\boldsymbol{\epsilon}}$ drifts along the way, and the larger this gap becomes.

## Flow Matching

Now flow matching almost writes itself. While reading the previous section, you have probably been asking:

***Why can't I just go in a straight line between $\boldsymbol{\epsilon}$ and $\mathbf{x}_0$?***

Yes, you can — and that is exactly the idea of flow matching. Flow matching replaces the curved VP arc with a **straight linear interpolation**:

$$
\mathbf{x}_t = (1 - t)\,\mathbf{x}_0 + t\,\boldsymbol{\epsilon}, \quad t \in [0, 1].
$$

(Equivalently $\alpha_t = 1 - t,\ \sigma_t = t$. Note this drops the variance-preserving constraint $\alpha_t^2 + \sigma_t^2 = 1$ — that is the price of going straight.)

With a straight line, the velocity is constant along each trajectory and equals the chord direction:

$$
\mathbf{u}_t = \frac{d\mathbf{x}_t}{dt} = \boldsymbol{\epsilon} - \mathbf{x}_0.
$$

So the tangent *is* the chord — exactly the naive direction you wanted all along. Flow matching trains a velocity field $\mathbf{u}_t$ that transports the data distribution to a simpler one (e.g. a Gaussian). Rectified flow uses this straight-line construction.

**Training.**
- Sample $\boldsymbol{\epsilon}$, $\mathbf{x}_0$, and a time step $t$; form $\mathbf{x}_t$ by linear interpolation.
- Given $\mathbf{x}_t$ and $t$, predict the velocity $\hat{\mathbf{u}}_t$ at that point.
- Regress it against the ground-truth velocity $\mathbf{u}_t = \boldsymbol{\epsilon} - \mathbf{x}_0$.

**Inference.**
- Start from $\mathbf{x}_1 \sim \mathcal{N}(0, \mathbf{I})$.
- At each step, predict $\hat{\mathbf{u}}_t$ and integrate the ODE $\dot{\mathbf{x}} = \hat{\mathbf{u}}_t$ toward $t = 0$, e.g. with an Euler step $\mathbf{x}_{t-\Delta} = \mathbf{x}_t - \Delta\,\hat{\mathbf{u}}_t$.
- Because each trained segment is locally straight, the velocity along it is constant and Euler integration is essentially exact — this is why flow matching can sample in few steps.

![diff_to_flow](blog-posts/imgs/diffusion_to_flow_matching.png)

Note that although each *training* segment is a straight line, the *learned* path is generally curved. The reason: at inference the network sees only $\mathbf{x}_t$ — not the pair $(\mathbf{x}_0, \boldsymbol{\epsilon})$ that produced it — so it outputs the **average** velocity over all $(\mathbf{x}_0, \boldsymbol{\epsilon})$ pairs whose straight lines pass through that point. Averaging many straight lines yields a curved marginal trajectory. Rectified flow's *reflow* step re-pairs endpoints to reduce this crossing and straighten the trajectories.

![rectified_flow](blog-posts/imgs/rectified_flow.png)

This illustrates two properties of the flow-matching field:
1. After training, the deterministic trajectories do not cross (by uniqueness of ODE solutions).
2. The marginal paths are generally curved, even though each training target is a straight line.

## So why did something this simple take the field ~2 years?

Geometrically, it really is that simple: **diffusion and flow matching do the same thing** — interpolate between data and noise, predict an endpoint/velocity, and bend the estimated path toward the true one — **differing only in the path shape (curved VP arc vs. straight line) and the prior.** Three caveats keep this honest, and together they explain why the unification was not obvious at the time.

**1. Why assume straight is better than curved?**
A straight line is only "best" under a flat-metric prior. On a globe, the straight line drawn on a flat map is *not* the geodesic. Whether straightening actually helps depends on our prior about the geometry of the data manifold — so "straight vs. curved" is a modeling choice, not a universal truth.

**2. Geometry is not the whole story.**
The 2D arc picture is a *single-trajectory slice* of a high-dimensional process. It shows the skeleton clearly but hides what the SDE/ODE language makes precise: convergence, sampling-error bounds, exact likelihoods, and why the score is even learnable. Geometry tells you *what* moves; the ODE/SDE view tells you *whether, and how well, it works*.

**3. A higher-level lesson: a strong formalism is also a filter.**
The reason this took so long, despite being geometrically intuitive, is that the dominant score-SDE language treated the noise schedule as a *necessity* — forced by the Ornstein–Uhlenbeck process — rather than as a *free knob*. Flow matching's real contribution was decoupling the training objective from the path, which finally made "just swap the path" a sentence one could even say. A powerful formalism grants computational superpowers along its grain, but flattens the degrees of freedom across it. Progress here was less about a stronger tool and more about switching to a language in which the hidden knob became visible again.