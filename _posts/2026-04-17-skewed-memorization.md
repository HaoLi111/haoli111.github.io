---
layout: default
title: "LLMs as dynamical systems — a decomposition story"
date: 2026-04-17
permalink: /blog/llm-dynamics-decomposition/
---
# LLM behaviors and privacy— a decomposition story

*This post is the prose companion to the playlist [Quantification and decomposition of LLM and Agents — research](https://www.youtube.com/playlist?list=PLJJwa67T0Zu5P8lx6A6B1YiuMlQgCazaX) and to our paper [Skewed Memorization in Large Language Models: Quantification and Decomposition](https://arxiv.org/abs/2502.01187). Memorization is the **observable**; the real subject is the **dynamics**.*

---

## Reframing: an LLM is a state transition

It's tempting to describe an autoregressive model as "a probability distribution over next tokens." That is true and not useful. A more honest description: an LLM is a **state-transition system**. At each step there is a state (the KV cache, the residual stream, whatever compact object you want to call it), a transition operator (one forward pass), and an output (a token sampled from the induced next-token distribution). Training reshapes the operator. Inference iterates it.

Once you say it that way, most of the classical machinery comes back online:

- **Information theory** — the entropy of the next-token distribution, the mutual information between past and future tokens, the KL divergence between the learned transition and the data-generating one.
- **Queueing and renewal theory** — the run-lengths of "things happening," waiting times between events, geometric and heavy-tailed sojourn distributions.
- **Dynamical systems** — fixed points, attractors, basins, Lyapunov-style divergence of nearby trajectories, phase transitions.
- **Classical NLP** — suffix trees, smoothing, back-off, the Bayes-optimal term-wise classifier that n-gram models were implicitly trying to approximate before we had transformers.

Our paper is, on its surface, about memorization in supervised fine-tuning. Under the surface it is a worked example of all four — how decomposing the *transition* (not just the *output*) reveals structure that a loss curve cannot see.

---

## The observable: prefix match length

Pick a training sample $s$, cut it into a prefix $r_\text{pre}$ and a suffix $r_\text{ref}$, run the model, count how many tokens of its output match the reference before divergence:

$$
n_\text{pre} = \max\{k \mid s_{1:k} = r_{\text{ref},1:k}\}.
$$

This is a renewal-theory object. It is the *run length* of a "state agrees with the training trajectory" event. Across the dataset, $N_\text{pre}$ is a random variable. Its distribution — *especially its upper tail* — is the empirical signature of the model's dynamics on this dataset.

Why the tail matters, in renewal terms: averages describe typical sojourns. The tail describes the rare long runs where the model's trajectory locks onto a training trajectory and stays on it. Those are the events that dominate privacy risk, copyright exposure, and the weird cases where a model's dynamics get trapped in a basin carved by a specific training sequence.

Small-sample estimators miss these on purpose. With a training set of size $M$ and a sample of $z$, the probability of missing the top-$k$ most-memorized samples is

$$
\mathbb{P}(\max(n_1, \ldots, n_z) < N_{(-k)}) = \frac{\binom{M-k}{z}}{\binom{M}{z}},
$$

which for $z \ll M$ factors into $\prod_{i=1}^{z}\mathbb{P}(N_{\text{pre}} < n)$. This is the same structure as a record-value problem. If you want tail behavior, you don't sample-average — you resample non-parametrically and estimate the distribution.

---

## Decomposition 1: factor the transition, not the sequence

The first decomposition is the one everyone knows but rarely uses carefully. Because the model is autoregressive, the probability of a consecutive match up to length $n_\text{pre}$ factors over token positions:

$$
\mathbb{P}(N_\text{pre} = n_\text{pre}) = \left(\prod_j p_j^o\right)\bigl(1 - p_{n[\text{pre}]}^o\bigr),
$$

with $p_j^o$ the probability that token $j$ is "correctly recalled." If we assume approximate independence across positions — a testable assumption via mutual information $MI(C_{J_\text{pre}}, C_j) = H(j) - H(j \mid J_\text{pre})$ — then $N_\text{pre}$ is geometric: $N_\text{pre} \sim \text{Geom}(1-p)$.

**This is the classical queueing baseline.** Geometric run-lengths are what you get from a memoryless success/failure chain. Anywhere reality deviates from geometric is where the dynamics have *structure* beyond a homogeneous Markov step — long-range dependence, non-stationarity, position-dependent failure rates. A richer model where $p_j = \alpha j + p_0$ gives

$$
\mathbb{P}(N_\text{pre} = n_\text{pre}) = p_0^{n_\text{pre}} \frac{\Gamma(n_\text{pre} + 1 + \alpha/p_0)}{\Gamma(1 + \alpha/p_0)}(1 - p_{n_\text{pre}+1}),
$$

which is the kind of non-stationary hazard you see in survival analysis and in queues with time-varying service rates. The distribution of $n_\text{pre}$ becomes a diagnostic for *how the transition changes as the trajectory lengthens*.

---

## Decomposition 2: three operators that could be generating the trajectory

The more useful decomposition is about *which operator* the LLM is approximating. There are three natural candidates:

1. **Bayes Optimal Classifier** $M_b$ over suffixes — chooses $\arg\max_{S^n} \pi(S^n = r^n \mid r_\text{ref})$. This is the operator a perfect memorizer would run at the sequence level.
2. **Term-wise Bayes Optimal Classifier** $M_t$ — chooses $\arg\max_{s[j]} \pi(s[j] = r[j] \mid r_\text{ref}, \ldots, j-1)$. This is n-gram-style local optimality; it's exactly the operator classical NLP smoothing was reaching for.
3. **Actual LLM dynamics** $M_C$ at training checkpoint $C$.

The paper's cleanest theoretical result (Theorem 2.6): greedy LLM decoding approximates $M_t$ if and only if $\arg\max_{\pi[M]} \pi_M(s[j]) = \arg\max_\pi \pi(r[j])$. This is **strictly weaker** than full distributional convergence $\pi_M = \pi$.

The dynamical reading: **the transition operator can match the data's term-wise argmax — reliably picking the right next token — without matching the full distribution.** Generalization (distributional match) and memorization (trajectory lock-on) are not on the same axis. They are separate questions about the operator. One is about agreement on modes; the other is about agreement on mass.

This also connects back to old NLP. An n-gram model with good smoothing approximates $M_t$ under its own factorization assumption. A transformer with a much longer context is doing a richer version of the same thing. What changes is the effective context window — the number of past states that meaningfully influence the transition. The *form* of the decomposition is the same.

---

## Decomposition 3: the embedding gap and basin geometry

Define

$$
\Delta S = S_\text{full} - S_\text{input},
$$

with $S_\text{input}$ the mean cosine similarity between training *prefixes* and $S_\text{full}$ the mean cosine similarity between full training sequences. A large $\Delta S$ means suffixes vary more than prefixes — similar inputs scatter to diverse outputs.

Read this through the dynamical lens: $\Delta S$ is a coarse measure of how the operator *disperses* nearby starting states. High $\Delta S$ means similar prefixes sit in basins whose trajectories fan out. Low $\Delta S$ means the operator is contractive on this neighborhood — nearby prefixes converge to nearly the same suffix.

**Contractive basins are exactly where memorization concentrates.** Remark 2.7 in the paper is the mathematical statement of this: samples with highly similar prefixes but diverse full sequences memorize less. The memorization tail is not a property of individual samples; it is a property of the *local Lyapunov behavior* of the transition operator over the embedding manifold.

Practically, this means you cannot audit a dataset row by row. The risk of a sample is determined by its neighborhood under the learned operator, and that neighborhood only exists after training.

---

## What the experiments actually show

We fine-tuned Llama-3.1-8B-Instruct with LoRA on two datasets: **Lavita-Medical-QA** (domain-specific, 9,723 QA pairs reformatted to single-answer) and **GPTeacher-General-Instruct** (open-domain, same size). Inference ran through vLLM at temperature 0.

Three observations, read as dynamical statements:

**1. Worst-case run-lengths grow before loss converges.** On Lavita, max $n_\text{pre}$ exceeds 10 by epoch 10 while training loss is still high, and drifts to ~50 by epoch 100 after the loss has flattened. The dynamics keep evolving after the bulk energy has settled — a classic slow-mode phenomenon. Loss is the order parameter for the mean; it is not the order parameter for the tail.

**2. Dataset composition changes the operator, not just the scores.** Mixing 200 GPTeacher samples into Lavita pushes max $n_\text{pre}$ above 40 at epoch 10 and over 50 by epoch 100 — while Lavita-alone caps around 16. The *same* 200 overlap samples have statistically different memorization (signed-rank test) in the two regimes. The neighborhood changed; the operator's contractivity around those samples changed; the observable changed.

**3. Memorization is a neighborhood property.** The risk of a sample is a function of the operator restricted to its basin, not of the sample in isolation.

We cross-check against ROUGE and Levenshtein distance so the metric is comparable to established evaluations.

---

## Why this is a step toward understanding LLM dynamics

What surprised us at the time — and this is the part I keep coming back to — is how much of the picture is *already* described by tools that predate transformers by decades. Geometric run-lengths, term-wise Bayes-optimal classifiers, contractive maps, entropy decompositions, non-stationary hazard rates, record-value statistics. The reason these tools apply is not that LLMs are "secretly simple." It is that **iterated state transitions are a very old object**, and LLMs are a particular, very expressive instance of it. The math doesn't care which function class realizes the transition.

The decomposition matters because it separates questions that are usually conflated:

- *What does the operator look like locally?* (term-wise classifier, $M_t$)
- *What does the trajectory look like globally?* (sequence-level classifier, $M_b$)
- *How does the operator contract or disperse neighborhoods?* (embedding similarity gap, $\Delta S$)
- *How do run-lengths distribute across the dataset?* (the prefix-match distribution, its tail)

Each of these is a different slice of the dynamics. Each has a classical analogue. Putting them together gives a picture that a single scalar like "loss" or "perplexity" cannot give.

This is also where the work connects to the other threads in my research — agentic systems where the transition is much richer than a single forward pass, interpretability that asks *what* the transition is computing,  Memorization is one observable; the framework is the point.

---

## Walkthrough by video

The playlist — **"Quantification and decomposition of LLM and Agents — research"** — walks the same logic at whiteboard pace. Each video is one slice of the decomposition, and below I annotate each with the dynamical-systems reading, not just the memorization one. First, memorization is like: usually, there is none, but sometimes there can be a lot--does that recalls what N. Taleb says about risk management?

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; margin-bottom: 1.5em;">
  <iframe src="https://www.youtube.com/embed/videoseries?list=PLJJwa67T0Zu5P8lx6A6B1YiuMlQgCazaX" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" frameborder="0" allowfullscreen allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"></iframe>
</div>

Direct link: [YouTube playlist](https://www.youtube.com/playlist?list=PLJJwa67T0Zu5P8lx6A6B1YiuMlQgCazaX).

---

### 1. Overview

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; margin-bottom: 1em;">
  <iframe src="https://www.youtube.com/embed/6m36Yf7Xp8I?list=PLJJwa67T0Zu5P8lx6A6B1YiuMlQgCazaX" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" frameborder="0" allowfullscreen allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"></iframe>
</div>

Sets up the prefix continuation framework and the tension between generalization and memorization. Read as dynamics: the question is not "does the model learn the distribution" but "what does the iterated transition operator look like, and when does its trajectory lock onto a training trajectory." The rest of the playlist is decompositions of that one question.

### 2. Average is Not All You Need in Accessing LLM Risks

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; margin-bottom: 1em;">
  <iframe src="https://www.youtube.com/embed/XWp8v7_207Y?list=PLJJwa67T0Zu5P8lx6A6B1YiuMlQgCazaX" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" frameborder="0" allowfullscreen allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"></iframe>
</div>

The core methodological complaint: mean-based metrics hide the tail. A low mean memorization rate is compatible with perfect verbatim recall on a small, sensitive subset. In renewal-theory terms, you are being told the average sojourn length of a Markov chain while the distribution of sojourns is heavy-tailed — the average is not the event that hurts you. The video motivates non-parametric, distributional evaluation and the record-value statistics from the paper.

### 3. Decomposing Sequential Memorization

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; margin-bottom: 1em;">
  <iframe src="https://www.youtube.com/embed/mY8S2z70m_M?list=PLJJwa67T0Zu5P8lx6A6B1YiuMlQgCazaX" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" frameborder="0" allowfullscreen allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"></iframe>
</div>

The first decomposition: factor the probability of consecutive match into token-level factors and analyze how the factors evolve over sequence position. This is where the geometric-run-length baseline comes from, and where non-stationary hazard rates ($p_j = \alpha j + p_0$) show up as deviations from that baseline. Inter-sample similarity and local data density appear here as the things that bend the hazard rate. But why? this dynamics creates skewness!

### 4. Bayesian Optimal and Term-Wise Greedy as Two Limits to Sequential Memorization

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; margin-bottom: 1em;">
  <iframe src="https://www.youtube.com/embed/R_S8Zp_m9_8?list=PLJJwa67T0Zu5P8lx6A6B1YiuMlQgCazaX" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" frameborder="0" allowfullscreen allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"></iframe>
</div>

The second decomposition, and the one I think is most worth your time. Two theoretical operators bounds its behavior (well, I agree, loosely...) — the sequence-level Bayes-optimal classifier $M_b$ and the term-wise Bayes-optimal classifier $M_t$ — bracket the behavior of real LLM inference. Generalization lives near one limit, memorization near the other, and real models live on the interior. Theorem 2.6 is the condition under which greedy decoding approximates $M_t$ without matching the full distribution — i.e., the model can be locally right without being globally calibrated. This is also the bridge back to classical NLP: $M_t$ is what n-gram models with smoothing were implicitly targeting.

### 5. Proof: Guessing is Not Better (Privacy in Large Language Models)

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; margin-bottom: 1em;">
  <iframe src="https://www.youtube.com/embed/Z_vU_yG_p_E?list=PLJJwa67T0Zu5P8lx6A6B1YiuMlQgCazaX" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" frameborder="0" allowfullscreen allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"></iframe>
</div>

A proof that verbatim reproduction cannot be explained as lucky sampling — it is a deterministic consequence of the operator's geometry, not a statistical coincidence. Dynamically, this says long prefix-match runs are not random concurrences of Bernoulli trials; they are signatures of the trajectory being pulled into a narrow basin. This is why extreme memorization keeps increasing even while the loss improves: the basin deepens.

### 6. Memorization vs. String Algorithm vs. NLP Metrics — Simple Math

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; margin-bottom: 1em;">
  <iframe src="https://www.youtube.com/embed/fXvW_y_Xv_M?list=PLJJwa67T0Zu5P8lx6A6B1YiuMlQgCazaX" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" frameborder="0" allowfullscreen allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"></iframe>
</div>

Bridges prefix match length with classical string algorithms (Levenshtein, longest common subsequence) and NLP similarity metrics (ROUGE). These metrics weren't designed for tail-diagnosis, but they can be adapted — the video shows how. The broader point: run-length-based measurements on trajectories are an old family of tools, and most of the adaptation is in choosing which statistic of the distribution you report.

### 7. Reviewing Cumulative Product of Arithmetic Sequence and One Special Case of Memorization

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; margin-bottom: 1em;">
  <iframe src="https://www.youtube.com/embed/E_vU_yG_p_E?list=PLJJwa67T0Zu5P8lx6A6B1YiuMlQgCazaX" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" frameborder="0" allowfullscreen allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"></iframe>
</div>

Works through the cumulative-product form

$$
\mathbb{P}(N_\text{pre} = n_\text{pre}) = p_0^{n_\text{pre}} \frac{\Gamma(n_\text{pre} + 1 + \alpha/p_0)}{\Gamma(1 + \alpha/p_0)}(1 - p_{n_\text{pre}+1})
$$

that arises when the per-token memorization probability is linear in position. This is the analytic closed-form for one specific non-stationary hazard — a toy case where the decomposition is fully tractable. In dynamical language, it is the simplest operator with position-dependent contraction, and it already reproduces the skewed run-length shape the experiments find.

---

For citations, see the paper's bibliography: [arXiv:2502.01187](https://arxiv.org/abs/2502.01187).

*Cheers,*
*Hao*
