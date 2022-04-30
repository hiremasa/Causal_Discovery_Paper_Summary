# Causal Inference by Stochastic Complexity
The algorithmic Markov condition states that the most likely causal  direction between two random variables `X` and `Y` can be identified  as that direction with the lowest Kolmogorov complexity. Due to the halting problem, however, this notion is not computable.

We hence propose to do causal inference by stochastic complexity. That is, we propose to approximate Kolmogorov complexity via the Minimum Description Length (MDL) principle, using a score that is mini-max optimal with regard to the model class under consideration. This means that even in an adversarial setting, such as when the true distribution is not in this class, we still obtain the optimal encoding for the data relative to the class.

We instantiate this framework, which we call CISC, for pairs of univariate discrete variables, using the class of multinomial distributions.
Experiments show that CISC is highly accurate on synthetic, benchmark, as well as real-world data, outperforming the state of the art by a margin, and scales extremely well with regard to sample and domain sizes.

