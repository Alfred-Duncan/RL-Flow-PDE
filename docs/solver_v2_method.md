# Solver V2 Method

We formulate PDE solving as a sequential reinforcement-learning problem and train a residual-conditioned deterministic neural operator policy to directly generate iterative solution-field updates under governing-equation constraints.

The V2 actor is a full-field FNO-style neural operator. It receives the current solution, residual field, source, IC field, BC field, PDE scalars, and step fraction, and outputs a bounded correction field. TD3+BC fine-tunes a supervised warm-start actor using accuracy-improvement rewards, behavior-cloning regularization, and scale-normalized physics regularization. Full-solver checkpoints are selected on the validation split before final test evaluation.
