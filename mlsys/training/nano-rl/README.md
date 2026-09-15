# RL-Training

## SFT

- Supervised Fine Tuning
- Each training sample is a `(prompt, desired completion)` pair
- Define the loss as negative log-probabilities of desired completion tokens
  - i.e. mask the loss so only completion tokens contribute
- Update model weights with standard training

## GRPO

- Group Relative Policy Optimization
- Each training sample provides a prompt and an expected answer used by the reward function, not a desired completion
- The current policy generates multiple completions (i.e. group) for the same prompt
- Score each completion, then compare its reward with the other completions (i.e. relative) in the group
- Increase the likelihood of better-than-group completions and decrease the likelihood of worse-than-group completions
  - Penalize the policy when it drifts too far from the frozen reference model
- Update the policy and repeat with new rollouts

## PPO

- Proximal Policy Optimization
- Each training sample provides a prompt and an expected answer used by the reward function, not a desired completion
- The current policy generates a completion and receives a reward for the result
- The critic estimates the expected future reward at each completion-token state
  - The critic uses a transformer plus a linear head to predict one scalar value per token state
- Compare the outcome with the critic's expectations, then propagate that feedback backward to calculate per-token advantages
- Use the advantages to train the policy, while training the critic to better predict future rewards

## DPO

- Direct Preference Optimization
- Each training sample is a `(prompt, chosen completion, rejected completion)` preference triple
  - Both completions come from the dataset, so training does not need rollouts, a reward function, or a critic
- Compare how likely the current policy and frozen reference model consider each completion
- The loss decreases as the policy improves its preference for the chosen completion over the rejected completion
- Update only the policy so it shifts more toward the chosen completion; keep the reference model frozen
