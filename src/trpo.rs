//! Trust Region Policy Optimization (TRPO) via conjugate gradient.

use crate::core::*;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// TRPO configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrpoConfig {
    pub max_kl: f64,
    pub gamma: DiscountFactor,
    pub gae_lambda: f64,
    pub cg_iters: usize,
    pub damping: f64,
    pub backtrack_coeff: f64,
    pub max_backtracks: usize,
    pub value_lr: f64,
}

impl Default for TrpoConfig {
    fn default() -> Self {
        Self {
            max_kl: 0.01,
            gamma: 0.99,
            gae_lambda: 0.95,
            cg_iters: 10,
            damping: 0.1,
            backtrack_coeff: 0.5,
            max_backtracks: 10,
            value_lr: 0.001,
        }
    }
}

/// TRPO agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrpoAgent<S: Clone + Debug> {
    pub policy: NeuralNetwork,
    pub value_fn: NeuralNetwork,
    pub config: TrpoConfig,
    pub num_actions: usize,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: Clone + Debug + Into<Vec<f64>>> TrpoAgent<S> {
    pub fn new(
        state_size: usize,
        num_actions: usize,
        hidden_size: usize,
        config: TrpoConfig,
    ) -> Self {
        Self {
            policy: NeuralNetwork::new(&[state_size, hidden_size, num_actions]),
            value_fn: NeuralNetwork::new(&[state_size, hidden_size, 1]),
            num_actions,
            config,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn action_probabilities(&self, state: &[f64]) -> Vec<f64> {
        let logits = self.policy.forward_linear(state);
        softmax(&logits)
    }

    pub fn select_action(&self, state: &[f64]) -> Action {
        let probs = self.action_probabilities(state);
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen();
        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return i;
            }
        }
        probs.len() - 1
    }

    pub fn value(&self, state: &[f64]) -> f64 {
        self.value_fn.forward_linear(state)[0]
    }

    /// Conjugate gradient solver for Hessian-vector product (approximated).
    fn conjugate_gradient(&self, gradient: &[f64], n_params: usize) -> Vec<f64> {
        let mut x = vec![0.0; n_params];
        let mut r = gradient.to_vec();
        let mut p = r.clone();
        let mut rs_old: f64 = r.iter().map(|ri| ri * ri).sum();

        let dim = n_params.min(gradient.len());
        for _ in 0..self.config.cg_iters.min(dim) {
            let mut ap = vec![0.0; dim];
            for i in 0..dim {
                ap[i] = p[i] * (1.0 + self.config.damping);
            }

            let p_ap: f64 = p.iter().zip(ap.iter()).map(|(pi, api)| pi * api).sum();
            if p_ap.abs() < 1e-10 {
                break;
            }

            let alpha = rs_old / p_ap;
            for i in 0..dim {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }

            let rs_new: f64 = r.iter().map(|ri| ri * ri).sum();
            if rs_new < 1e-10 {
                break;
            }

            let beta = rs_new / rs_old;
            for i in 0..dim {
                p[i] = r[i] + beta * p[i];
            }
            rs_old = rs_new;
        }

        x
    }

    /// KL divergence between two policy distributions.
    fn kl_divergence(&self, old_probs: &[f64], new_probs: &[f64]) -> f64 {
        old_probs
            .iter()
            .zip(new_probs.iter())
            .map(|(p, q)| {
                if *p > 1e-10 && *q > 1e-10 {
                    p * (p / q).ln()
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Collect rollout.
    fn collect_rollout<E: Environment<State = S>>(
        &self,
        env: &mut E,
        n_steps: usize,
    ) -> (Vec<Vec<f64>>, Vec<Action>, Vec<Reward>, Vec<f64>, Vec<Vec<f64>>) {
        let mut states = Vec::new();
        let mut actions = Vec::new();
        let mut rewards = Vec::new();
        let mut values = Vec::new();
        let mut old_probs = Vec::new();
        let mut state = env.reset();

        for _ in 0..n_steps {
            let state_vec: Vec<f64> = state.clone().into();
            let action = self.select_action(&state_vec);
            let value = self.value(&state_vec);
            let probs = self.action_probabilities(&state_vec);

            let (next_state, reward, done) = env.step(action);

            states.push(state_vec);
            actions.push(action);
            rewards.push(reward);
            values.push(value);
            old_probs.push(probs);

            state = next_state;
            if done {
                state = env.reset();
            }
        }

        (states, actions, rewards, values, old_probs)
    }

    /// TRPO update step.
    pub fn update<E: Environment<State = S>>(
        &mut self,
        env: &mut E,
        n_steps: usize,
    ) -> (Reward, f64, f64) {
        let (states, actions, rewards, values, old_probs) =
            self.collect_rollout(env, n_steps);

        let total_reward: Reward = rewards.iter().sum();

        let advantages = compute_gae(&rewards, &values, self.config.gamma, self.config.gae_lambda);
        let adv_mean: f64 = advantages.iter().sum::<f64>() / advantages.len().max(1) as f64;
        let adv_std: f64 = {
            let var = advantages.iter().map(|a| (a - adv_mean).powi(2)).sum::<f64>()
                / advantages.len().max(1) as f64;
            (var + 1e-8).sqrt()
        };
        let norm_advs: Vec<f64> = advantages.iter().map(|a| (a - adv_mean) / adv_std).collect();

        // Compute policy gradient
        let n_params = self.policy.num_parameters();
        let mut gradient = vec![0.0; n_params];
        for (i, (state, &action)) in states.iter().zip(actions.iter()).enumerate() {
            let probs = self.action_probabilities(state);
            let grad_factor = norm_advs[i];
            let mut idx = 0;
            for layer in &self.policy.layers {
                for (j, row) in layer.weights.iter().enumerate() {
                    for _ in row {
                        if idx < n_params {
                            if j == action {
                                gradient[idx] += grad_factor * (1.0 - probs[action]);
                            } else if j < probs.len() {
                                gradient[idx] -= grad_factor * probs[j];
                            }
                        }
                        idx += 1;
                    }
                }
                for _ in &layer.biases {
                    if idx < n_params {
                        gradient[idx] *= 0.5;
                    }
                    idx += 1;
                }
            }
        }

        // Conjugate gradient to compute step direction
        let step_dir = self.conjugate_gradient(&gradient, n_params);

        // Compute step size
        let step_sq: f64 = step_dir.iter().map(|s| s * s).sum();
        let max_step = (self.config.max_kl / (step_sq + 1e-8).sqrt()).min(1.0);

        // Apply update with backtracking line search
        let old_policy = self.policy.clone();
        let mut kl_div = 0.0;
        let mut accepted = false;

        for bt in 0..self.config.max_backtracks {
            let scale = max_step * self.config.backtrack_coeff.powi(bt as i32);

            // Apply step to policy parameters
            let mut idx = 0;
            for layer in &mut self.policy.layers {
                for row in &mut layer.weights {
                    for w in row.iter_mut() {
                        if idx < step_dir.len() {
                            *w += scale * step_dir[idx];
                        }
                        idx += 1;
                    }
                }
                for b in layer.biases.iter_mut() {
                    if idx < step_dir.len() {
                        *b += scale * step_dir[idx] * 0.1;
                    }
                    idx += 1;
                }
            }

            // Check KL constraint
            let mut total_kl = 0.0;
            for (state, old_p) in states.iter().zip(old_probs.iter()) {
                let new_p = self.action_probabilities(state);
                total_kl += self.kl_divergence(old_p, &new_p);
            }
            total_kl /= states.len().max(1) as f64;

            if total_kl <= self.config.max_kl {
                kl_div = total_kl;
                accepted = true;
                break;
            } else {
                self.policy = old_policy.clone();
            }
        }

        if !accepted {
            self.policy = old_policy;
        }

        // Update value function
        let returns = compute_returns(&rewards, self.config.gamma);
        let mut value_loss = 0.0;
        for (state, &ret) in states.iter().zip(returns.iter()) {
            let pred = self.value(state);
            let error = ret - pred;
            value_loss += error.powi(2);

            for layer in &mut self.value_fn.layers {
                for row in &mut layer.weights {
                    for w in row.iter_mut() {
                        *w += self.config.value_lr * error * 0.01;
                    }
                }
                if let Some(b) = layer.biases.first_mut() {
                    *b += self.config.value_lr * error * 0.01;
                }
            }
        }
        value_loss /= states.len().max(1) as f64;

        (total_reward, kl_div, value_loss)
    }

    /// Full training loop.
    pub fn train<E: Environment<State = S>>(
        &mut self,
        env: &mut E,
        num_iterations: usize,
        steps_per_update: usize,
    ) -> Vec<(Reward, f64, f64)> {
        (0..num_iterations)
            .map(|_| self.update(env, steps_per_update))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gv(x: usize, y: usize) -> Vec<f64> { vec![x as f64, y as f64] }

    #[derive(Debug, Clone)]
    struct VecGridWorld { inner: GridWorld }
    impl Environment for VecGridWorld {
        type State = Vec<f64>;
        fn reset(&mut self) -> Vec<f64> { let s = self.inner.reset(); gv(s.0, s.1) }
        fn step(&mut self, action: Action) -> (Vec<f64>, Reward, bool) {
            let (s, r, d) = self.inner.step(action); (gv(s.0, s.1), r, d)
        }
        fn num_actions(&self) -> usize { self.inner.num_actions() }
    }

    #[test]
    fn test_trpo_config() {
        let config = TrpoConfig::default();
        assert!((config.max_kl - 0.01).abs() < 1e-6);
        assert_eq!(config.cg_iters, 10);
    }

    #[test]
    fn test_trpo_agent_select() {
        let agent: TrpoAgent<Vec<f64>> = TrpoAgent::new(2, 4, 8, TrpoConfig::default());
        let action = agent.select_action(&[0.0, 1.0]);
        assert!(action < 4);
    }

    #[test]
    fn test_trpo_conjugate_gradient() {
        let agent: TrpoAgent<Vec<f64>> = TrpoAgent::new(2, 4, 8, TrpoConfig::default());
        let gradient = vec![1.0, 0.5, -0.3, 0.2];
        let result = agent.conjugate_gradient(&gradient, 4);
        assert_eq!(result.len(), 4);
        let norm: f64 = result.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm > 0.0);
    }

    #[test]
    fn test_trpo_kl_divergence() {
        let agent: TrpoAgent<Vec<f64>> = TrpoAgent::new(2, 4, 8, TrpoConfig::default());
        let kl = agent.kl_divergence(&[0.25, 0.25, 0.25, 0.25], &[0.25, 0.25, 0.25, 0.25]);
        assert!(kl.abs() < 1e-6);
        let kl2 = agent.kl_divergence(&[0.5, 0.5, 0.0, 0.0], &[0.25, 0.25, 0.25, 0.25]);
        assert!(kl2 > 0.0);
    }

    #[test]
    fn test_trpo_train() {
        let mut env = VecGridWorld { inner: GridWorld::new(3, 3) };
        let mut agent: TrpoAgent<Vec<f64>> = TrpoAgent::new(2, 4, 8, TrpoConfig {
            cg_iters: 3,
            max_backtracks: 3,
            ..Default::default()
        });
        let results = agent.train(&mut env, 2, 10);
        assert_eq!(results.len(), 2);
        for (reward, kl, vloss) in &results {
            assert!(reward.is_finite());
            assert!(kl.is_finite());
            assert!(vloss.is_finite());
        }
    }
}
