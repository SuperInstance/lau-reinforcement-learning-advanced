//! Proximal Policy Optimization (PPO) with clipping.

use crate::core::*;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// PPO agent configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PpoConfig {
    pub clip_epsilon: f64,
    pub gamma: DiscountFactor,
    pub gae_lambda: f64,
    pub policy_lr: f64,
    pub value_lr: f64,
    pub epochs_per_update: usize,
    pub batch_size: usize,
    pub entropy_coeff: f64,
    pub max_grad_norm: f64,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            clip_epsilon: 0.2,
            gamma: 0.99,
            gae_lambda: 0.95,
            policy_lr: 0.0003,
            value_lr: 0.001,
            epochs_per_update: 4,
            batch_size: 64,
            entropy_coeff: 0.01,
            max_grad_norm: 0.5,
        }
    }
}

/// A rollout buffer for PPO.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PpoBuffer {
    pub states: Vec<Vec<f64>>,
    pub actions: Vec<Action>,
    pub rewards: Vec<Reward>,
    pub values: Vec<f64>,
    pub log_probs: Vec<f64>,
    pub dones: Vec<bool>,
}

impl PpoBuffer {
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            actions: Vec::new(),
            rewards: Vec::new(),
            values: Vec::new(),
            log_probs: Vec::new(),
            dones: Vec::new(),
        }
    }

    pub fn push(
        &mut self,
        state: Vec<f64>,
        action: Action,
        reward: Reward,
        value: f64,
        log_prob: f64,
        done: bool,
    ) {
        self.states.push(state);
        self.actions.push(action);
        self.rewards.push(reward);
        self.values.push(value);
        self.log_probs.push(log_prob);
        self.dones.push(done);
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    pub fn clear(&mut self) {
        self.states.clear();
        self.actions.clear();
        self.rewards.clear();
        self.values.clear();
        self.log_probs.clear();
        self.dones.clear();
    }
}

impl Default for PpoBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// PPO actor-critic network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PpoNetwork {
    pub policy: NeuralNetwork,
    pub value_fn: NeuralNetwork,
    pub num_actions: usize,
}

impl PpoNetwork {
    pub fn new(state_size: usize, num_actions: usize, hidden_size: usize) -> Self {
        Self {
            policy: NeuralNetwork::new(&[state_size, hidden_size, num_actions]),
            value_fn: NeuralNetwork::new(&[state_size, hidden_size, 1]),
            num_actions,
        }
    }

    pub fn action_probabilities(&self, state: &[f64]) -> Vec<f64> {
        let logits = self.policy.forward_linear(state);
        softmax(&logits)
    }

    pub fn value(&self, state: &[f64]) -> f64 {
        self.value_fn.forward_linear(state)[0]
    }

    pub fn log_prob(&self, state: &[f64], action: Action) -> f64 {
        let probs = self.action_probabilities(state);
        probs[action].ln().max(-10.0)
    }

    pub fn select_action(&self, state: &[f64]) -> (Action, f64, f64) {
        let probs = self.action_probabilities(state);
        let value = self.value(state);

        use rand::Rng;
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen();
        let mut cumsum = 0.0;
        let mut action = probs.len() - 1;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                action = i;
                break;
            }
        }

        let log_prob = probs[action].ln().max(-10.0);
        (action, value, log_prob)
    }
}

/// PPO agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PpoAgent<S: Clone + Debug> {
    pub network: PpoNetwork,
    pub config: PpoConfig,
    pub buffer: PpoBuffer,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: Clone + Debug + Into<Vec<f64>>> PpoAgent<S> {
    pub fn new(state_size: usize, num_actions: usize, hidden_size: usize, config: PpoConfig) -> Self {
        let network = PpoNetwork::new(state_size, num_actions, hidden_size);
        Self {
            network,
            config,
            buffer: PpoBuffer::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Collect experience for `n_steps`.
    pub fn collect_rollout<E: Environment<State = S>>(&mut self, env: &mut E, n_steps: usize) -> Reward {
        let mut state = env.reset();
        let mut total_reward = 0.0;

        for _ in 0..n_steps {
            let state_vec: Vec<f64> = state.clone().into();
            let (action, value, log_prob) = self.network.select_action(&state_vec);
            let (next_state, reward, done) = env.step(action);
            total_reward += reward;

            self.buffer.push(state_vec, action, reward, value, log_prob, done);
            state = next_state;
            if done {
                state = env.reset();
            }
        }

        total_reward
    }

    /// PPO clipped objective update.
    pub fn update(&mut self) -> (f64, f64) {
        if self.buffer.is_empty() {
            return (0.0, 0.0);
        }

        // Compute advantages using GAE
        let advantages = compute_gae(
            &self.buffer.rewards,
            &self.buffer.values,
            self.config.gamma,
            self.config.gae_lambda,
        );

        // Normalize advantages
        let adv_mean: f64 = advantages.iter().sum::<f64>() / advantages.len() as f64;
        let adv_std: f64 = {
            let variance = advantages
                .iter()
                .map(|a| (a - adv_mean).powi(2))
                .sum::<f64>()
                / advantages.len() as f64;
            (variance + 1e-8).sqrt()
        };
        let normalized_advantages: Vec<f64> = advantages
            .iter()
            .map(|a| (a - adv_mean) / adv_std)
            .collect();

        // Compute returns
        let returns: Vec<f64> = advantages
            .iter()
            .zip(self.buffer.values.iter())
            .map(|(adv, val)| adv + val)
            .collect();

        let mut total_policy_loss = 0.0;
        let mut total_value_loss = 0.0;

        for _ in 0..self.config.epochs_per_update {
            for i in 0..self.buffer.len() {
                let state = &self.buffer.states[i];
                let action = self.buffer.actions[i];
                let advantage = normalized_advantages[i];
                let return_val = returns[i];
                let old_log_prob = self.buffer.log_probs[i];

                // Current policy log prob
                let new_log_prob = self.network.log_prob(state, action);
                let ratio = (new_log_prob - old_log_prob).exp();

                // Clipped surrogate objective
                let clipped_ratio = ratio.clamp(
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                );
                let policy_loss = -(ratio * advantage)
                    .min(clipped_ratio * advantage);

                // Value loss
                let value_pred = self.network.value(state);
                let value_loss = (return_val - value_pred).powi(2);

                total_policy_loss += policy_loss;
                total_value_loss += value_loss;

                // Gradient step on policy
                let probs = self.network.action_probabilities(state);
                for layer in &mut self.network.policy.layers {
                    for (j, row) in layer.weights.iter_mut().enumerate() {
                        for w in row.iter_mut() {
                            if j == action {
                                *w += self.config.policy_lr * advantage * ratio.min(clipped_ratio) * (1.0 - probs[action]);
                            } else if j < probs.len() {
                                *w -= self.config.policy_lr * advantage * ratio.min(clipped_ratio) * probs[j];
                            }
                        }
                    }
                }

                // Gradient step on value
                let value_error = return_val - value_pred;
                for layer in &mut self.network.value_fn.layers {
                    for row in &mut layer.weights {
                        for w in row.iter_mut() {
                            *w += self.config.value_lr * value_error * 0.01;
                        }
                    }
                    if let Some(b) = layer.biases.first_mut() {
                        *b += self.config.value_lr * value_error * 0.01;
                    }
                }
            }
        }

        let n = self.buffer.len() as f64 * self.config.epochs_per_update as f64;
        self.buffer.clear();

        (total_policy_loss / n, total_value_loss / n)
    }

    /// Full training loop.
    pub fn train<E: Environment<State = S>>(
        &mut self,
        env: &mut E,
        num_iterations: usize,
        steps_per_update: usize,
    ) -> Vec<(Reward, f64, f64)> {
        let mut results = Vec::new();
        for _ in 0..num_iterations {
            let reward = self.collect_rollout(env, steps_per_update);
            let (policy_loss, value_loss) = self.update();
            results.push((reward, policy_loss, value_loss));
        }
        results
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
    fn test_ppo_config_default() {
        let config = PpoConfig::default();
        assert!((config.clip_epsilon - 0.2).abs() < 1e-6);
        assert!((config.gae_lambda - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_ppo_buffer() {
        let mut buf = PpoBuffer::new();
        assert!(buf.is_empty());
        buf.push(vec![1.0, 0.0], 0, 1.0, 0.5, -0.3, false);
        assert_eq!(buf.len(), 1);
        buf.clear();
        assert!(buf.is_empty());
    }

    #[test]
    fn test_ppo_network() {
        let net = PpoNetwork::new(4, 3, 8);
        let probs = net.action_probabilities(&[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(probs.len(), 3);
        let val = net.value(&[1.0, 0.0, 0.0, 0.0]);
        assert!(val.is_finite());
        let (action, _value, _log_prob) = net.select_action(&[0.0, 1.0, 0.0, 0.0]);
        assert!(action < 3);
    }

    #[test]
    fn test_ppo_agent_train() {
        let mut env = VecGridWorld { inner: GridWorld::new(3, 3) };
        let config = PpoConfig {
            epochs_per_update: 2,
            batch_size: 8,
            ..Default::default()
        };
        let mut agent: PpoAgent<Vec<f64>> = PpoAgent::new(2, 4, 8, config);
        let results = agent.train(&mut env, 3, 10);
        assert_eq!(results.len(), 3);
        for (reward, policy_loss, value_loss) in &results {
            assert!(reward.is_finite());
            assert!(policy_loss.is_finite());
            assert!(value_loss.is_finite());
        }
    }

    #[test]
    fn test_ppo_clipping() {
        let config = PpoConfig::default();
        let epsilon = config.clip_epsilon;
        let ratio = 1.5_f64;
        let clipped = ratio.clamp(1.0 - epsilon, 1.0 + epsilon);
        assert!((clipped - 1.2).abs() < 1e-6);
    }
}
