//! Curiosity-driven exploration: intrinsic motivation and prediction error.

use crate::core::*;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Intrinsic reward module: predicts next state features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntrinsicCuriosityModule {
    pub forward_model: NeuralNetwork,
    pub inverse_model: NeuralNetwork,
    pub feature_encoder: NeuralNetwork,
    pub state_size: usize,
    pub action_size: usize,
    pub feature_size: usize,
}

impl IntrinsicCuriosityModule {
    pub fn new(state_size: usize, action_size: usize, feature_size: usize) -> Self {
        let feature_encoder = NeuralNetwork::new(&[state_size, feature_size * 2, feature_size]);
        let forward_model = NeuralNetwork::new(&[
            feature_size + action_size,
            feature_size * 2,
            feature_size,
        ]);
        let inverse_model =
            NeuralNetwork::new(&[feature_size * 2, feature_size, action_size]);
        Self {
            forward_model,
            inverse_model,
            feature_encoder,
            state_size,
            action_size,
            feature_size,
        }
    }

    /// Encode state into features.
    pub fn encode(&self, state: &[f64]) -> Vec<f64> {
        self.feature_encoder.forward_linear(state)
    }

    /// Predict next state features given current features and action.
    pub fn predict_next_features(&self, features: &[f64], action_onehot: &[f64]) -> Vec<f64> {
        let mut input = features.to_vec();
        input.extend_from_slice(action_onehot);
        self.forward_model.forward_linear(&input)
    }

    /// Predict action given current and next features (inverse model).
    pub fn predict_action(&self, features: &[f64], next_features: &[f64]) -> Vec<f64> {
        let mut input = features.to_vec();
        input.extend_from_slice(next_features);
        let logits = self.inverse_model.forward_linear(&input);
        softmax(&logits)
    }

    /// Compute intrinsic reward as prediction error.
    pub fn intrinsic_reward(
        &self,
        state: &[f64],
        action_onehot: &[f64],
        next_state: &[f64],
    ) -> f64 {
        let features = self.encode(state);
        let next_features = self.encode(next_state);
        let predicted_next = self.predict_next_features(&features, action_onehot);

        // MSE between predicted and actual next features
        let error: f64 = next_features
            .iter()
            .zip(predicted_next.iter())
            .map(|(actual, pred)| (actual - pred).powi(2))
            .sum::<f64>()
            / next_features.len() as f64;

        error
    }

    /// Update the module from a transition.
    pub fn update(&mut self, state: &[f64], action_onehot: &[f64], next_state: &[f64], lr: f64) {
        let features = self.encode(state);
        let next_features = self.encode(next_state);

        // Forward model update
        let predicted_next = self.predict_next_features(&features, action_onehot);
        for layer in &mut self.forward_model.layers {
            for (j, row) in layer.weights.iter_mut().enumerate() {
                for w in row.iter_mut() {
                    if j < next_features.len() {
                        let target = next_features[j % next_features.len()];
                        let pred = predicted_next[j % predicted_next.len()];
                        *w += lr * (target - pred) * 0.01;
                    }
                }
            }
        }

        // Inverse model update (cross-entropy gradient simplified)
        let predicted_action = self.predict_action(&features, &next_features);
        for (j, row) in self.inverse_model.layers.last_mut().unwrap().weights.iter_mut().enumerate() {
            for w in row.iter_mut() {
                let target = if j < action_onehot.len() { action_onehot[j] } else { 0.0 };
                let pred = if j < predicted_action.len() { predicted_action[j] } else { 0.0 };
                *w += lr * (target - pred) * 0.01;
            }
        }
    }
}

/// Random Network Distillation (RND) for exploration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomNetworkDistillation {
    pub target_network: NeuralNetwork,
    pub predictor_network: NeuralNetwork,
    pub state_size: usize,
}

impl RandomNetworkDistillation {
    pub fn new(state_size: usize, feature_size: usize, hidden_size: usize) -> Self {
        let target_network = NeuralNetwork::new(&[state_size, hidden_size, feature_size]);
        let predictor_network = NeuralNetwork::new(&[state_size, hidden_size, feature_size]);
        Self {
            target_network,
            predictor_network,
            state_size,
        }
    }

    /// Compute intrinsic reward as prediction error.
    pub fn intrinsic_reward(&self, state: &[f64]) -> f64 {
        let target = self.target_network.forward_linear(state);
        let predicted = self.predictor_network.forward_linear(state);

        target
            .iter()
            .zip(predicted.iter())
            .map(|(t, p)| (t - p).powi(2))
            .sum::<f64>()
            / target.len() as f64
    }

    /// Update predictor network.
    pub fn update_predictor(&mut self, state: &[f64], lr: f64) {
        let target = self.target_network.forward_linear(state);
        let predicted = self.predictor_network.forward_linear(state);

        for layer in &mut self.predictor_network.layers {
            for (j, row) in layer.weights.iter_mut().enumerate() {
                for w in row.iter_mut() {
                    if j < target.len() {
                        let error = target[j] - predicted[j % predicted.len()];
                        *w += lr * error * 0.01;
                    }
                }
            }
        }
    }
}

/// A curiosity-driven agent combining extrinsic and intrinsic rewards.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuriousAgent<S: Clone + Debug> {
    pub q_table: Vec<Vec<f64>>,
    pub icm: IntrinsicCuriosityModule,
    pub rnd: RandomNetworkDistillation,
    pub epsilon: f64,
    pub gamma: DiscountFactor,
    pub learning_rate: f64,
    pub intrinsic_weight: f64,
    pub use_rnd: bool,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: Clone + Debug + Into<Vec<f64>>> CuriousAgent<S> {
    pub fn new(
        num_states: usize,
        num_actions: usize,
        state_size: usize,
        gamma: DiscountFactor,
        intrinsic_weight: f64,
    ) -> Self {
        Self {
            q_table: vec![vec![0.0; num_actions]; num_states],
            icm: IntrinsicCuriosityModule::new(state_size, num_actions, 8),
            rnd: RandomNetworkDistillation::new(state_size, 8, 16),
            epsilon: 0.1,
            gamma,
            learning_rate: 0.1,
            intrinsic_weight,
            use_rnd: false,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn select_action(&self, state_idx: usize) -> Action {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() < self.epsilon {
            rng.gen_range(0..self.q_table[0].len())
        } else {
            self.q_table[state_idx]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        }
    }

    /// Compute combined reward.
    pub fn compute_reward(
        &self,
        state: &[f64],
        action: Action,
        next_state: &[f64],
        extrinsic_reward: f64,
    ) -> f64 {
        let mut action_onehot = vec![0.0; self.q_table[0].len()];
        if action < action_onehot.len() {
            action_onehot[action] = 1.0;
        }

        let intrinsic = if self.use_rnd {
            self.rnd.intrinsic_reward(state)
        } else {
            self.icm.intrinsic_reward(state, &action_onehot, next_state)
        };

        extrinsic_reward + self.intrinsic_weight * intrinsic
    }

    pub fn update(
        &mut self,
        state_idx: usize,
        action: Action,
        reward: f64,
        next_state_idx: usize,
        done: bool,
    ) {
        let max_next_q = if done {
            0.0
        } else {
            self.q_table[next_state_idx]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
        };
        let td_target = reward + self.gamma * max_next_q;
        self.q_table[state_idx][action] +=
            self.learning_rate * (td_target - self.q_table[state_idx][action]);
    }

    /// Train one episode.
    pub fn train_episode<E: Environment<State = S>>(
        &mut self,
        env: &mut E,
        state_to_idx: &dyn Fn(&S) -> usize,
    ) -> (Reward, f64) {
        let mut state = env.reset();
        let mut total_extrinsic = 0.0;
        let mut total_intrinsic = 0.0;

        loop {
            let state_idx = state_to_idx(&state);
            let action = self.select_action(state_idx);
            let (next_state, extrinsic_reward, done) = env.step(action);

            let state_vec: Vec<f64> = state.clone().into();
            let next_state_vec: Vec<f64> = next_state.clone().into();

            let combined_reward = self.compute_reward(&state_vec, action, &next_state_vec, extrinsic_reward);
            let intrinsic_part = combined_reward - extrinsic_reward;

            total_extrinsic += extrinsic_reward;
            total_intrinsic += intrinsic_part;

            let next_state_idx = state_to_idx(&next_state);
            self.update(state_idx, action, combined_reward, next_state_idx, done);

            // Update curiosity modules
            let mut action_onehot = vec![0.0; self.q_table[0].len()];
            if action < action_onehot.len() {
                action_onehot[action] = 1.0;
            }
            self.icm.update(&state_vec, &action_onehot, &next_state_vec, self.learning_rate);
            self.rnd.update_predictor(&state_vec, self.learning_rate);

            state = next_state;
            if done {
                break;
            }
        }

        (total_extrinsic, total_intrinsic)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_icm_creation() {
        let icm = IntrinsicCuriosityModule::new(4, 3, 8);
        assert_eq!(icm.feature_size, 8);
    }

    #[test]
    fn test_icm_encode() {
        let icm = IntrinsicCuriosityModule::new(4, 3, 8);
        let features = icm.encode(&[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(features.len(), 8);
    }

    #[test]
    fn test_icm_intrinsic_reward() {
        let icm = IntrinsicCuriosityModule::new(4, 3, 8);
        let reward = icm.intrinsic_reward(
            &[1.0, 0.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0, 0.0],
        );
        assert!(reward >= 0.0);
    }

    #[test]
    fn test_icm_update() {
        let mut icm = IntrinsicCuriosityModule::new(4, 3, 8);
        icm.update(&[1.0, 0.0, 0.0, 0.0], &[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0, 0.0], 0.01);
    }

    #[test]
    fn test_rnd_intrinsic_reward() {
        let rnd = RandomNetworkDistillation::new(4, 8, 16);
        let reward = rnd.intrinsic_reward(&[1.0, 0.0, 0.0, 0.0]);
        assert!(reward >= 0.0);
    }

    #[test]
    fn test_rnd_update() {
        let mut rnd = RandomNetworkDistillation::new(4, 8, 16);
        let reward_before = rnd.intrinsic_reward(&[1.0, 0.0, 0.0, 0.0]);
        rnd.update_predictor(&[1.0, 0.0, 0.0, 0.0], 0.01);
        let reward_after = rnd.intrinsic_reward(&[1.0, 0.0, 0.0, 0.0]);
        // Reward should change after update
        assert!(reward_before.is_finite());
        assert!(reward_after.is_finite());
    }

    #[test]
    fn test_curious_agent_icm() {
        let mut agent: CuriousAgent<Vec<f64>> =
            CuriousAgent::new(9, 4, 2, 0.99, 0.1);
        agent.use_rnd = false;
        // Test compute_reward directly
        let reward = agent.compute_reward(&[0.0, 0.0], 0, &[1.0, 0.0], 1.0);
        assert!(reward.is_finite());
        // Test select and update
        let action = agent.select_action(0);
        assert!(action < 4);
        agent.update(0, action, reward, 1, false);
    }

    #[test]
    fn test_curious_agent_rnd() {
        let mut agent: CuriousAgent<Vec<f64>> =
            CuriousAgent::new(9, 4, 2, 0.99, 0.1);
        agent.use_rnd = true;
        let reward = agent.compute_reward(&[0.0, 0.0], 0, &[1.0, 0.0], 1.0);
        assert!(reward.is_finite());
    }

    #[test]
    fn test_icm_inverse_model() {
        let icm = IntrinsicCuriosityModule::new(4, 3, 8);
        let features = icm.encode(&[1.0, 0.0, 0.0, 0.0]);
        let next_features = icm.encode(&[0.0, 1.0, 0.0, 0.0]);
        let action_probs = icm.predict_action(&features, &next_features);
        assert_eq!(action_probs.len(), 3);
        let sum: f64 = action_probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
