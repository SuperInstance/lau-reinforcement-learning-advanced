//! Model-based RL: Dyna, world models, MBPO.

use crate::core::*;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// A learned dynamics model (world model).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldModel {
    pub transition_network: NeuralNetwork,
    pub reward_network: NeuralNetwork,
    pub state_size: usize,
    pub num_actions: usize,
}

impl WorldModel {
    pub fn new(state_size: usize, num_actions: usize, hidden_size: usize) -> Self {
        // Transition: [state, action_onehot] -> next_state
        let transition_network =
            NeuralNetwork::new(&[state_size + num_actions, hidden_size, state_size]);
        // Reward: [state, action_onehot] -> reward
        let reward_network =
            NeuralNetwork::new(&[state_size + num_actions, hidden_size, 1]);
        Self {
            transition_network,
            reward_network,
            state_size,
            num_actions,
        }
    }

    /// Predict next state and reward.
    pub fn predict(&self, state: &[f64], action: Action) -> (Vec<f64>, f64) {
        let mut input = state.to_vec();
        let mut action_onehot = vec![0.0; self.num_actions];
        if action < self.num_actions {
            action_onehot[action] = 1.0;
        }
        input.extend_from_slice(&action_onehot);

        let next_state = self.transition_network.forward_linear(&input);
        let reward = self.reward_network.forward_linear(&input)[0];
        (next_state, reward)
    }

    /// Update the world model from a real transition.
    pub fn update(&mut self, state: &[f64], action: Action, next_state: &[f64], reward: f64, lr: f64) {
        let mut input = state.to_vec();
        let mut action_onehot = vec![0.0; self.num_actions];
        if action < self.num_actions {
            action_onehot[action] = 1.0;
        }
        input.extend_from_slice(&action_onehot);

        let predicted_next = self.transition_network.forward_linear(&input);
        let n_layers = self.transition_network.layers.len();
        for (i, layer) in self.transition_network.layers.iter_mut().enumerate() {
            for (j, row) in layer.weights.iter_mut().enumerate() {
                for w in row.iter_mut() {
                    if i == n_layers - 1 && j < next_state.len() {
                        *w += lr * (next_state[j] - predicted_next[j.min(predicted_next.len() - 1)]) * 0.01;
                    }
                }
            }
        }

        // Update reward network
        let predicted_reward = self.reward_network.forward_linear(&input)[0];
        let reward_error = reward - predicted_reward;
        for layer in &mut self.reward_network.layers {
            for row in &mut layer.weights {
                for w in row.iter_mut() {
                    *w += lr * reward_error * 0.01;
                }
            }
            if let Some(b) = layer.biases.first_mut() {
                *b += lr * reward_error * 0.01;
            }
        }
    }

    /// Generate imagined trajectories.
    pub fn imagine_trajectory(
        &self,
        initial_state: &[f64],
        policy: &dyn Fn(&[f64]) -> Action,
        max_steps: usize,
    ) -> Vec<(Vec<f64>, Action, f64)> {
        let mut trajectory = Vec::new();
        let mut state = initial_state.to_vec();

        for _ in 0..max_steps {
            let action = policy(&state);
            let (next_state, reward) = self.predict(&state, action);
            trajectory.push((state.clone(), action, reward));
            state = next_state;
        }

        trajectory
    }
}

/// Dyna-Q agent: combines model-free Q-learning with model-based planning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynaQAgent<S: Clone + Debug> {
    pub q_values: Vec<Vec<f64>>,
    pub world_model: WorldModel,
    pub gamma: DiscountFactor,
    pub learning_rate: f64,
    pub epsilon: f64,
    pub planning_steps: usize,
    pub model_buffer: Vec<(Vec<f64>, Action, Vec<f64>, f64)>,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: Clone + Debug + Into<Vec<f64>>> DynaQAgent<S> {
    pub fn new(
        num_states: usize,
        num_actions: usize,
        state_size: usize,
        gamma: DiscountFactor,
        learning_rate: f64,
        planning_steps: usize,
    ) -> Self {
        Self {
            q_values: vec![vec![0.0; num_actions]; num_states],
            world_model: WorldModel::new(state_size, num_actions, 16),
            gamma,
            learning_rate,
            epsilon: 0.1,
            planning_steps,
            model_buffer: Vec::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn select_action(&self, state_idx: usize) -> Action {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() < self.epsilon {
            rng.gen_range(0..self.q_values[0].len())
        } else {
            self.q_values[state_idx]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        }
    }

    /// Q-learning update.
    pub fn q_update(&mut self, state_idx: usize, action: Action, reward: f64, next_state_idx: usize, done: bool) {
        let max_next_q = if done {
            0.0
        } else {
            self.q_values[next_state_idx]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
        };
        let td_target = reward + self.gamma * max_next_q;
        let td_error = td_target - self.q_values[state_idx][action];
        self.q_values[state_idx][action] += self.learning_rate * td_error;
    }

    /// Planning step: sample from model and update Q-values.
    pub fn planning_step(&mut self) {
        if self.model_buffer.is_empty() {
            return;
        }

        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        for _ in 0..self.planning_steps {
            if let Some((state, action, next_state, reward)) =
                self.model_buffer.choose(&mut rng).cloned()
            {
                self.world_model.update(&state, action, &next_state, reward, self.learning_rate);
                // Use world model for additional Q-updates
                let (imagined_next, imagined_reward) = self.world_model.predict(&state, action);
                // Simplified: use argmax of imagined state as index
                let imagined_next_idx = imagined_next
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let state_idx = state
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                self.q_update(
                    state_idx.min(self.q_values.len() - 1),
                    action,
                    imagined_reward,
                    imagined_next_idx.min(self.q_values.len() - 1),
                    false,
                );
            }
        }
    }

    /// Train one episode.
    pub fn train_episode<E: Environment<State = S>>(
        &mut self,
        env: &mut E,
        state_to_idx: &dyn Fn(&S) -> usize,
    ) -> Reward {
        let mut state = env.reset();
        let mut total_reward = 0.0;

        loop {
            let state_idx = state_to_idx(&state);
            let action = self.select_action(state_idx);
            let (next_state, reward, done) = env.step(action);
            let next_state_idx = state_to_idx(&next_state);
            total_reward += reward;

            let state_vec: Vec<f64> = state.clone().into();
            let next_state_vec: Vec<f64> = next_state.clone().into();

            // Q-learning update
            self.q_update(state_idx, action, reward, next_state_idx, done);

            // Store in model buffer
            self.model_buffer.push((state_vec, action, next_state_vec, reward));
            if self.model_buffer.len() > 1000 {
                self.model_buffer.remove(0);
            }

            // Planning
            self.planning_step();

            state = next_state;
            if done {
                break;
            }
        }

        total_reward
    }
}

/// MBPO (Model-Based Policy Optimization) agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MbpoAgent<S: Clone + Debug> {
    pub world_model: WorldModel,
    pub q_network: NeuralNetwork,
    pub gamma: DiscountFactor,
    pub learning_rate: f64,
    pub model_retain_epochs: usize,
    pub imagination_ratio: usize,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: Clone + Debug + Into<Vec<f64>>> MbpoAgent<S> {
    pub fn new(
        state_size: usize,
        num_actions: usize,
        hidden_size: usize,
        gamma: DiscountFactor,
    ) -> Self {
        Self {
            world_model: WorldModel::new(state_size, num_actions, hidden_size),
            q_network: NeuralNetwork::new(&[state_size, hidden_size, num_actions]),
            gamma,
            learning_rate: 0.001,
            model_retain_epochs: 5,
            imagination_ratio: 4,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn select_action(&self, state: &[f64]) -> Action {
        let q_values = self.q_network.forward_linear(state);
        q_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Update world model from real experience.
    pub fn update_model(&mut self, state: &[f64], action: Action, next_state: &[f64], reward: f64) {
        self.world_model.update(state, action, next_state, reward, self.learning_rate);
    }

    /// Generate imagined data and update Q-function.
    pub fn imagined_update(&mut self, initial_states: &[Vec<f64>]) -> f64 {
        let mut total_loss = 0.0;
        let mut count = 0;

        for state in initial_states {
            for _ in 0..self.imagination_ratio {
                let action = self.select_action(state);
                let (next_state, reward) = self.world_model.predict(state, action);

                // Q-learning target
                let next_q = self.q_network.forward_linear(&next_state);
                let max_next_q = next_q.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let target = reward + self.gamma * max_next_q;

                let current_q = self.q_network.forward_linear(state)[action];
                let error = target - current_q;
                total_loss += error.powi(2);
                count += 1;

                // Simplified update
                for layer in &mut self.q_network.layers {
                    for row in &mut layer.weights {
                        for w in row.iter_mut() {
                            *w += self.learning_rate * error * 0.01;
                        }
                    }
                }
            }
        }

        if count > 0 {
            total_loss / count as f64
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world_model_predict() {
        let wm = WorldModel::new(4, 3, 8);
        let (next_state, reward) = wm.predict(&[1.0, 0.0, 0.0, 0.0], 0);
        assert_eq!(next_state.len(), 4);
        assert!(reward.is_finite());
    }

    #[test]
    fn test_world_model_update() {
        let mut wm = WorldModel::new(4, 3, 8);
        wm.update(&[1.0, 0.0, 0.0, 0.0], 0, &[0.0, 1.0, 0.0, 0.0], 1.0, 0.01);
        let (next_state, reward) = wm.predict(&[1.0, 0.0, 0.0, 0.0], 0);
        assert_eq!(next_state.len(), 4);
    }

    #[test]
    fn test_world_model_imagine() {
        let wm = WorldModel::new(4, 3, 8);
        let policy = |state: &[f64]| -> Action {
            state.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i).unwrap_or(0) % 3
        };
        let traj = wm.imagine_trajectory(&[1.0, 0.0, 0.0, 0.0], &policy, 5);
        assert_eq!(traj.len(), 5);
    }

    #[test]
    fn test_dyna_q_agent() {
        let mut agent: DynaQAgent<Vec<f64>> = DynaQAgent::new(9, 4, 2, 0.99, 0.1, 5);
        let state_idx = 0;
        let action = agent.select_action(state_idx);
        assert!(action < 4);
        agent.q_update(state_idx, action, 1.0, 1, false);
        assert!(agent.q_values[state_idx][action] > 0.0);
    }

    #[test]
    fn test_dyna_q_select_action() {
        let agent: DynaQAgent<Vec<f64>> = DynaQAgent::new(9, 4, 2, 0.99, 0.1, 5);
        let action = agent.select_action(0);
        assert!(action < 4);
    }

    #[test]
    fn test_mbpo_agent() {
        let mut agent: MbpoAgent<Vec<f64>> = MbpoAgent::new(2, 4, 8, 0.99);
        let action = agent.select_action(&[0.0, 1.0]);
        assert!(action < 4);

        agent.update_model(&[0.0, 1.0], 0, &[1.0, 1.0], 0.5);
        let loss = agent.imagined_update(&[vec![0.0, 1.0]]);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_world_model_serialization() {
        let wm = WorldModel::new(4, 3, 8);
        let json = serde_json::to_string(&wm).unwrap();
        let deserialized: WorldModel = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.state_size, 4);
        assert_eq!(deserialized.num_actions, 3);
    }
}
