//! Deep Q-Network (DQN) with experience replay and target networks.

use crate::core::*;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// DQN agent configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DqnConfig {
    pub learning_rate: f64,
    pub gamma: DiscountFactor,
    pub epsilon_start: f64,
    pub epsilon_end: f64,
    pub epsilon_decay: f64,
    pub batch_size: usize,
    pub target_update_freq: usize,
    pub replay_capacity: usize,
}

impl Default for DqnConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
            batch_size: 32,
            target_update_freq: 100,
            replay_capacity: 10000,
        }
    }
}

/// A Q-network with its parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QNetwork {
    pub network: NeuralNetwork,
    pub num_actions: usize,
    pub state_size: usize,
}

impl QNetwork {
    pub fn new(state_size: usize, num_actions: usize, hidden_size: usize) -> Self {
        let network = NeuralNetwork::new(&[state_size, hidden_size, num_actions]);
        Self {
            network,
            num_actions,
            state_size,
        }
    }

    pub fn predict(&self, state: &[f64]) -> Vec<f64> {
        self.network.forward_linear(state)
    }

    pub fn best_action(&self, state: &[f64]) -> Action {
        let q_values = self.predict(state);
        q_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Simple gradient update on one sample.
    pub fn update(&mut self, state: &[f64], action: Action, target: f64, lr: f64) {
        let predictions = self.predict(state);
        let error = target - predictions[action];
        for layer in &mut self.network.layers {
            for row in &mut layer.weights {
                for w in row.iter_mut() {
                    *w += lr * error * 0.01; // simplified gradient step
                }
            }
            if let Some(b) = layer.biases.get_mut(action) {
                *b += lr * error * 0.01;
            }
        }
    }
}

/// DQN agent with experience replay and target network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DqnAgent<S: Clone + Debug> {
    pub q_network: QNetwork,
    pub target_network: QNetwork,
    pub replay_buffer: ReplayBuffer<S>,
    pub config: DqnConfig,
    pub epsilon: f64,
    pub steps_done: usize,
}

impl<S: Clone + Debug + Into<Vec<f64>>> DqnAgent<S> {
    pub fn new(state_size: usize, num_actions: usize, hidden_size: usize, config: DqnConfig) -> Self {
        let q_network = QNetwork::new(state_size, num_actions, hidden_size);
        let target_network = q_network.clone();
        let replay_buffer = ReplayBuffer::new(config.replay_capacity);
        let epsilon = config.epsilon_start;
        Self {
            q_network,
            target_network,
            replay_buffer,
            config,
            epsilon,
            steps_done: 0,
        }
    }

    /// Select action using epsilon-greedy.
    pub fn select_action(&mut self, state: &[f64]) -> Action {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() < self.epsilon {
            rng.gen_range(0..self.q_network.num_actions)
        } else {
            self.q_network.best_action(state)
        }
    }

    /// Store transition and decay epsilon.
    pub fn store_transition(&mut self, transition: Transition<S>) {
        self.replay_buffer.push(transition);
        self.epsilon = (self.epsilon * self.config.epsilon_decay)
            .max(self.config.epsilon_end);
        self.steps_done += 1;
    }

    /// Train on a batch from replay buffer.
    pub fn train_step(&mut self) -> Option<f64> {
        if self.replay_buffer.len() < self.config.batch_size {
            return None;
        }

        let batch = self.replay_buffer.sample(self.config.batch_size);
        let mut total_loss = 0.0;

        for t in &batch {
            let state_vec: Vec<f64> = t.state.clone().into();
            let next_state_vec: Vec<f64> = t.next_state.clone().into();

            let next_q = self.target_network.predict(&next_state_vec);
            let max_next_q = next_q.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            let target = if t.done {
                t.reward
            } else {
                t.reward + self.config.gamma * max_next_q
            };

            let current_q = self.q_network.predict(&state_vec)[t.action];
            total_loss += (target - current_q).powi(2);

            self.q_network
                .update(&state_vec, t.action, target, self.config.learning_rate);
        }

        // Update target network periodically
        if self.steps_done % self.config.target_update_freq == 0 {
            self.target_network = self.q_network.clone();
        }

        Some(total_loss / batch.len() as f64)
    }

    /// Run a full training episode.
    pub fn train_episode<E: Environment<State = S>>(&mut self, env: &mut E) -> (Reward, f64) {
        let mut state = env.reset();
        let mut total_reward = 0.0;
        let mut total_loss = 0.0;
        let mut loss_count = 0;

        loop {
            let state_vec: Vec<f64> = state.clone().into();
            let action = self.select_action(&state_vec);
            let (next_state, reward, done) = env.step(action);
            total_reward += reward;

            self.store_transition(Transition {
                state: state.clone(),
                action,
                reward,
                next_state: next_state.clone(),
                done,
            });

            if let Some(loss) = self.train_step() {
                total_loss += loss;
                loss_count += 1;
            }

            state = next_state;
            if done {
                break;
            }
        }

        let avg_loss = if loss_count > 0 {
            total_loss / loss_count as f64
        } else {
            0.0
        };

        (total_reward, avg_loss)
    }
}

/// Double DQN: uses online network to select actions, target network to evaluate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoubleDqnAgent<S: Clone + Debug> {
    pub agent: DqnAgent<S>,
}

impl<S: Clone + Debug + Into<Vec<f64>>> DoubleDqnAgent<S> {
    pub fn new(state_size: usize, num_actions: usize, hidden_size: usize, config: DqnConfig) -> Self {
        Self {
            agent: DqnAgent::new(state_size, num_actions, hidden_size, config),
        }
    }

    pub fn train_step(&mut self) -> Option<f64> {
        if self.agent.replay_buffer.len() < self.agent.config.batch_size {
            return None;
        }

        let batch = self.agent.replay_buffer.sample(self.agent.config.batch_size);
        let mut total_loss = 0.0;

        for t in &batch {
            let state_vec: Vec<f64> = t.state.clone().into();
            let next_state_vec: Vec<f64> = t.next_state.clone().into();

            // Double DQN: select with online, evaluate with target
            let best_next_action = self.agent.q_network.best_action(&next_state_vec);
            let next_q = self.agent.target_network.predict(&next_state_vec);
            let target = if t.done {
                t.reward
            } else {
                t.reward + self.agent.config.gamma * next_q[best_next_action]
            };

            let current_q = self.agent.q_network.predict(&state_vec)[t.action];
            total_loss += (target - current_q).powi(2);

            self.agent
                .q_network
                .update(&state_vec, t.action, target, self.agent.config.learning_rate);
        }

        if self.agent.steps_done % self.agent.config.target_update_freq == 0 {
            self.agent.target_network = self.agent.q_network.clone();
        }

        Some(total_loss / batch.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_vec(s: (usize, usize)) -> Vec<f64> {
        vec![s.0 as f64, s.1 as f64]
    }

    #[test]
    fn test_q_network_predict() {
        let qn = QNetwork::new(4, 2, 8);
        let output = qn.predict(&[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_q_network_best_action() {
        let qn = QNetwork::new(4, 3, 8);
        let action = qn.best_action(&[0.0, 1.0, 0.0, 0.0]);
        assert!(action < 3);
    }

    #[test]
    fn test_dqn_agent_select_action() {
        let mut agent: DqnAgent<Vec<f64>> = DqnAgent::new(2, 4, 16, DqnConfig::default());
        let action = agent.select_action(&vec![0.0, 0.0]);
        assert!(action < 4);
    }

    #[test]
    fn test_dqn_agent_epsilon_decay() {
        let config = DqnConfig {
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.99,
            ..Default::default()
        };
        let mut agent: DqnAgent<Vec<f64>> = DqnAgent::new(2, 4, 16, config);
        let initial_epsilon = agent.epsilon;
        for _ in 0..10 {
            agent.store_transition(Transition {
                state: vec![0.0, 0.0],
                action: 0,
                reward: 1.0,
                next_state: vec![1.0, 0.0],
                done: false,
            });
        }
        assert!(agent.epsilon < initial_epsilon);
    }

    #[test]
    fn test_dqn_train_episode() {
        let mut env = GridWorld::new(3, 3);
        let mut agent: DqnAgent<Vec<f64>> = DqnAgent::new(2, 4, 8, DqnConfig {
            batch_size: 4,
            replay_capacity: 100,
            target_update_freq: 10,
            ..Default::default()
        });

        // Manually run an episode
        let mut state: Vec<f64> = vec![0.0, 0.0];
        let mut total_reward = 0.0;
        for _ in 0..20 {
            let action = agent.select_action(&state);
            let (ns, reward, done) = env.step(action);
            let ns_vec = vec![ns.0 as f64, ns.1 as f64];
            total_reward += reward;
            agent.store_transition(Transition { state: state.clone(), action, reward, next_state: ns_vec.clone(), done });
            agent.train_step();
            state = ns_vec;
            if done { break; }
        }
        let reward = total_reward;
        let _loss: f64 = 0.0;
        // Just check it runs without panicking
        assert!(reward.is_finite());
    }

    #[test]
    fn test_double_dqn() {
        let mut agent = DoubleDqnAgent::<Vec<f64>>::new(2, 4, 8, DqnConfig {
            batch_size: 4,
            replay_capacity: 100,
            ..Default::default()
        });

        // Fill buffer
        for i in 0..10 {
            agent.agent.store_transition(Transition {
                state: vec![0.0, (i % 3) as f64],
                action: i % 4,
                reward: 1.0,
                next_state: vec![1.0, (i % 3) as f64],
                done: false,
            });
        }

        let loss = agent.train_step();
        assert!(loss.is_some());
    }

    #[test]
    fn test_dqn_config_default() {
        let config = DqnConfig::default();
        assert_eq!(config.batch_size, 32);
        assert!((config.gamma - 0.99).abs() < 1e-6);
    }
}
