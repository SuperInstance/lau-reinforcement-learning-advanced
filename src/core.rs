//! Core types and traits shared across all RL algorithms.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// An action identifier.
pub type Action = usize;

/// A scalar reward.
pub type Reward = f64;

/// A discount factor (typically 0.99).
pub type DiscountFactor = f64;

/// A step in the environment: (state, action, reward, next_state, done).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transition<S: Clone + Debug> {
    pub state: S,
    pub action: Action,
    pub reward: Reward,
    pub next_state: S,
    pub done: bool,
}

/// An environment trait that RL agents interact with.
pub trait Environment: Clone + Debug {
    type State: Clone + Debug;

    /// Reset the environment, returning the initial state.
    fn reset(&mut self) -> Self::State;

    /// Take an action, returning (next_state, reward, done).
    fn step(&mut self, action: Action) -> (Self::State, Reward, bool);

    /// Number of possible actions.
    fn num_actions(&self) -> usize;

    /// Render or describe the current state (optional).
    fn render(&self) -> String {
        String::new()
    }
}

/// A policy maps states to action probabilities or selections.
pub trait Policy<S: Clone + Debug> {
    /// Select an action given a state.
    fn select_action(&self, state: &S) -> Action;

    /// Get action probabilities (if applicable).
    fn action_probabilities(&self, state: &S) -> Option<Vec<f64>> {
        None
    }
}

/// A neural network layer (simplified).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (input_size + output_size) as f64).sqrt();
        let weights = (0..output_size)
            .map(|_| {
                (0..input_size)
                    .map(|_| rng.gen_range(-scale..scale))
                    .collect()
            })
            .collect();
        let biases = vec![0.0; output_size];
        Self { weights, biases }
    }

    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        self.weights
            .iter()
            .zip(self.biases.iter())
            .map(|(row, bias)| {
                row.iter()
                    .zip(input.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f64>()
                    + bias
            })
            .collect()
    }

    pub fn output_size(&self) -> usize {
        self.biases.len()
    }

    pub fn input_size(&self) -> usize {
        self.weights.first().map(|r| r.len()).unwrap_or(0)
    }
}

/// A simple feedforward network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new(layer_sizes: &[usize]) -> Self {
        let layers = layer_sizes
            .windows(2)
            .map(|w| Layer::new(w[0], w[1]))
            .collect();
        Self { layers }
    }

    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut output = input.to_vec();
        for layer in &self.layers {
            output = layer.forward(&output);
            // ReLU for hidden layers (not last)
            output = output.into_iter().map(|x| x.max(0.0)).collect();
        }
        output
    }

    /// Forward pass without activation on the last layer.
    pub fn forward_linear(&self, input: &[f64]) -> Vec<f64> {
        let mut output = input.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            output = layer.forward(&output);
            if i < self.layers.len() - 1 {
                output = output.into_iter().map(|x| x.max(0.0)).collect();
            }
        }
        output
    }

    pub fn num_parameters(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.weights.len() * l.weights[0].len() + l.biases.len())
            .sum()
    }
}

/// A circular replay buffer for experience replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayBuffer<S: Clone + Debug> {
    buffer: Vec<Transition<S>>,
    capacity: usize,
    position: usize,
}

impl<S: Clone + Debug> ReplayBuffer<S> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            position: 0,
        }
    }

    pub fn push(&mut self, transition: Transition<S>) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(transition);
        } else {
            self.buffer[self.position] = transition;
        }
        self.position = (self.position + 1) % self.capacity;
    }

    pub fn sample(&self, batch_size: usize) -> Vec<Transition<S>> {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        self.buffer
            .choose_multiple(&mut rng, batch_size.min(self.buffer.len()))
            .cloned()
            .collect()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
        self.position = 0;
    }
}

/// Compute discounted returns from a sequence of rewards.
pub fn compute_returns(rewards: &[Reward], gamma: DiscountFactor) -> Vec<Reward> {
    let mut returns = Vec::with_capacity(rewards.len());
    let mut g = 0.0;
    for &r in rewards.iter().rev() {
        g = r + gamma * g;
        returns.push(g);
    }
    returns.reverse();
    returns
}

/// Compute advantages using Generalized Advantage Estimation (GAE).
pub fn compute_gae(
    rewards: &[Reward],
    values: &[f64],
    gamma: DiscountFactor,
    lambda: f64,
) -> Vec<f64> {
    let mut advantages = Vec::with_capacity(rewards.len());
    let mut last_advantage = 0.0;
    for t in (0..rewards.len()).rev() {
        let next_value = if t + 1 < values.len() { values[t + 1] } else { 0.0 };
        let delta = rewards[t] + gamma * next_value - values[t];
        last_advantage = delta + gamma * lambda * last_advantage;
        advantages.push(last_advantage);
    }
    advantages.reverse();
    advantages
}

/// Softmax function.
pub fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|x| (x - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|x| x / sum).collect()
}

/// An episode trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory<S: Clone + Debug> {
    pub transitions: Vec<Transition<S>>,
}

impl<S: Clone + Debug> Trajectory<S> {
    pub fn new() -> Self {
        Self {
            transitions: Vec::new(),
        }
    }

    pub fn push(&mut self, t: Transition<S>) {
        self.transitions.push(t);
    }

    pub fn rewards(&self) -> Vec<Reward> {
        self.transitions.iter().map(|t| t.reward).collect()
    }

    pub fn total_reward(&self) -> Reward {
        self.transitions.iter().map(|t| t.reward).sum()
    }

    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }
}

impl<S: Clone + Debug> Default for Trajectory<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple environment: a discrete grid world.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridWorld {
    pub width: usize,
    pub height: usize,
    pub agent_pos: (usize, usize),
    pub goal_pos: (usize, usize),
    pub obstacles: Vec<(usize, usize)>,
    pub max_steps: usize,
    pub steps: usize,
}

impl GridWorld {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            agent_pos: (0, 0),
            goal_pos: (width - 1, height - 1),
            obstacles: Vec::new(),
            max_steps: width * height * 2,
            steps: 0,
        }
    }

    pub fn with_obstacles(mut self, obstacles: Vec<(usize, usize)>) -> Self {
        self.obstacles = obstacles;
        self
    }

    pub fn state_index(&self) -> usize {
        self.agent_pos.1 * self.width + self.agent_pos.0
    }

    pub fn state_vector(&self) -> DVector<f64> {
        let mut v = vec![0.0; self.width * self.height];
        v[self.state_index()] = 1.0;
        DVector::from_vec(v)
    }

    pub fn num_states(&self) -> usize {
        self.width * self.height
    }
}

impl Environment for GridWorld {
    type State = (usize, usize);

    fn reset(&mut self) -> Self::State {
        self.agent_pos = (0, 0);
        self.steps = 0;
        self.agent_pos
    }

    fn step(&mut self, action: Action) -> (Self::State, Reward, bool) {
        // 0=up, 1=right, 2=down, 3=left
        let (x, y) = self.agent_pos;
        let new_pos = match action {
            0 => (x, y.saturating_sub(1)),
            1 => (x.saturating_add(1).min(self.width - 1), y),
            2 => (x, y.saturating_add(1).min(self.height - 1)),
            3 => (x.saturating_sub(1), y),
            _ => (x, y),
        };

        if !self.obstacles.contains(&new_pos) {
            self.agent_pos = new_pos;
        }
        self.steps += 1;

        let done = self.agent_pos == self.goal_pos || self.steps >= self.max_steps;
        let reward = if self.agent_pos == self.goal_pos {
            1.0
        } else {
            -0.01
        };

        (self.agent_pos, reward, done)
    }

    fn num_actions(&self) -> usize {
        4
    }
}

/// A simple bandit environment for testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanditEnv {
    pub arms: Vec<f64>,
    pub rng_seed: Option<u64>,
}

impl BanditEnv {
    pub fn new(arms: Vec<f64>) -> Self {
        Self {
            arms,
            rng_seed: None,
        }
    }
}

impl Environment for BanditEnv {
    type State = ();

    fn reset(&mut self) -> Self::State {}

    fn step(&mut self, action: Action) -> (Self::State, Reward, bool) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mean = self.arms.get(action).copied().unwrap_or(0.0);
        let reward = mean + rng.gen_range(-0.5..0.5);
        ((), reward, true)
    }

    fn num_actions(&self) -> usize {
        self.arms.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_returns() {
        let rewards = vec![1.0, 1.0, 1.0];
        let returns = compute_returns(&rewards, 0.99);
        assert!((returns[0] - 2.9701).abs() < 1e-4);
        assert!((returns[1] - 1.99).abs() < 1e-4);
        assert!((returns[2] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_compute_gae() {
        let rewards = vec![1.0, 0.0, 1.0];
        let values = vec![0.5, 0.5, 0.5];
        let advantages = compute_gae(&rewards, &values, 0.99, 0.95);
        assert_eq!(advantages.len(), 3);
        // Advantages should sum to something reasonable
        let sum: f64 = advantages.iter().sum();
        assert!(sum > 0.0);
    }

    #[test]
    fn test_softmax() {
        let probs = softmax(&[1.0, 2.0, 3.0]);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_replay_buffer() {
        let mut buf: ReplayBuffer<i32> = ReplayBuffer::new(5);
        assert!(buf.is_empty());
        for i in 0..7 {
            buf.push(Transition {
                state: i,
                action: 0,
                reward: 1.0,
                next_state: i + 1,
                done: false,
            });
        }
        assert_eq!(buf.len(), 5);
        let sample = buf.sample(3);
        assert_eq!(sample.len(), 3);
    }

    #[test]
    fn test_neural_network_forward() {
        let net = NeuralNetwork::new(&[3, 4, 2]);
        let output = net.forward_linear(&[1.0, 0.0, 0.0]);
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_layer_new() {
        let layer = Layer::new(4, 3);
        assert_eq!(layer.input_size(), 4);
        assert_eq!(layer.output_size(), 3);
        let output = layer.forward(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_grid_world() {
        let mut gw = GridWorld::new(5, 5);
        let state = gw.reset();
        assert_eq!(state, (0, 0));
        let (next, reward, done) = gw.step(1); // right
        assert_eq!(next, (1, 0));
        assert!(!done);
    }

    #[test]
    fn test_grid_world_goal() {
        let mut gw = GridWorld::new(2, 2);
        gw.reset();
        gw.step(1); // right: (1,0)
        let (_, reward, done) = gw.step(2); // down: (1,1) = goal
        assert!(done);
        assert!((reward - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_trajectory() {
        let mut traj: Trajectory<i32> = Trajectory::new();
        assert!(traj.is_empty());
        traj.push(Transition {
            state: 0,
            action: 1,
            reward: 0.5,
            next_state: 1,
            done: false,
        });
        assert_eq!(traj.len(), 1);
        assert!((traj.total_reward() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_bandit_env() {
        let mut env = BanditEnv::new(vec![0.2, 0.5, 0.8]);
        env.reset();
        let (_, reward, done) = env.step(2);
        assert!(done);
    }

    #[test]
    fn test_neural_network_num_params() {
        let net = NeuralNetwork::new(&[3, 4, 2]);
        // Layer 0: 3*4 + 4 = 16
        // Layer 1: 4*2 + 2 = 10
        assert_eq!(net.num_parameters(), 26);
    }

    #[test]
    fn test_replay_buffer_overflow() {
        let mut buf: ReplayBuffer<i32> = ReplayBuffer::new(3);
        for i in 0..5 {
            buf.push(Transition {
                state: i,
                action: 0,
                reward: 0.0,
                next_state: i,
                done: false,
            });
        }
        assert_eq!(buf.len(), 3);
        // oldest should be evicted
    }
}
