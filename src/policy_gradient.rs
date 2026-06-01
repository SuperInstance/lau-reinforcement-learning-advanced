//! Policy gradient methods: REINFORCE, baseline, advantage estimation.

use crate::core::*;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// A parameterized policy network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyNetwork {
    pub network: NeuralNetwork,
    pub num_actions: usize,
}

impl PolicyNetwork {
    pub fn new(state_size: usize, num_actions: usize, hidden_size: usize) -> Self {
        let network = NeuralNetwork::new(&[state_size, hidden_size, num_actions]);
        Self { network, num_actions }
    }

    pub fn action_probabilities(&self, state: &[f64]) -> Vec<f64> {
        let logits = self.network.forward_linear(state);
        softmax(&logits)
    }

    pub fn select_action(&self, state: &[f64]) -> Action {
        use rand::Rng;
        let probs = self.action_probabilities(state);
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

    /// Log probability of an action.
    pub fn log_prob(&self, state: &[f64], action: Action) -> f64 {
        let probs = self.action_probabilities(state);
        probs[action].ln().max(-10.0)
    }

    /// Update policy parameters using gradient ascent.
    pub fn update(&mut self, state: &[f64], action: Action, advantage: f64, lr: f64) {
        let probs = self.action_probabilities(state);
        for layer in &mut self.network.layers {
            for (j, row) in layer.weights.iter_mut().enumerate() {
                for w in row.iter_mut() {
                    if j == action {
                        *w += lr * advantage * (1.0 - probs[action]);
                    } else if j < probs.len() {
                        *w -= lr * advantage * probs[j];
                    }
                }
            }
        }
    }
}

/// Type of baseline to use for variance reduction.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BaselineType {
    None,
    MeanReturn,
    ValueFunction { hidden_size: usize },
}

/// REINFORCE agent (Monte Carlo policy gradient).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReinforceAgent<S: Clone + Debug> {
    pub policy: PolicyNetwork,
    pub gamma: DiscountFactor,
    pub learning_rate: f64,
    pub baseline: BaselineType,
    _phantom: std::marker::PhantomData<S>,
}

/// A simple value function estimator for baselines.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueFunction {
    pub network: NeuralNetwork,
}

impl ValueFunction {
    pub fn new(state_size: usize, hidden_size: usize) -> Self {
        let network = NeuralNetwork::new(&[state_size, hidden_size, 1]);
        Self { network }
    }

    pub fn predict(&self, state: &[f64]) -> f64 {
        self.network.forward_linear(state)[0]
    }

    pub fn update(&mut self, state: &[f64], target: f64, lr: f64) {
        let prediction = self.predict(state);
        let error = target - prediction;
        for layer in &mut self.network.layers {
            for row in &mut layer.weights {
                for w in row.iter_mut() {
                    *w += lr * error * 0.01;
                }
            }
            if let Some(b) = layer.biases.first_mut() {
                *b += lr * error * 0.01;
            }
        }
    }
}

impl<S: Clone + Debug + Into<Vec<f64>>> ReinforceAgent<S> {
    pub fn new(
        state_size: usize,
        num_actions: usize,
        hidden_size: usize,
        gamma: DiscountFactor,
        learning_rate: f64,
        baseline: BaselineType,
    ) -> Self {
        let policy = PolicyNetwork::new(state_size, num_actions, hidden_size);
        Self {
            policy,
            gamma,
            learning_rate,
            baseline,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Collect a full episode trajectory.
    pub fn collect_episode<E: Environment<State = S>>(&self, env: &mut E) -> Trajectory<S> {
        let mut traj = Trajectory::new();
        let mut state = env.reset();
        loop {
            let state_vec: Vec<f64> = state.clone().into();
            let action = self.policy.select_action(&state_vec);
            let (next_state, reward, done) = env.step(action);
            traj.push(Transition {
                state: state.clone(),
                action,
                reward,
                next_state: next_state.clone(),
                done,
            });
            state = next_state;
            if done {
                break;
            }
        }
        traj
    }

    /// Update policy from a completed episode.
    pub fn update_from_episode(&mut self, trajectory: &Trajectory<S>) -> f64 {
        let rewards: Vec<Reward> = trajectory.rewards();
        let returns = compute_returns(&rewards, self.gamma);

        let baseline_value = match &self.baseline {
            BaselineType::None => 0.0,
            BaselineType::MeanReturn => {
                let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
                mean
            }
            BaselineType::ValueFunction { .. } => 0.0,
        };

        let mut total_loss = 0.0;
        for (i, t) in trajectory.transitions.iter().enumerate() {
            let state_vec: Vec<f64> = t.state.clone().into();
            let advantage = match &self.baseline {
                BaselineType::ValueFunction { hidden_size } => {
                    let vf = ValueFunction::new(state_vec.len(), *hidden_size);
                    returns[i] - vf.predict(&state_vec)
                }
                _ => returns[i] - baseline_value,
            };

            self.policy.update(&state_vec, t.action, advantage, self.learning_rate);
            total_loss += advantage.powi(2);
        }

        total_loss / trajectory.len().max(1) as f64
    }

    /// Train for multiple episodes.
    pub fn train<E: Environment<State = S>>(
        &mut self,
        env: &mut E,
        num_episodes: usize,
    ) -> Vec<(Reward, f64)> {
        let mut results = Vec::new();
        for _ in 0..num_episodes {
            let trajectory = self.collect_episode(env);
            let total_reward = trajectory.total_reward();
            let loss = self.update_from_episode(&trajectory);
            results.push((total_reward, loss));
        }
        results
    }
}

/// REINFORCE with a learned baseline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReinforceWithBaseline<S: Clone + Debug> {
    pub policy: PolicyNetwork,
    pub value_fn: ValueFunction,
    pub gamma: DiscountFactor,
    pub policy_lr: f64,
    pub value_lr: f64,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: Clone + Debug + Into<Vec<f64>>> ReinforceWithBaseline<S> {
    pub fn new(
        state_size: usize,
        num_actions: usize,
        hidden_size: usize,
        gamma: DiscountFactor,
        policy_lr: f64,
        value_lr: f64,
    ) -> Self {
        Self {
            policy: PolicyNetwork::new(state_size, num_actions, hidden_size),
            value_fn: ValueFunction::new(state_size, hidden_size),
            gamma,
            policy_lr,
            value_lr,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn train_episode<E: Environment<State = S>>(&mut self, env: &mut E) -> (Reward, f64) {
        let mut state = env.reset();
        let mut states = Vec::new();
        let mut actions = Vec::new();
        let mut rewards = Vec::new();
        let mut total_reward = 0.0;

        loop {
            let state_vec: Vec<f64> = state.clone().into();
            let action = self.policy.select_action(&state_vec);
            let (next_state, reward, done) = env.step(action);
            states.push(state_vec);
            actions.push(action);
            rewards.push(reward);
            total_reward += reward;
            state = next_state;
            if done {
                break;
            }
        }

        let returns = compute_returns(&rewards, self.gamma);
        let mut total_loss = 0.0;

        for (i, (state_vec, &action)) in states.iter().zip(actions.iter()).enumerate() {
            let baseline = self.value_fn.predict(state_vec);
            let advantage = returns[i] - baseline;

            self.policy.update(state_vec, action, advantage, self.policy_lr);
            self.value_fn.update(state_vec, returns[i], self.value_lr);

            total_loss += advantage.powi(2);
        }

        let avg_loss = total_loss / states.len().max(1) as f64;
        (total_reward, avg_loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gv(x: usize, y: usize) -> Vec<f64> {
        vec![x as f64, y as f64]
    }

    /// A simple Vec<f64> environment wrapper around GridWorld.
    #[derive(Debug, Clone)]
    struct VecGridWorld {
        inner: GridWorld,
    }

    impl Environment for VecGridWorld {
        type State = Vec<f64>;
        fn reset(&mut self) -> Vec<f64> {
            let s = self.inner.reset();
            gv(s.0, s.1)
        }
        fn step(&mut self, action: Action) -> (Vec<f64>, Reward, bool) {
            let (s, r, d) = self.inner.step(action);
            (gv(s.0, s.1), r, d)
        }
        fn num_actions(&self) -> usize { self.inner.num_actions() }
    }

    #[test]
    fn test_policy_network_probabilities() {
        let pn = PolicyNetwork::new(4, 3, 8);
        let probs = pn.action_probabilities(&[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(probs.len(), 3);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_policy_network_select_action() {
        let pn = PolicyNetwork::new(4, 3, 8);
        let action = pn.select_action(&[0.0, 1.0, 0.0, 0.0]);
        assert!(action < 3);
    }

    #[test]
    fn test_policy_network_log_prob() {
        let pn = PolicyNetwork::new(4, 3, 8);
        let lp = pn.log_prob(&[1.0, 0.0, 0.0, 0.0], 0);
        assert!(lp.is_finite());
        assert!(lp <= 0.0);
    }

    #[test]
    fn test_reinforce_agent() {
        let mut env = VecGridWorld { inner: GridWorld::new(3, 3) };
        let mut agent: ReinforceAgent<Vec<f64>> = ReinforceAgent::new(
            2, 4, 8, 0.99, 0.01, BaselineType::None,
        );
        let results = agent.train(&mut env, 3);
        assert_eq!(results.len(), 3);
        for (reward, loss) in &results {
            assert!(reward.is_finite());
            assert!(loss.is_finite());
        }
    }

    #[test]
    fn test_reinforce_with_baseline_mean() {
        let mut env = VecGridWorld { inner: GridWorld::new(3, 3) };
        let mut agent: ReinforceAgent<Vec<f64>> = ReinforceAgent::new(
            2, 4, 8, 0.99, 0.01, BaselineType::MeanReturn,
        );
        let results = agent.train(&mut env, 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_reinforce_with_learned_baseline() {
        let mut env = VecGridWorld { inner: GridWorld::new(3, 3) };
        let mut agent = ReinforceWithBaseline::<Vec<f64>>::new(
            2, 4, 8, 0.99, 0.01, 0.01,
        );
        let (reward, loss) = agent.train_episode(&mut env);
        assert!(reward.is_finite());
        assert!(loss.is_finite());
    }

    #[test]
    fn test_value_function() {
        let mut vf = ValueFunction::new(4, 8);
        let val = vf.predict(&[1.0, 0.0, 0.0, 0.0]);
        assert!(val.is_finite());
        vf.update(&[1.0, 0.0, 0.0, 0.0], 1.0, 0.01);
        let val_after = vf.predict(&[1.0, 0.0, 0.0, 0.0]);
        assert!(val_after.is_finite());
    }
}
