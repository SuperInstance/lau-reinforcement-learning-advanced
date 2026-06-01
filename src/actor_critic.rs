//! Actor-Critic methods: A2C and A3C with parallel environments.

use crate::core::*;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Actor-Critic network combining policy and value functions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActorCriticNetwork {
    pub shared: NeuralNetwork,
    pub policy_head: Layer,
    pub value_head: Layer,
}

impl ActorCriticNetwork {
    pub fn new(state_size: usize, num_actions: usize, hidden_size: usize) -> Self {
        let shared = NeuralNetwork::new(&[state_size, hidden_size]);
        let policy_head = Layer::new(hidden_size, num_actions);
        let value_head = Layer::new(hidden_size, 1);
        Self {
            shared,
            policy_head,
            value_head,
        }
    }

    pub fn forward(&self, state: &[f64]) -> (Vec<f64>, f64) {
        let shared_out = self.shared.forward(state);
        let policy_logits = self.policy_head.forward(&shared_out);
        let probs = softmax(&policy_logits);
        let value = self.value_head.forward(&shared_out)[0];
        (probs, value)
    }

    pub fn select_action(&self, state: &[f64]) -> (Action, f64) {
        let (probs, value) = self.forward(state);
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
        (action, value)
    }

    /// Simple parameter update.
    pub fn update(
        &mut self,
        state: &[f64],
        action: Action,
        advantage: f64,
        value_target: f64,
        policy_lr: f64,
        value_lr: f64,
    ) {
        let (probs, value) = self.forward(state);
        let value_error = value_target - value;

        // Update value head
        for row in &mut self.value_head.weights {
            for w in row.iter_mut() {
                *w += value_lr * value_error * 0.01;
            }
        }
        if let Some(b) = self.value_head.biases.first_mut() {
            *b += value_lr * value_error * 0.01;
        }

        // Update policy head
        for (j, row) in self.policy_head.weights.iter_mut().enumerate() {
            for w in row.iter_mut() {
                if j == action {
                    *w += policy_lr * advantage * (1.0 - probs[action]);
                } else if j < probs.len() {
                    *w -= policy_lr * advantage * probs[j];
                }
            }
        }
    }
}

/// A2C (Advantage Actor-Critic) agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct A2CAgent<S: Clone + Debug> {
    pub network: ActorCriticNetwork,
    pub gamma: DiscountFactor,
    pub policy_lr: f64,
    pub value_lr: f64,
    pub entropy_coeff: f64,
    pub n_steps: usize,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: Clone + Debug + Into<Vec<f64>>> A2CAgent<S> {
    pub fn new(
        state_size: usize,
        num_actions: usize,
        hidden_size: usize,
        gamma: DiscountFactor,
        policy_lr: f64,
        value_lr: f64,
    ) -> Self {
        let network = ActorCriticNetwork::new(state_size, num_actions, hidden_size);
        Self {
            network,
            gamma,
            policy_lr,
            value_lr,
            entropy_coeff: 0.01,
            n_steps: 5,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Collect n-step rollout.
    fn collect_rollout<E: Environment<State = S>>(
        &self,
        env: &mut E,
        n_steps: usize,
    ) -> (Vec<Vec<f64>>, Vec<Action>, Vec<Reward>, Vec<f64>, bool) {
        let mut states = Vec::new();
        let mut actions = Vec::new();
        let mut rewards = Vec::new();
        let mut values = Vec::new();

        let mut state = env.reset();
        let mut done = false;

        for _ in 0..n_steps {
            if done {
                break;
            }
            let state_vec: Vec<f64> = state.clone().into();
            let (action, value) = self.network.select_action(&state_vec);
            states.push(state_vec);
            actions.push(action);
            values.push(value);

            let (next_state, reward, d) = env.step(action);
            rewards.push(reward);
            state = next_state;
            done = d;
        }

        (states, actions, rewards, values, done)
    }

    /// Train for one update using n-step returns.
    pub fn train_step<E: Environment<State = S>>(&mut self, env: &mut E) -> (Reward, f64) {
        let (states, actions, rewards, values, _done) =
            self.collect_rollout(env, self.n_steps);

        if states.is_empty() {
            return (0.0, 0.0);
        }

        let total_reward: Reward = rewards.iter().sum();
        let mut total_loss = 0.0;

        // Compute n-step returns
        let mut returns = Vec::new();
        let mut g = 0.0;
        for &r in rewards.iter().rev() {
            g = r + self.gamma * g;
            returns.push(g);
        }
        returns.reverse();

        for (i, (state_vec, &action)) in states.iter().zip(actions.iter()).enumerate() {
            let advantage = returns[i] - values[i];
            self.network.update(
                state_vec,
                action,
                advantage,
                returns[i],
                self.policy_lr,
                self.value_lr,
            );
            total_loss += advantage.powi(2);
        }

        let avg_loss = total_loss / states.len() as f64;
        (total_reward, avg_loss)
    }

    /// Train for multiple updates.
    pub fn train<E: Environment<State = S>>(
        &mut self,
        env: &mut E,
        num_updates: usize,
    ) -> Vec<(Reward, f64)> {
        (0..num_updates)
            .map(|_| self.train_step(env))
            .collect()
    }
}

/// A3C-style agent (simulated parallel workers, single-threaded).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct A3CAgent<S: Clone + Debug> {
    pub network: ActorCriticNetwork,
    pub num_workers: usize,
    pub gamma: DiscountFactor,
    pub policy_lr: f64,
    pub value_lr: f64,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: Clone + Debug + Into<Vec<f64>>> A3CAgent<S> {
    pub fn new(
        state_size: usize,
        num_actions: usize,
        hidden_size: usize,
        num_workers: usize,
        gamma: DiscountFactor,
    ) -> Self {
        let network = ActorCriticNetwork::new(state_size, num_actions, hidden_size);
        Self {
            network,
            num_workers,
            gamma,
            policy_lr: 0.001,
            value_lr: 0.001,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Simulate parallel rollouts and aggregate gradients.
    pub fn train_round<E: Environment<State = S> + Clone>(
        &mut self,
        env_template: &E,
        n_steps: usize,
    ) -> (Reward, f64) {
        let mut all_states = Vec::new();
        let mut all_actions = Vec::new();
        let mut all_advantages = Vec::new();
        let mut all_returns = Vec::new();
        let mut total_reward = 0.0;

        for _ in 0..self.num_workers {
            let mut env = env_template.clone();
            let mut state = env.reset();
            let mut worker_states = Vec::new();
            let mut worker_actions = Vec::new();
            let mut worker_rewards = Vec::new();
            let mut worker_values = Vec::new();

            for _ in 0..n_steps {
                let state_vec: Vec<f64> = state.clone().into();
                let (action, value) = self.network.select_action(&state_vec);
                let (next_state, reward, done) = env.step(action);

                worker_states.push(state_vec);
                worker_actions.push(action);
                worker_rewards.push(reward);
                worker_values.push(value);
                total_reward += reward;

                state = next_state;
                if done {
                    break;
                }
            }

            let returns = compute_returns(&worker_rewards, self.gamma);
            for (i, (s, &a)) in worker_states.iter().zip(worker_actions.iter()).enumerate() {
                all_states.push(s.clone());
                all_actions.push(a);
                all_advantages.push(returns[i] - worker_values[i]);
                all_returns.push(returns[i]);
            }
        }

        let mut total_loss = 0.0;
        for (i, (state_vec, &action)) in all_states.iter().zip(all_actions.iter()).enumerate() {
            self.network.update(
                state_vec,
                action,
                all_advantages[i],
                all_returns[i],
                self.policy_lr / self.num_workers as f64,
                self.value_lr / self.num_workers as f64,
            );
            total_loss += all_advantages[i].powi(2);
        }

        let avg_reward = total_reward / self.num_workers as f64;
        let avg_loss = if !all_states.is_empty() {
            total_loss / all_states.len() as f64
        } else {
            0.0
        };

        (avg_reward, avg_loss)
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
    fn test_actor_critic_network() {
        let ac = ActorCriticNetwork::new(4, 3, 8);
        let (probs, value) = ac.forward(&[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(probs.len(), 3);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(value.is_finite());
    }

    #[test]
    fn test_actor_critic_select_action() {
        let ac = ActorCriticNetwork::new(4, 3, 8);
        let (action, value) = ac.select_action(&[0.0, 1.0, 0.0, 0.0]);
        assert!(action < 3);
        assert!(value.is_finite());
    }

    #[test]
    fn test_a2c_agent() {
        let mut env = VecGridWorld { inner: GridWorld::new(3, 3) };
        let mut agent: A2CAgent<Vec<f64>> = A2CAgent::new(2, 4, 8, 0.99, 0.01, 0.01);
        let results = agent.train(&mut env, 5);
        assert_eq!(results.len(), 5);
        for (reward, loss) in &results {
            assert!(reward.is_finite());
            assert!(loss.is_finite());
        }
    }

    #[test]
    fn test_a3c_agent() {
        let env = VecGridWorld { inner: GridWorld::new(3, 3) };
        let mut agent: A3CAgent<Vec<f64>> = A3CAgent::new(2, 4, 8, 2, 0.99);
        let (avg_reward, avg_loss) = agent.train_round(&env, 5);
        assert!(avg_reward.is_finite());
        assert!(avg_loss.is_finite());
    }

    #[test]
    fn test_a2c_n_steps() {
        let mut env = VecGridWorld { inner: GridWorld::new(3, 3) };
        let mut agent: A2CAgent<Vec<f64>> = A2CAgent::new(2, 4, 8, 0.99, 0.01, 0.01);
        agent.n_steps = 10;
        let (reward, _loss) = agent.train_step(&mut env);
        assert!(reward.is_finite());
    }
}
