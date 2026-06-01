//! Hierarchical RL: Options framework and feudal networks.

use crate::core::*;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// An option (temporally extended action) in the options framework.
pub struct HOption<S: Clone + Debug> {
    pub id: usize,
    pub initiation_set: Vec<S>,
    pub policy: Vec<(S, Action)>,
    pub termination_condition: fn(&S) -> bool,
    pub name: String,
}

impl<S: Clone + Debug + PartialEq> std::fmt::Debug for HOption<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HOption").field("id", &self.id).field("name", &self.name).finish()
    }
}

impl<S: Clone + Debug + PartialEq> Clone for HOption<S> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            initiation_set: self.initiation_set.clone(),
            policy: self.policy.clone(),
            termination_condition: self.termination_condition,
            name: self.name.clone(),
        }
    }
}

impl<S: Clone + Debug + PartialEq> HOption<S> {
    pub fn new(
        id: usize,
        name: String,
        initiation_set: Vec<S>,
        policy: Vec<(S, Action)>,
        termination: fn(&S) -> bool,
    ) -> Self {
        Self {
            id,
            name,
            initiation_set,
            policy,
            termination_condition: termination,
        }
    }

    pub fn can_initiate(&self, state: &S) -> bool {
        self.initiation_set.contains(state)
    }

    pub fn select_action(&self, state: &S) -> Option<Action> {
        self.policy
            .iter()
            .find(|(s, _)| s == state)
            .map(|(_, a)| *a)
    }

    pub fn should_terminate(&self, state: &S) -> bool {
        (self.termination_condition)(state)
    }
}

/// An intra-option Q-learning agent over options.
pub struct OptionsAgent<S: Clone + Debug + PartialEq> {
    pub q_table: Vec<Vec<f64>>,
    pub options: Vec<HOption<S>>,
    pub gamma: DiscountFactor,
    pub learning_rate: f64,
    pub epsilon: f64,
    pub current_option: Option<usize>,
}

impl<S: Clone + Debug + PartialEq> OptionsAgent<S> {
    pub fn new(
        num_states: usize,
        options: Vec<HOption<S>>,
        gamma: DiscountFactor,
    ) -> Self {
        let num_options = options.len();
        Self {
            q_table: vec![vec![0.0; num_options]; num_states],
            options,
            gamma,
            learning_rate: 0.1,
            epsilon: 0.1,
            current_option: None,
        }
    }

    pub fn select_option(&mut self, state_idx: usize, state: &S) -> usize {
        if let Some(opt_idx) = self.current_option {
            if !self.options[opt_idx].should_terminate(state) {
                return opt_idx;
            }
            self.current_option = None;
        }

        use rand::Rng;
        let mut rng = rand::thread_rng();
        let option_idx = if rng.gen::<f64>() < self.epsilon {
            rng.gen_range(0..self.options.len())
        } else {
            self.q_table[state_idx]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        };

        self.current_option = Some(option_idx);
        option_idx
    }

    pub fn get_action(&self, option_idx: usize, state: &S) -> Action {
        self.options[option_idx]
            .select_action(state)
            .unwrap_or(0)
    }

    pub fn update(
        &mut self,
        state_idx: usize,
        option_idx: usize,
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
        self.q_table[state_idx][option_idx] +=
            self.learning_rate * (td_target - self.q_table[state_idx][option_idx]);
    }
}

/// A manager (high-level) policy in feudal networks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeudalManager {
    pub network: NeuralNetwork,
    pub goal_size: usize,
}

impl FeudalManager {
    pub fn new(state_size: usize, goal_size: usize, hidden_size: usize) -> Self {
        Self {
            network: NeuralNetwork::new(&[state_size, hidden_size, goal_size]),
            goal_size,
        }
    }

    pub fn set_goal(&self, state: &[f64]) -> Vec<f64> {
        self.network.forward_linear(state)
    }
}

/// A worker (low-level) policy in feudal networks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeudalWorker {
    pub network: NeuralNetwork,
    pub num_actions: usize,
}

impl FeudalWorker {
    pub fn new(state_size: usize, goal_size: usize, num_actions: usize, hidden_size: usize) -> Self {
        Self {
            network: NeuralNetwork::new(&[state_size + goal_size, hidden_size, num_actions]),
            num_actions,
        }
    }

    pub fn select_action(&self, state: &[f64], goal: &[f64]) -> Action {
        let mut input = state.to_vec();
        input.extend_from_slice(goal);
        let q_values = self.network.forward_linear(&input);
        q_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Cosine similarity between direction and goal for intrinsic reward.
    pub fn intrinsic_reward(&self, prev_state: &[f64], state: &[f64], goal: &[f64]) -> f64 {
        let direction: Vec<f64> = state
            .iter()
            .zip(prev_state.iter())
            .map(|(s, ps)| s - ps)
            .collect();

        let dot: f64 = direction.iter().zip(goal.iter()).map(|(d, g)| d * g).sum();
        let dir_norm: f64 = direction.iter().map(|d| d * d).sum::<f64>().sqrt();
        let goal_norm: f64 = goal.iter().map(|g| g * g).sum::<f64>().sqrt();

        if dir_norm > 1e-8 && goal_norm > 1e-8 {
            dot / (dir_norm * goal_norm)
        } else {
            0.0
        }
    }
}

/// Full feudal network agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeudalAgent {
    pub manager: FeudalManager,
    pub worker: FeudalWorker,
    pub goal_update_freq: usize,
    pub manager_lr: f64,
    pub worker_lr: f64,
}

impl FeudalAgent {
    pub fn new(
        state_size: usize,
        goal_size: usize,
        num_actions: usize,
        hidden_size: usize,
    ) -> Self {
        Self {
            manager: FeudalManager::new(state_size, goal_size, hidden_size),
            worker: FeudalWorker::new(state_size, goal_size, num_actions, hidden_size),
            goal_update_freq: 10,
            manager_lr: 0.001,
            worker_lr: 0.001,
        }
    }

    pub fn select_action(&self, state: &[f64], goal: &[f64]) -> Action {
        self.worker.select_action(state, goal)
    }

    pub fn set_goal(&self, state: &[f64]) -> Vec<f64> {
        self.manager.set_goal(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn never_terminate(_state: &(usize, usize)) -> bool { false }
    fn terminate_at_goal(state: &(usize, usize)) -> bool { *state == (2, 2) }

    #[test]
    fn test_option_creation() {
        let option = HOption::new(0, "go_right".into(), vec![(0, 0), (1, 0)], vec![((0, 0), 1), ((1, 0), 1)], never_terminate);
        assert!(option.can_initiate(&(0, 0)));
        assert!(!option.can_initiate(&(2, 2)));
    }

    #[test]
    fn test_option_select_action() {
        let option = HOption::new(0, "go_right".into(), vec![(0, 0)], vec![((0, 0), 1)], never_terminate);
        assert_eq!(option.select_action(&(0, 0)), Some(1));
        assert_eq!(option.select_action(&(2, 2)), None);
    }

    #[test]
    fn test_option_termination() {
        let option = HOption::new(0, "go_to_goal".into(), vec![(0, 0)], vec![], terminate_at_goal);
        assert!(!option.should_terminate(&(0, 0)));
        assert!(option.should_terminate(&(2, 2)));
    }

    #[test]
    fn test_options_agent() {
        let options = vec![
            HOption::new(0, "right".into(), vec![(0,0)], vec![((0,0), 1)], never_terminate),
            HOption::new(1, "down".into(), vec![(0,0)], vec![((0,0), 2)], never_terminate),
        ];
        let mut agent = OptionsAgent::new(9, options, 0.99);
        let opt = agent.select_option(0, &(0, 0));
        assert!(opt < 2);
        let action = agent.get_action(opt, &(0, 0));
        assert!(action < 4);
    }

    #[test]
    fn test_options_agent_termination() {
        let options = vec![
            HOption::new(0, "right".into(), vec![(0,0), (1,0)], vec![((0,0), 1)], terminate_at_goal),
        ];
        let mut agent = OptionsAgent::new(9, options, 0.99);
        let _opt = agent.select_option(0, &(0, 0));
        // At goal, should terminate and select new option
        let _new_opt = agent.select_option(8, &(2, 2));
    }

    #[test]
    fn test_feudal_manager() {
        let manager = FeudalManager::new(4, 3, 8);
        let goal = manager.set_goal(&[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(goal.len(), 3);
    }

    #[test]
    fn test_feudal_worker() {
        let worker = FeudalWorker::new(4, 3, 4, 8);
        let action = worker.select_action(&[1.0, 0.0, 0.0, 0.0], &[0.5, -0.3, 0.1]);
        assert!(action < 4);
    }

    #[test]
    fn test_feudal_intrinsic_reward() {
        let worker = FeudalWorker::new(4, 3, 4, 8);
        let reward = worker.intrinsic_reward(&[0.0, 0.0, 0.0, 0.0], &[1.0, 0.0, 0.0, 0.0], &[1.0, 0.0, 0.0]);
        assert!(reward > 0.0);
    }

    #[test]
    fn test_feudal_agent() {
        let agent = FeudalAgent::new(4, 3, 4, 8);
        let goal = agent.set_goal(&[1.0, 0.0, 0.0, 0.0]);
        let action = agent.select_action(&[1.0, 0.0, 0.0, 0.0], &goal);
        assert!(action < 4);
    }

    #[test]
    fn test_options_agent_update() {
        let options = vec![
            HOption::new(0, "right".into(), vec![(0,0)], vec![((0,0), 1)], never_terminate),
        ];
        let mut agent = OptionsAgent::new(9, options, 0.99);
        agent.update(0, 0, 1.0, 1, false);
        assert!(agent.q_table[0][0] > 0.0);
    }
}
