//! Multi-agent RL: independent, shared, and communication-based protocols.

use crate::core::*;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// An agent identifier.
pub type AgentId = usize;

/// A message between agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessage {
    pub sender: AgentId,
    pub receiver: Option<AgentId>, // None = broadcast
    pub content: Vec<f64>,
    pub message_type: MessageType,
}

/// Types of inter-agent messages.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MessageType {
    Observation,
    Action,
    Reward,
    Policy,
    Custom(String),
}

/// A multi-agent environment step result.
#[derive(Debug, Clone)]
pub struct MultiAgentStep<S: Clone + Debug> {
    pub observations: Vec<S>,
    pub rewards: Vec<Reward>,
    pub dones: Vec<bool>,
    pub messages: Vec<AgentMessage>,
}

/// Multi-agent environment trait.
pub trait MultiAgentEnvironment: Clone + Debug {
    type State: Clone + Debug;

    fn num_agents(&self) -> usize;
    fn reset(&mut self) -> Vec<Self::State>;
    fn step(&mut self, actions: Vec<Action>) -> MultiAgentStep<Self::State>;
    fn num_actions_per_agent(&self) -> usize;
}

/// Independent Q-learning agent (no communication).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndependentQLearner {
    pub agent_id: AgentId,
    pub q_table: Vec<Vec<f64>>,
    pub epsilon: f64,
    pub gamma: DiscountFactor,
    pub learning_rate: f64,
}

impl IndependentQLearner {
    pub fn new(agent_id: AgentId, num_states: usize, num_actions: usize, gamma: DiscountFactor) -> Self {
        Self {
            agent_id,
            q_table: vec![vec![0.0; num_actions]; num_states],
            epsilon: 0.1,
            gamma,
            learning_rate: 0.1,
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

    pub fn update(&mut self, state_idx: usize, action: Action, reward: f64, next_state_idx: usize, done: bool) {
        let max_next_q = if done {
            0.0
        } else {
            self.q_table[next_state_idx]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
        };
        let td_target = reward + self.gamma * max_next_q;
        self.q_values_mut(state_idx)[action] += self.learning_rate * (td_target - self.q_table[state_idx][action]);
    }

    fn q_values_mut(&mut self, state: usize) -> &mut Vec<f64> {
        &mut self.q_table[state]
    }
}

/// Shared policy agent (centralized training, decentralized execution).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedPolicyAgent {
    pub policy: NeuralNetwork,
    pub agent_id: AgentId,
    pub num_actions: usize,
    pub gamma: DiscountFactor,
    pub learning_rate: f64,
}

impl SharedPolicyAgent {
    pub fn new(agent_id: AgentId, state_size: usize, num_actions: usize, hidden_size: usize) -> Self {
        Self {
            policy: NeuralNetwork::new(&[state_size, hidden_size, num_actions]),
            agent_id,
            num_actions,
            gamma: 0.99,
            learning_rate: 0.01,
        }
    }

    pub fn select_action(&self, state: &[f64]) -> Action {
        let logits = self.policy.forward_linear(state);
        let probs = softmax(&logits);
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
}

/// A communicating agent with message passing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicatingAgent {
    pub agent_id: AgentId,
    pub q_table: Vec<Vec<f64>>,
    pub inbox: Vec<AgentMessage>,
    pub communication_range: usize,
    pub epsilon: f64,
    pub gamma: DiscountFactor,
    pub learning_rate: f64,
}

impl CommunicatingAgent {
    pub fn new(
        agent_id: AgentId,
        num_states: usize,
        num_actions: usize,
        communication_range: usize,
        gamma: DiscountFactor,
    ) -> Self {
        Self {
            agent_id,
            q_table: vec![vec![0.0; num_actions]; num_states],
            inbox: Vec::new(),
            communication_range,
            epsilon: 0.1,
            gamma,
            learning_rate: 0.1,
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

    pub fn send_message(&self, state_idx: usize, action: Action) -> AgentMessage {
        let mut content = vec![0.0; self.q_table.len()];
        if state_idx < content.len() {
            content[state_idx] = 1.0;
        }
        AgentMessage {
            sender: self.agent_id,
            receiver: None, // broadcast
            content,
            message_type: MessageType::Observation,
        }
    }

    pub fn receive_message(&mut self, message: AgentMessage) {
        self.inbox.push(message);
    }

    /// Incorporate received information into Q-values.
    pub fn process_messages(&mut self) {
        for msg in self.inbox.drain(..) {
            // Use message content to update Q-values
            let max_idx = msg
                .content
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            // Small update based on communicated info
            if max_idx < self.q_table.len() {
                for q_val in &mut self.q_table[max_idx] {
                    *q_val += 0.01 * (msg.sender as f64 * 0.1);
                }
            }
        }
    }
}

/// Multi-agent coordinator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAgentCoordinator {
    pub agents: Vec<IndependentQLearner>,
    pub shared_policy: Option<SharedPolicyAgent>,
    pub communicating_agents: Vec<CommunicatingAgent>,
}

impl MultiAgentCoordinator {
    pub fn new_independent(num_agents: usize, num_states: usize, num_actions: usize, gamma: DiscountFactor) -> Self {
        let agents = (0..num_agents)
            .map(|id| IndependentQLearner::new(id, num_states, num_actions, gamma))
            .collect();
        Self {
            agents,
            shared_policy: None,
            communicating_agents: Vec::new(),
        }
    }

    pub fn new_communicating(
        num_agents: usize,
        num_states: usize,
        num_actions: usize,
        communication_range: usize,
        gamma: DiscountFactor,
    ) -> Self {
        let communicating_agents = (0..num_agents)
            .map(|id| {
                CommunicatingAgent::new(id, num_states, num_actions, communication_range, gamma)
            })
            .collect();
        Self {
            agents: Vec::new(),
            shared_policy: None,
            communicating_agents,
        }
    }

    /// Run one step of independent agents.
    pub fn step_independent(&mut self, state_indices: Vec<usize>) -> Vec<Action> {
        self.agents
            .iter()
            .zip(state_indices.iter())
            .map(|(agent, &state)| agent.select_action(state))
            .collect()
    }

    /// Update independent agents.
    pub fn update_independent(
        &mut self,
        state_indices: Vec<usize>,
        actions: Vec<Action>,
        rewards: Vec<Reward>,
        next_state_indices: Vec<usize>,
        dones: Vec<bool>,
    ) {
        for (agent, (s, a, r, ns, d)) in self.agents.iter_mut().zip(
            state_indices
                .iter()
                .zip(actions.iter())
                .zip(rewards.iter())
                .zip(next_state_indices.iter())
                .zip(dones.iter())
                .map(|((((s, a), r), ns), d)| (*s, *a, *r, *ns, *d)),
        ) {
            agent.update(s, a, r, ns, d);
        }
    }

    /// Run communicating agents with message exchange.
    pub fn step_communicating(&mut self, state_indices: Vec<usize>) -> Vec<Action> {
        // First, have each agent broadcast
        let messages: Vec<AgentMessage> = self
            .communicating_agents
            .iter()
            .zip(state_indices.iter())
            .map(|(agent, &state)| agent.send_message(state, 0))
            .collect();

        // Deliver messages to all agents
        for msg in &messages {
            for agent in &mut self.communicating_agents {
                if msg.sender != agent.agent_id {
                    agent.receive_message(msg.clone());
                }
            }
        }

        // Process and select actions
        for agent in &mut self.communicating_agents {
            agent.process_messages();
        }

        self.communicating_agents
            .iter()
            .zip(state_indices.iter())
            .map(|(agent, &state)| agent.select_action(state))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_independent_q_learner() {
        let mut agent = IndependentQLearner::new(0, 10, 4, 0.99);
        let action = agent.select_action(0);
        assert!(action < 4);
        agent.update(0, action, 1.0, 1, false);
        assert!(agent.q_table[0][action] > 0.0);
    }

    #[test]
    fn test_independent_q_learner_greedy() {
        let mut agent = IndependentQLearner::new(0, 5, 3, 0.99);
        agent.epsilon = 0.0;
        agent.q_table[0][2] = 10.0;
        let action = agent.select_action(0);
        assert_eq!(action, 2);
    }

    #[test]
    fn test_shared_policy_agent() {
        let agent = SharedPolicyAgent::new(0, 4, 3, 8);
        let action = agent.select_action(&[1.0, 0.0, 0.0, 0.0]);
        assert!(action < 3);
    }

    #[test]
    fn test_communicating_agent() {
        let mut agent = CommunicatingAgent::new(0, 10, 4, 3, 0.99);
        let action = agent.select_action(0);
        assert!(action < 4);

        let msg = agent.send_message(0, action);
        assert_eq!(msg.sender, 0);
        assert_eq!(msg.message_type, MessageType::Observation);
    }

    #[test]
    fn test_communicating_agent_receive() {
        let mut agent1 = CommunicatingAgent::new(0, 10, 4, 3, 0.99);
        let mut agent2 = CommunicatingAgent::new(1, 10, 4, 3, 0.99);

        let msg = agent1.send_message(0, 1);
        agent2.receive_message(msg);
        assert!(!agent2.inbox.is_empty());
        agent2.process_messages();
        assert!(agent2.inbox.is_empty());
    }

    #[test]
    fn test_multi_agent_coordinator_independent() {
        let mut coord = MultiAgentCoordinator::new_independent(3, 10, 4, 0.99);
        let actions = coord.step_independent(vec![0, 1, 2]);
        assert_eq!(actions.len(), 3);
        for a in &actions {
            assert!(*a < 4);
        }
    }

    #[test]
    fn test_multi_agent_coordinator_update() {
        let mut coord = MultiAgentCoordinator::new_independent(2, 10, 4, 0.99);
        let actions = coord.step_independent(vec![0, 1]);
        coord.update_independent(vec![0, 1], actions, vec![1.0, 0.5], vec![1, 2], vec![false, false]);
    }

    #[test]
    fn test_multi_agent_coordinator_communicating() {
        let mut coord = MultiAgentCoordinator::new_communicating(3, 10, 4, 2, 0.99);
        let actions = coord.step_communicating(vec![0, 1, 2]);
        assert_eq!(actions.len(), 3);
    }

    #[test]
    fn test_agent_message_serialization() {
        let msg = AgentMessage {
            sender: 0,
            receiver: Some(1),
            content: vec![1.0, 0.0],
            message_type: MessageType::Action,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: AgentMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.sender, 0);
        assert_eq!(deserialized.message_type, MessageType::Action);
    }
}
