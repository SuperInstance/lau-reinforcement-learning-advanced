//! PLATO-specific applications: advanced agent training for rooms and fleet coordination.

use crate::core::*;
use crate::multi_agent::*;
use serde::{Deserialize, Serialize};

/// A room in the PLATO system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatoRoom {
    pub id: String,
    pub capacity: usize,
    pub current_occupants: usize,
    pub temperature: f64,
    pub light_level: f64,
    pub resources: Vec<f64>,
}

impl PlatoRoom {
    pub fn new(id: impl Into<String>, capacity: usize) -> Self {
        Self {
            id: id.into(),
            capacity,
            current_occupants: 0,
            temperature: 22.0,
            light_level: 0.5,
            resources: vec![1.0; 5],
        }
    }

    pub fn occupancy_ratio(&self) -> f64 {
        self.current_occupants as f64 / self.capacity.max(1) as f64
    }

    pub fn is_full(&self) -> bool {
        self.current_occupants >= self.capacity
    }

    pub fn state_vector(&self) -> Vec<f64> {
        let mut state = vec![
            self.current_occupants as f64 / self.capacity.max(1) as f64,
            self.temperature / 30.0,
            self.light_level,
        ];
        state.extend_from_slice(&self.resources);
        state
    }

    pub fn state_size() -> usize {
        8 // 3 base + 5 resources
    }
}

/// A fleet agent operating in the PLATO environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetAgent {
    pub id: String,
    pub current_room: Option<String>,
    pub assigned_tasks: Vec<String>,
    pub energy: f64,
    pub capabilities: Vec<f64>,
}

impl FleetAgent {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            current_room: None,
            assigned_tasks: Vec::new(),
            energy: 1.0,
            capabilities: vec![0.5; 4],
        }
    }

    pub fn with_capabilities(mut self, caps: Vec<f64>) -> Self {
        self.capabilities = caps;
        self
    }

    pub fn state_vector(&self) -> Vec<f64> {
        let mut state = vec![self.energy];
        state.extend_from_slice(&self.capabilities);
        state
    }

    pub fn state_size() -> usize {
        5 // energy + 4 capabilities
    }
}

/// PLATO environment for room management.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatoEnvironment {
    pub rooms: Vec<PlatoRoom>,
    pub agents: Vec<FleetAgent>,
    pub time_step: usize,
    pub max_steps: usize,
}

impl PlatoEnvironment {
    pub fn new(rooms: Vec<PlatoRoom>, agents: Vec<FleetAgent>, max_steps: usize) -> Self {
        Self {
            rooms,
            agents,
            time_step: 0,
            max_steps,
        }
    }

    /// Get the full state vector.
    pub fn full_state(&self) -> Vec<f64> {
        let mut state = Vec::new();
        for room in &self.rooms {
            state.extend(room.state_vector());
        }
        for agent in &self.agents {
            state.extend(agent.state_vector());
        }
        state
    }

    /// Compute reward: balance occupancy across rooms, maintain comfort.
    pub fn compute_reward(&self) -> f64 {
        if self.rooms.is_empty() {
            return 0.0;
        }

        // Reward for balanced occupancy
        let avg_occupancy: f64 =
            self.rooms.iter().map(|r| r.occupancy_ratio()).sum::<f64>() / self.rooms.len() as f64;
        let occupancy_variance: f64 = self.rooms
            .iter()
            .map(|r| (r.occupancy_ratio() - avg_occupancy).powi(2))
            .sum::<f64>()
            / self.rooms.len() as f64;
        let balance_reward = -occupancy_variance;

        // Reward for comfort (temperature near 22, light near 0.7)
        let comfort_reward: f64 = self.rooms
            .iter()
            .map(|r| {
                let temp_comfort = -(r.temperature - 22.0).powi(2) * 0.1;
                let light_comfort = -(r.light_level - 0.7).powi(2) * 0.1;
                temp_comfort + light_comfort
            })
            .sum::<f64>()
            / self.rooms.len() as f64;

        balance_reward + comfort_reward
    }

    /// Step: apply actions (move agent to room, adjust environment).
    pub fn step(&mut self, actions: Vec<usize>) -> (Vec<f64>, f64, bool) {
        // Move agents to rooms
        for (i, &action) in actions.iter().enumerate() {
            if i < self.agents.len() && action < self.rooms.len() {
                // Remove from old room
                if let Some(ref room_id) = self.agents[i].current_room {
                    if let Some(room) = self.rooms.iter_mut().find(|r| r.id == *room_id) {
                        room.current_occupants = room.current_occupants.saturating_sub(1);
                    }
                }
                // Add to new room
                let room = &mut self.rooms[action];
                if !room.is_full() {
                    room.current_occupants += 1;
                    self.agents[i].current_room = Some(room.id.clone());
                }
                self.agents[i].energy = (self.agents[i].energy - 0.01).max(0.0);
            }
        }

        self.time_step += 1;
        let reward = self.compute_reward();
        let done = self.time_step >= self.max_steps;

        (self.full_state(), reward, done)
    }

    pub fn reset(&mut self) -> Vec<f64> {
        self.time_step = 0;
        for room in &mut self.rooms {
            room.current_occupants = 0;
        }
        for agent in &mut self.agents {
            agent.current_room = None;
            agent.energy = 1.0;
        }
        self.full_state()
    }

    pub fn num_actions(&self) -> usize {
        self.rooms.len()
    }
}

/// A PLATO room agent using Q-learning for room management.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatoRoomAgent {
    pub q_table: Vec<Vec<f64>>,
    pub epsilon: f64,
    pub gamma: f64,
    pub learning_rate: f64,
    pub room_id: String,
}

impl PlatoRoomAgent {
    pub fn new(room_id: impl Into<String>, num_states: usize, num_actions: usize) -> Self {
        Self {
            q_table: vec![vec![0.0; num_actions]; num_states],
            epsilon: 0.1,
            gamma: 0.99,
            learning_rate: 0.1,
            room_id: room_id.into(),
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
        let max_next_q = if done { 0.0 } else {
            self.q_table[next_state_idx].iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        };
        let td_target = reward + self.gamma * max_next_q;
        self.q_table[state_idx][action] += self.learning_rate * (td_target - self.q_table[state_idx][action]);
    }
}

/// Fleet coordinator using multi-agent RL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetCoordinator {
    pub agents: Vec<PlatoRoomAgent>,
    pub env: PlatoEnvironment,
}

impl FleetCoordinator {
    pub fn new(env: PlatoEnvironment, num_states: usize) -> Self {
        let num_actions = env.num_actions();
        let agents = (0..env.agents.len())
            .map(|i| PlatoRoomAgent::new(format!("agent_{}", i), num_states, num_actions))
            .collect();
        Self { agents, env }
    }

    /// Run one training episode.
    pub fn train_episode(&mut self) -> f64 {
        let state = self.env.reset();
        let state_idx = Self::discretize_state(&state);
        let mut total_reward = 0.0;

        loop {
            let actions: Vec<Action> = self
                .agents
                .iter()
                .map(|a| a.select_action(state_idx))
                .collect();

            let (next_state_vec, reward, done) = self.env.step(actions);
            let next_state_idx = Self::discretize_state(&next_state_vec);
            total_reward += reward;

            for agent in &mut self.agents {
                agent.update(state_idx, 0, reward, next_state_idx, done);
            }

            if done {
                break;
            }
        }

        total_reward
    }

    /// Simple state discretization.
    fn discretize_state(state: &[f64]) -> usize {
        let hash: f64 = state.iter().enumerate().map(|(i, s)| (i as f64 + 1.0) * s).sum();
        (hash.abs() * 100.0) as usize % 100
    }

    /// Get current assignment of agents to rooms.
    pub fn get_assignments(&self) -> Vec<(String, Option<String>)> {
        self.agents
            .iter()
            .zip(self.env.agents.iter())
            .map(|(agent, fleet)| (agent.room_id.clone(), fleet.current_room.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plato_room() {
        let mut room = PlatoRoom::new("room_1", 10);
        assert_eq!(room.occupancy_ratio(), 0.0);
        assert!(!room.is_full());
        room.current_occupants = 10;
        assert!(room.is_full());
        let state = room.state_vector();
        assert_eq!(state.len(), PlatoRoom::state_size());
    }

    #[test]
    fn test_fleet_agent() {
        let agent = FleetAgent::new("agent_1").with_capabilities(vec![0.8, 0.6, 0.9, 0.7]);
        let state = agent.state_vector();
        assert_eq!(state.len(), FleetAgent::state_size());
        assert!((state[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_plato_environment() {
        let rooms = vec![
            PlatoRoom::new("r1", 5),
            PlatoRoom::new("r2", 10),
        ];
        let agents = vec![
            FleetAgent::new("a1"),
            FleetAgent::new("a2"),
        ];
        let mut env = PlatoEnvironment::new(rooms, agents, 100);
        let state = env.reset();
        assert!(!state.is_empty());
        let (next_state, reward, done) = env.step(vec![0, 1]);
        assert!(reward.is_finite());
        assert!(!done);
    }

    #[test]
    fn test_plato_environment_reward() {
        let rooms = vec![
            PlatoRoom::new("r1", 5),
            PlatoRoom::new("r2", 5),
        ];
        let agents = vec![FleetAgent::new("a1")];
        let env = PlatoEnvironment::new(rooms, agents, 10);
        let reward = env.compute_reward();
        assert!(reward.is_finite());
    }

    #[test]
    fn test_plato_room_agent() {
        let mut agent = PlatoRoomAgent::new("r1", 20, 3);
        let action = agent.select_action(0);
        assert!(action < 3);
        agent.update(0, action, 1.0, 1, false);
        assert!(agent.q_table[0][action] > 0.0);
    }

    #[test]
    fn test_fleet_coordinator() {
        let rooms = vec![
            PlatoRoom::new("r1", 5),
            PlatoRoom::new("r2", 5),
        ];
        let agents = vec![FleetAgent::new("a1"), FleetAgent::new("a2")];
        let env = PlatoEnvironment::new(rooms, agents, 10);
        let mut coord = FleetCoordinator::new(env, 100);
        let reward = coord.train_episode();
        assert!(reward.is_finite());
    }

    #[test]
    fn test_fleet_coordinator_assignments() {
        let rooms = vec![PlatoRoom::new("r1", 5)];
        let agents = vec![FleetAgent::new("a1")];
        let env = PlatoEnvironment::new(rooms, agents, 5);
        let coord = FleetCoordinator::new(env, 20);
        let assignments = coord.get_assignments();
        assert_eq!(assignments.len(), 1);
    }

    #[test]
    fn test_plato_room_serialization() {
        let room = PlatoRoom::new("test_room", 20);
        let json = serde_json::to_string(&room).unwrap();
        let deserialized: PlatoRoom = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, "test_room");
        assert_eq!(deserialized.capacity, 20);
    }
}
