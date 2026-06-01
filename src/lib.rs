//! Advanced reinforcement learning algorithms.
//!
//! Includes DQN, policy gradients, actor-critic, PPO, TRPO, model-based RL,
//! multi-agent RL, hierarchical RL, and curiosity-driven exploration.

pub mod core;
pub mod dqn;
pub mod policy_gradient;
pub mod actor_critic;
pub mod ppo;
pub mod trpo;
pub mod model_based;
pub mod multi_agent;
pub mod hierarchical;
pub mod curiosity;
pub mod plato;

pub use core::*;
