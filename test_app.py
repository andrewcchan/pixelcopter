import unittest
import torch
import numpy as np
from app import PPOAgent, ActorNetwork, CriticNetwork

class TestNetworks(unittest.TestCase):
    def setUp(self):
        self.input_dim = 5
        self.output_dim = 2
        self.actor = ActorNetwork(self.input_dim, self.output_dim)
        self.critic = CriticNetwork(self.input_dim)
        self.test_input = torch.randn(1, self.input_dim)

    def test_actor_network(self):
        output = self.actor(self.test_input)
        self.assertEqual(output.shape, (1, self.output_dim))
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.allclose(torch.sum(output), torch.tensor(1.0)))

    def test_critic_network(self):
        output = self.critic(self.test_input)
        self.assertEqual(output.shape, (1, 1))

class TestPPOAgent(unittest.TestCase):
    def setUp(self):
        self.action_set = [119, None] # w and no action
        self.state_keys = ['player_y', 'player_vel', 'next_pipe_dist_to_player', 'next_pipe_top_y', 'next_pipe_bottom_y']
        self.agent = PPOAgent(allowed_actions=self.action_set, state_keys=self.state_keys, lr=1e-2)
        self.state = {key: np.random.rand() for key in self.state_keys}

    def test_normalize_state(self):
        state_vec = np.array([self.state[k] for k in self.state_keys], dtype=np.float32)
        self.agent.normalize_state(self.state)
        self.assertEqual(self.agent.state_count, 1)
        self.assertTrue(np.allclose(self.agent.running_state_mean, state_vec))

    def test_select_action(self):
        action, action_idx, log_prob = self.agent.select_action(self.state)
        self.assertIn(action, self.action_set)
        self.assertIsInstance(action_idx, torch.Tensor)
        self.assertIsInstance(log_prob, torch.Tensor)

    def test_store_transition(self):
        self.agent.store_transition(self.state, torch.tensor(0), torch.tensor(0.1), 1, False)
        self.assertEqual(len(self.agent.memory), 1)

    def test_update(self):
        # Fill memory with varied transitions
        for i in range(32): # Use a batch size
            state = {key: np.random.rand() for key in self.state_keys}
            action, action_idx, log_prob = self.agent.select_action(state)
            self.agent.store_transition(state, action_idx, log_prob, reward=float(i), done=(i==31))

        # Get initial actor and critic parameters
        initial_actor_params = {name: p.clone() for name, p in self.agent.actor.named_parameters()}
        initial_critic_params = {name: p.clone() for name, p in self.agent.critic.named_parameters()}

        # Perform an update
        self.agent.update()

        # Check that the parameters have been updated
        actor_params_updated = any(not torch.equal(p, initial_actor_params[name]) for name, p in self.agent.actor.named_parameters())
        self.assertTrue(actor_params_updated, "Actor parameters were not updated")

        critic_params_updated = any(not torch.equal(p, initial_critic_params[name]) for name, p in self.agent.critic.named_parameters())
        self.assertTrue(critic_params_updated, "Critic parameters were not updated")

        # Check that memory is cleared
        self.assertEqual(len(self.agent.memory), 0)

if __name__ == '__main__':
    unittest.main()
