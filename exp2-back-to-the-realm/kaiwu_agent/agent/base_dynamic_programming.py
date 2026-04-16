import kaiwu_agent
from kaiwu_agent.agent.base_agent import BaseAgent
import numpy as np
import copy
from kaiwu_agent.gorge_walk.utils import get_logger
import os
from kaiwu_agent.agent.base_agent import save_model_wrapper, load_model_wrapper, check_hasattr

class Agent(BaseAgent):
    @check_hasattr(["Q", "V", "policy", "gamma", "theta", "episodes", "state_size", "action_size", "algo"])
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
        super().__init__(agent_type, device, logger, monitor)
        self.agent_type = agent_type
        self.device = device
        self.logger = logger if logger else kaiwu_agent.logger
        self.monitor = monitor if monitor else kaiwu_agent.monitor

    def learn(self, F):
        episodes, gamma, theta, algo = self.episodes, self.gamma, self.theta, self.algo
        assert algo in ['policy_iteration', 'value_iteration'], "Invalid algorithm"

        if algo == 'policy_iteration':
            self.policy_iteration(F, episodes, gamma, theta)
        elif algo == 'value_iteration':
            self.value_iteration(F, episodes, gamma, theta)

    def policy_evaluation(self, policy, F, gamma, theta):
        """Calculate state-value array for the given policy

        Args:
            policy (np.array): policy array
            F (dict): transition function (state-action pair -> next state, reward, done)
            gamma (float): discount factor
            theta (float): threshold for convergence

        Returns:
            V (np.array): state-value array for the given policy
        """
        # Initialize state-value array (16,)
        V = np.zeros(self.state_size)
        delta = theta + 1

        while delta > theta:
            delta = 0
            # Loop over all states
            for state in range(self.state_size):
                v = 0
                # Loop over all actions fot the given state
                for action, action_prob in enumerate(policy[state]):
                    v += action_prob * self._get_value(state, action, F, gamma, V)

                # Calculate delta between old and new value for the given state
                delta = max(delta, abs(v - V[state]))

                # Update state-value array
                V[state] = v

        return V

    def q_value_iteration(self, V, F, gamma):
        """Calculate the Q value for all state-action pairs

        Args:
            V (np.array): array of state values obtained from policy evaluation
            F (dict): transition function (state-action pair -> next state, reward, done)
            gamma (float): discount factor

        Returns:
            Q (np.array): action-value array for the given state-action pair
        """
        Q = np.zeros([self.state_size, self.action_size])

        for state in range(self.state_size):
            for action in range(self.action_size):
                Q[state][action] = self._get_value(state, action, F, gamma, V)

        return Q

    def policy_improvement(self, Q):
        """Improve the policy based on action value (Q)

        Args:
            V (np.array): array of state values obtained from policy evaluation
            gamma (float): discount factor
        """
        # Blank policy initialized with zeros
        policy = np.zeros([self.state_size, self.action_size])

        for state in range(self.state_size):
            action_values = Q[state]

            # Update policy
            policy[state] = np.eye(self.action_size)[np.argmax(action_values)]

        return policy

    def policy_iteration(self, F, episodes, gamma, theta):
        """
        Calculate optimal policy using policy iteration \n

        Args: \n
            - F (dict): transition function (state-action pair -> next state, reward, done) \n
            - episodes (int): number of episodes \n
            - gamma (float): discount factor \n
            - theta (float): threshold for convergence \n

        Returns:
            - policy (np.array): optimal policy \n
            - V (np.array): optimal state-value array \n
        """
        policy = np.ones([self.state_size, self.action_size]) / self.action_size

        i = 0
        while i < episodes:
            V = self.policy_evaluation(policy, F, gamma, theta)
            Q = self.q_value_iteration(V, F, gamma)
            new_policy = self.policy_improvement(Q)

            if np.allclose(policy, new_policy, atol=1e-3):
                break

            policy = copy.copy(new_policy)

            if i % 10 == 0:
                self.logger.info("Iteration {}".format(i))
            i += 1

        self.policy, self.Q, self.V = policy, Q, V

        return policy, V

    def value_iteration(self, F, episodes, gamma, theta):
        """
        Calculate optimal policy using value iteration \n

        Args: \n
            - F (dict): transition function (state-action pair -> next state, reward, done) \n
            - episodes (int): number of episodes \n
            - gamma (float): discount factor \n
            - theta (float): threshold for convergence \n

        Returns:
            - policy (np.array): optimal policy \n
            - V (np.array): optimal state-value array \n
        """
        V = np.zeros(self.state_size)

        i = 0
        while i < episodes:
            delta = 0

            for state in range(self.state_size):
                v = V[state]

                V[state] = max(self._get_value(state, action, F, gamma, V) for action in range(self.action_size))

                delta = max(delta, abs(v - V[state]))

            if delta < theta:
                self.episodes = i
                break

            policy = self.policy_improvement(self.q_value_iteration(V, F, gamma))

            if i % 10 == 0:
                self.logger.info("Iteration {}".format(i))
            i += 1

        self.policy, self.V = policy, V

        return policy, V

    def _get_value(self, state, action, F, gamma, V):
        """Get value of the state-action pair

        Args:
            state (int): current state
            action (int): action taken
            F (dict): transition function (state-action pair -> next state, reward, done)
            gamma (float): discount factor
            V (np.array): state-value array

        Returns:
            value (float): value of the state-action pair
        """
        value = 0

        try:
            next_state, reward, _ = F[str(state)][str(action)]
            if reward == 0:
                reward = -1
            value = reward + gamma * V[next_state]
        except KeyError:
            pass

        return value

    @save_model_wrapper
    def save_model(self, path='/data/projects/Metagent/model/dynamic_programming', id='1'):
        if not os.path.exists(path):
            os.makedirs(path)
            self.logger.info(f"Create directory: {path}")

        np.save(f"{path}/model-{str(id)}", self.policy)
        self.save_checkpoint(path, id)
        os.system(f"touch {path}/model-{str(id)}.done")
        self.logger.info(f"Save model {path}/model-{str(id)}.npy successfully")

    @load_model_wrapper
    def load_model(self, path='/data/projects/Metagent/model/dynamic_programming', id='1'):
        if not os.path.exists(f"{path}/model-{str(id)}.done"):
            self.logger.error(f"File {path}/model-{str(id)}.done not found")
            exit(1)
        try:
            self.policy = np.load(f"{path}/model-{str(id)}.npy")
        except FileNotFoundError:
            self.logger.info(f"File {path}/model-{str(id)}.npy not found")
            exit(1)

    def save_checkpoint(self, path, id):
        file_path = f"{path}/checkpoint"
        model_checkpoint = "model_checkpoint_path: " + f"{path}/model-{id}.npy"
        all_model_checkpoint = f"all_model_checkpoint_path: {path}/model-{id}.npy"

        with open(file_path, 'w') as file:
            file.write(model_checkpoint + '\n')
            file.write(all_model_checkpoint + '\n')