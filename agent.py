import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# dealer showing is from 1 - 10
# our states are from 1 - 21
class Agent:
    def __init__(self, no: int = 100, lam: int = 0.5) -> None:
        self.Q = np.zeros((22, 11, 2))
        self.N = np.zeros((22, 11, 2))
        self.E = np.zeros((22, 11, 2))
        self.no = no
        self.lam = lam
        self.action_names = ["hit", "stick"]
        self.wins = 0
        self.iterations = 0

    def epsilon_greedy_argmax(self, player_sum, dealer_showing):
        if player_sum < 0 or player_sum >= 22:
            return 0

        q_values = self.Q[player_sum, dealer_showing]
        epsilon = self.no / (self.no + self.N[player_sum, dealer_showing].sum())
        # choose random with prob epsilon
        if random.random() < epsilon:
            return random.choice([0, 1])

        return q_values.argmax()

    # sarsa update
    def online_update(
        self, state, action_idx, reward, next_state, next_action_idx, done
    ):
        return

    def train(self, env):
        state = env.reset()
        action_idx = self.epsilon_greedy_argmax(*state)
        episode = []
        self.E *= 0.0
        done = False
        while not done:
            player_sum, dealer_showing = state
            action = self.action_names[action_idx]
            self.N[player_sum, dealer_showing, action_idx] += 1
            next_state, reward, done = env.step(state, action)
            episode.append([state, action_idx, reward])
            next_action_idx = self.epsilon_greedy_argmax(*next_state)
            self.online_update(
                state, action_idx, reward, next_state, next_action_idx, done
            )
            state, action_idx = next_state, next_action_idx

        if episode[-1][-1] == 1:
            self.wins += 1
        self.iterations += 1

        self.offline_update(episode)

        return episode

    def offline_update(self, episode: list):
        return

    def plot_value_function(self, title):
        _, ax = plt.subplots(subplot_kw={"projection": "3d"})
        plt.title(title)
        x_range, y_range = np.arange(1, 22), np.arange(1, 11)
        X, Y = np.meshgrid(x_range, y_range)
        Z = self.Q.max(axis=2)[X, Y]
        ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
        plt.xlabel("Player sum")
        plt.ylabel("Dealer showing")
        ax.view_init(30, 120)


class MCControl(Agent):
    def offline_update(self, episode: list):
        """Updates q values based on episode = [s, a, r, ... s, a, r]"""
        run_sum = 0

        for (player_sum, dealer_showing), action_idx, reward in reversed(episode):
            run_sum += reward
            # move Q[s, a] towards run_sum
            self.Q[player_sum, dealer_showing, action_idx] += (
                run_sum - self.Q[player_sum, dealer_showing, action_idx]
            ) / self.N[player_sum, dealer_showing, action_idx]


class Sarsa(Agent):
    def online_update(
        self, state, action_idx, reward, next_state, next_action_idx, done
    ):
        next_q = 0 if done else self.Q[next_state[0], next_state[1], next_action_idx]
        curr_q = self.Q[state[0], state[1], action_idx]

        delta = reward + (next_q - curr_q)
        alpha = 1.0 / self.N[state[0], state[1], action_idx]
        self.E *= self.lam
        self.E[state[0], state[1], action_idx] += 1
        self.Q = self.Q + alpha * delta * self.E


class LinearFunctionApproxator(Agent):
    def __init__(self, no: int = 100, lam: int = 0.5) -> None:
        super().__init__(no, lam)
        self.theta = np.random.random(36)
        self.player_buckets = [[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]]
        self.dealer_buckets = [[1, 4], [4, 7], [7, 10]]
        self.epsilon = 5e-2
        self.alpha = 1e-2
        self.E = np.zeros_like(self.theta)

    def phi(self, state, action_idx):
        try:
            result = np.zeros((6, 3, 2), dtype=float)
            if state[0] <= 0 or state[1] <= 0:
                return result.flatten()
            player_idx = [a <= state[0] <= b for a, b in self.player_buckets]
            dealer_idx = [a <= state[1] <= b for a, b in self.dealer_buckets]
            for idx in np.where(player_idx):
                for jdx in np.where(dealer_idx):
                    result[idx, jdx, action_idx] = 1
        except:
            print(state, action_idx)
            raise
        return result.flatten()

    def q_function(self, state, action):
        return np.dot(self.phi(state, action), self.theta)

    def online_update(
        self, state, action_idx, reward, next_state, next_action_idx, done
    ):
        next_q = 0 if done else self.q_function(next_state, next_action_idx)
        curr_q = self.q_function(state, action_idx)

        delta = reward + (next_q - curr_q)

        self.E *= self.lam
        self.E += self.phi(state, action_idx)
        self.theta += self.alpha * delta * self.E

    def epsilon_greedy_argmax(self, player_sum, dealer_showing):
        if player_sum < 0 or player_sum >= 22:
            return 0

        q_values = np.array(
            [
                self.q_function((player_sum, dealer_showing), 0),
                self.q_function((player_sum, dealer_showing), 1),
            ]
        )

        # choose random with prob epsilon
        if random.random() < self.epsilon:
            return random.choice([0, 1])

        return q_values.argmax()

    def set_q_function(self):
        for player_sum in range(22):
            for dealer_showing in range(11):
                for action_idx in range(2):
                    self.Q[player_sum, dealer_showing, action_idx] = self.q_function(
                        (player_sum, dealer_showing), action_idx
                    )
