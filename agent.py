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
        self.E = np.zeros((22, 11, 2))
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
