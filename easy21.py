import random


def sample_card():
    color = random.choice([-1, 1, 1])
    number = random.choice(range(1, 11))
    return color * number


def dealer_step(state: tuple):
    has_terminated, (player_sum, dealer_showing), reward = True, state, -1

    # dealer hits till 17 and sticks after
    while 1 <= dealer_showing < 17:
        dealer_showing += sample_card()
    next_state = player_sum, dealer_showing

    # sum is either negative or more than 17
    # sum not in valid_range -> player wins
    if dealer_showing < 1 or dealer_showing > 21:
        reward = 1

    # check between player sum and dealer sum
    elif player_sum > dealer_showing:
        reward = 1

    # draw when both sums are equal
    elif player_sum == dealer_showing:
        reward = 0

    return has_terminated, next_state, reward


class Easy21:
    @staticmethod
    def get_starting_state():
        return abs(sample_card()), abs(sample_card())

    @staticmethod
    def step(state: tuple, action: str):
        has_terminated = False
        next_state = state
        reward = 0
        # action is hit
        if action == "hit":
            next_state = next_state[0] + sample_card(), next_state[1]

            # player goes bust and we return -1 reward
            if next_state[0] < 1 or next_state[0] > 21:
                has_terminated, reward = True, -1

            # valid sum -> return with zero reward
            return has_terminated, next_state, reward

        # when action is stick, we just play out the dealer
        return dealer_step(state)


def main():
    env = Easy21()
    state = env.get_starting_state()
    print(state)
    has_terminated, next_state, reward = env.step(state, "hit")
    if not has_terminated:
        has_terminated, next_state, reward = env.step(next_state, "hit")
    if not has_terminated:
        has_terminated, next_state, reward = env.step(next_state, "stick")
    print(has_terminated, next_state, reward)


if __name__ == "__main__":
    main()
