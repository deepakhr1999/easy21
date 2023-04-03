import random


def sample_card():
    color = random.choice([-1, 1, 1])
    number = random.choice(range(1, 11))
    return color * number


def dealer_step(state: tuple):
    done, (player_sum, dealer_showing), reward = True, state, -1

    # dealer hits till 17 and sticks after
    while 1 <= dealer_showing < 17:
        dealer_showing += sample_card()

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

    # but while returning, return dealer showing of the start state
    return state, reward, done


class Easy21:
    @staticmethod
    def reset():
        return abs(sample_card()), abs(sample_card())

    @staticmethod
    def step(state: tuple, action: str):
        done = False
        next_state = state
        reward = 0
        # action is hit
        if action == "hit":
            next_state = next_state[0] + sample_card(), next_state[1]

            # player goes bust and we return -1 reward
            if next_state[0] < 1 or next_state[0] > 21:
                done, reward = True, -1

            # valid sum -> return with zero reward
            return next_state, reward, done

        # when action is stick, we just play out the dealer
        return dealer_step(state)


def main():
    env = Easy21()
    state = env.reset()
    print(state)
    next_state, reward, done = env.step(state, "hit")
    if not done:
        next_state, reward, done = env.step(next_state, "hit")
    if not done:
        next_state, reward, done = env.step(next_state, "stick")
    print(next_state, reward, done)


if __name__ == "__main__":
    main()
