import random
from dataclasses import dataclass, astuple


def sample_card():
    color = random.choice([-1, 1, 1])
    number = random.choice(range(1, 11))
    return color * number


@dataclass
class State:
    player_sum: int = abs(sample_card())
    dealer_showing: int = abs(sample_card())


@dataclass
class StepReturn:
    has_terminated: bool = False
    next_state: State = State()
    reward: int = 0


def dealer_step(state: State):
    dealer_sum = state.dealer_showing
    while 1 <= dealer_sum < 17:
        dealer_sum += sample_card()

    # sum is either negative or more than 17
    next_state = State(state.player_sum, dealer_sum)

    # sum not in valid_range -> player wins
    if next_state.dealer_showing < 1 or next_state.dealer_showing > 21:
        return StepReturn(has_terminated=True, next_state=next_state, reward=1)

    # check between player sum and dealer sum
    reward = -1
    if next_state.player_sum > next_state.dealer_showing:
        reward = 1
    elif next_state.player_sum == next_state.dealer_showing:
        reward = 0

    return StepReturn(True, next_state, reward)


def step(state: State, action: str):
    result = StepReturn()
    if action == "hit":
        card = sample_card()
        result.next_state = State(state.player_sum + card, state.dealer_showing)

        # valid sum, then return with zero reward
        if 0 < result.next_state.player_sum < 22:
            return result

        # player goes bust and we return -1 reward
        result.has_terminated = True
        result.reward = -1
        return result

    # when action is stick, we just play out the dealer
    return dealer_step(state)


if __name__ == "__main__":
    state = State()
    print(state)
    result = step(state, "hit")
    if not result.has_terminated:
        result = step(result.next_state, "hit")
    if not result.has_terminated:
        result = step(result.next_state, "stick")
    print(result)
