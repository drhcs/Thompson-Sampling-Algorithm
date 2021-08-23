from scipy.stats import beta
import numpy as np

X_SCALE = np.linspace(0, 1, 100)
TURN_CHOICES = 3
ROCK = 0
PAPER = 1
SCISSORS = 2

alphas = [1, 1, 1]
betas = [1, 1, 1]


def get_confidence(turn_choice):

    # Point percent functions
    choice_ppf = None
    ppfs = []

    for choice in range(TURN_CHOICES):
        if choice == turn_choice:
            choice_ppf = beta.ppf(X_SCALE, alphas[choice], betas[choice])
        else:
            ppfs.append(beta.ppf(X_SCALE, alphas[choice], betas[choice]))

    percentiles_with_higher_prob = 0
    for percentile in range(0, 100):
        greater_than_other_percentiles = 0
        for other_ppf in ppfs:
            if choice_ppf[percentile] > other_ppf[percentile]:
                greater_than_other_percentiles += 1
        if greater_than_other_percentiles == len(ppfs):
            percentiles_with_higher_prob += 1

    return percentiles_with_higher_prob


def get_all_confidences():
    confidences = []
    for choice in range(TURN_CHOICES):
        confidences.append(get_confidence(choice))
    return confidences


def get_optimal_choice():
    return np.argmax(get_all_confidences())


def convert_player_input(input):

    input = input.lower()

    if input == "rock":
        return ROCK
    elif input == "paper":
        return PAPER
    elif input == "scissors":
        return SCISSORS

    return -1


def choice_to_string(choice):

    if choice == ROCK:
        return "Rock"
    elif choice == PAPER:
        return "Paper"
    elif choice == SCISSORS:
        return "Scissors"

    return ""


def print_statistics(player_choice, action, regret):
    print("Player Chooses: {}".format(choice_to_string(player_choice)))
    print("AI Chooses: {}".format(choice_to_string(action)))
    confidences = get_all_confidences()
    print("AI Confidence: ROCK - {}%. PAPER - {}%. SCISSORS - {}%.".format(confidences[ROCK], confidences[PAPER], confidences[SCISSORS]))
    print("AI Regret This Turn: {}".format(regret))
    print("----------")


def prompt_player(action, regret):
    player_input = input("Rock! Paper! Scissors! Go! ")
    player_choice = convert_player_input(player_input)

    reward = 0
    if action == ROCK and player_choice == SCISSORS:
        reward = 1
    elif action == PAPER and player_choice == ROCK:
        reward = 1
    elif action == SCISSORS and player_choice == PAPER:
        reward = 1

    print_statistics(player_choice, action, regret)

    return reward


def take_turn():

    thetas = []

    # Sample models
    for choice in range(TURN_CHOICES):
        thetas.append(np.random.beta(alphas[choice], betas[choice]))

    # Select action
    action = np.argmax(thetas)

    # Calculate per-period regret (mean reward of the optimal action - mean reward of selected action)
    optimal = get_optimal_choice()
    regret = beta.mean(alphas[optimal], betas[optimal]) - beta.mean(alphas[action], betas[action])

    # Apply action and observe reward
    reward = prompt_player(action, regret)

    # Update distribution (Reinforcement learning)
    alphas[action] += reward
    betas[action] += 1 - reward


while True:
    take_turn()
