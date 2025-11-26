import matplotlib.pyplot as plt

def plot_rewards(rewards):
    plt.figure(figsize=(8,4))
    plt.plot(rewards)
    plt.title("Q-Learning Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.show()
