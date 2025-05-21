
import argparse
import sys
import numpy as np # For potential averaging if running multiple trials
import matplotlib.pyplot as plt # For plotting
from FourRooms import FourRooms
from Q_agent import QAgent 
def calculate_reward_s1(cell_type: int, packages_remaining: int, is_terminal: bool) -> float:
    reward = -1  # Cost per step
    if packages_remaining == 0 and is_terminal:
        reward += 100 # Reward for collecting the package
    return reward

def run_training_session(fourRoomsObj, agent_params, num_epochs, max_steps_per_epoch, session_label=""):
    """
    Runs a full training session for a given agent configuration and logs metrics.
    agent_params should be a dict: {'epsilon_start', 'epsilon_end', 'epsilon_decay_rate_divisor'}
    epsilon_decay_rate_divisor: e.g., 1 for decay over all epochs, 0.5 for half, etc.
    """
    agent = QAgent(grid_dim_x=13, # Q-table uses 0-12
                   grid_dim_y=13,
                   max_initial_packages=1, 
                   num_actions=4,
                   learning_rate=0.1, # Keep these constant for fair comparison of exploration
                   discount_factor=0.9,
                   epsilon_start=agent_params['epsilon_start'],
                   epsilon_end=agent_params['epsilon_end'],
                   # Epsilon decay rate calculated based on num_epochs and divisor
                   epsilon_decay_rate = (agent_params['epsilon_start'] - agent_params['epsilon_end']) / (num_epochs * agent_params['epsilon_decay_rate_divisor'])
                   )
    
    session_epoch_steps = []
    # session_epoch_rewards = [] # Optional: if you track cumulative reward

    print(f"\n--- Starting Training: {session_label} ---")
    print(f"Epsilon params: start={agent.epsilon_start:.2f}, end={agent.epsilon_end:.2f}, decay_rate_calc_divisor={agent_params['epsilon_decay_rate_divisor']:.2f} (actual rate: {agent.epsilon_decay_rate:.6f})")

    for epoch in range(num_epochs):
        fourRoomsObj.newEpoch() # Reset environment
        current_pos = fourRoomsObj.getPosition()
        current_k = fourRoomsObj.getPackagesRemaining()
        current_state = (current_pos[0], current_pos[1], current_k)
        
        steps_this_epoch = 0
        # reward_this_epoch = 0

        for step in range(max_steps_per_epoch):
            steps_this_epoch += 1
            action = agent.choose_action(current_state)
            cellType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(action)
            reward = calculate_reward_s1(cellType, packagesRemaining, isTerminal)
            # reward_this_epoch += reward
            next_state = (newPos[0], newPos[1], packagesRemaining)
            agent.update_q_table(current_state, action, reward, next_state, isTerminal)
            current_state = next_state
            if isTerminal:
                break
        
        session_epoch_steps.append(steps_this_epoch)
        # session_epoch_rewards.append(reward_this_epoch)
        agent.decay_epsilon(epoch)

        if (epoch + 1) % (num_epochs // 10) == 0:
            print(f"  {session_label} - Epoch {epoch+1}/{num_epochs}. Steps: {steps_this_epoch}. Epsilon: {agent.epsilon:.4f}")
    
    print(f"--- Training Complete: {session_label} ---")
    
    # Run a final test path for this agent
    print(f"Demonstrating learned policy for {session_label}...")
    fourRoomsObj.newEpoch()
    current_pos = fourRoomsObj.getPosition()
    current_k = fourRoomsObj.getPackagesRemaining()
    current_state = (current_pos[0], current_pos[1], current_k)
    test_steps = 0
    for step in range(max_steps_per_epoch):
        test_steps +=1
        action = agent.get_greedy_action(current_state)
        _, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(action)
        current_state = (newPos[0], newPos[1], packagesRemaining)
        if isTerminal:
            break
    print(f"{session_label} - Test path collected package in {test_steps} steps.")
    fourRoomsObj.showPath(-1, savefig=f'scenario1_path_{session_label.replace(" ", "_")}.png')
    
    return session_epoch_steps #, session_epoch_rewards

def main():
    parser = argparse.ArgumentParser(description="Run Four-Rooms RL Scenario 1 with exploration strategy comparison.")
    parser.add_argument('--stochastic', action='store_true', help="Enable stochastic actions.")
    parser.add_argument('--nightmare', action='store_true', help="Save plots instead of showing them.")
    args = parser.parse_args()

    fourRoomsObj = FourRooms('simple', stochastic=args.stochastic)
    
    num_epochs = 2000 # Number of epochs for each strategy run
    max_steps_per_epoch = 100 # Max steps for S1, can be lower

    # Define Exploration Strategies
    # Strategy 1: Standard decay to a low epsilon_end over all epochs
    params_strategy_A = {
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay_rate_divisor': 1.0 # Decays over num_epochs
    }
    # Strategy 2: Slower decay OR higher epsilon_end OR decay over fewer epochs then constant
    params_strategy_B = {
        'epsilon_start': 1.0,
        'epsilon_end': 0.1, # Higher minimum exploration
        'epsilon_decay_rate_divisor': 0.75 # Decays to min over 75% of epochs, then constant at min
    }


    
    print(f"Running Scenario 1 (Stochastic: {args.stochastic})")

    steps_A = run_training_session(fourRoomsObj, params_strategy_A, num_epochs, max_steps_per_epoch, "Strategy_A_Fast_Decay_Low_End")

    
    steps_B = run_training_session(fourRoomsObj, params_strategy_B, num_epochs, max_steps_per_epoch, "Strategy_B_Slower_Decay_High_End")

    # --- Plotting the comparison ---
    epochs_axis = range(num_epochs)
    plt.figure(figsize=(12, 7))
    
    # Apply a simple moving average for smoother plots if desired
    window_size = 50 # e.g., average over 50 epochs
    if len(steps_A) >= window_size:
        steps_A_smooth = np.convolve(steps_A, np.ones(window_size)/window_size, mode='valid')
        epochs_A_smooth = epochs_axis[window_size-1:]
    else:
        steps_A_smooth = steps_A
        epochs_A_smooth = epochs_axis

    if len(steps_B) >= window_size:
        steps_B_smooth = np.convolve(steps_B, np.ones(window_size)/window_size, mode='valid')
        epochs_B_smooth = epochs_axis[window_size-1:]
    else:
        steps_B_smooth = steps_B
        epochs_B_smooth = epochs_axis

    plt.plot(epochs_A_smooth, steps_A_smooth, label='Strategy A (Fast Decay, ε_end=0.01)')
    plt.plot(epochs_B_smooth, steps_B_smooth, label='Strategy B (Slower Decay, ε_end=0.1)')
    
    plt.xlabel('Epoch')
    plt.ylabel(f'Steps to Collect Package (Smoothed, Window={window_size})')
    plt.title(f'Scenario 1: Exploration Strategy Comparison (Stochastic: {args.stochastic})')
    plt.legend()
    plt.grid(True)
    
    plot_filename = 'scenario1_exploration_comparison.png'
    plt.savefig(plot_filename)
    print(f"\nComparison plot saved to {plot_filename}")
    if not args.nightmare:
        plt.show()

    print("\nScenario 1 with exploration comparison complete.")

if __name__ == "__main__":
    main()