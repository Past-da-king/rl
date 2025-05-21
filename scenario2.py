import argparse
import sys 
from FourRooms import FourRooms
from Q_agent import QAgent 

def calculate_reward_s2(cell_type: int, new_pos: tuple, packages_remaining: int, is_terminal: bool, old_k: int) -> float:
    

    reward = -1 
    if packages_remaining < old_k:
        reward += 50 

    # Check if the goal (collecting the single package) has been achieved
    if packages_remaining == 0 and is_terminal:
        reward += 100
    return reward

def main():

    parser = argparse.ArgumentParser(description="Run Four-Rooms RL Scenario 2.")
    parser.add_argument(
        '--stochastic',
        action='store_true', # If '--stochastic' is present, args.stochastic will be True mostly for future work
        help="Enable stochastic actions where the agent's intended movement has a 20% chance of random deviation."
    )
    args = parser.parse_args() # Parse command-line arguments

    
    fourRoomsObj = FourRooms('multi', stochastic=args.stochastic)

    num_epochs = 5000 # Total number of training episodes. More epochs allow for more learning. moved this from 2000 to 5000
    max_steps_per_epoch = 1000 # 500 A 1000 Maximum steps an agent can take in one episode.
                              # Prevents infinite loops if agent gets stuck or policy is bad during early training.

    learning_rate = 0.1     # Alpha (α): Controls how quickly the agent learns from new information.
                            # A value of 0.1 means new updates adjust the Q-value by 10%.
    discount_factor = 0.9   # Gamma (γ): Determines the importance of future rewards.
                            # A high value (0.9) means future rewards are heavily considered,
                            # encouraging long-term planning towards the goal.

    epsilon_start = 1.0     # Epsilon (ε) initial value: Agent starts with 100% exploration
                            # to discover all possible states and actions.
    epsilon_end = 0.01      # Epsilon minimum value: Agent maintains a small chance of exploration (1%)
                            # even after extensive training to avoid local optima and adapt to new situations.
    
    epsilon_decay_rate = (epsilon_start - epsilon_end) / num_epochs

    agent = QAgent(num_x=11, num_y=11, max_k=4, num_actions=4,
                   alpha=learning_rate, gamma=discount_factor,
                   epsilon_start=epsilon_start, epsilon_end=epsilon_end,
                   epsilon_decay_rate=epsilon_decay_rate)

 
    print(f"Starting Q-learning training for Scenario 2 (Stochastic: {args.stochastic})...")
    for epoch in range(num_epochs):
        fourRoomsObj.newEpoch()
        current_x, current_y = fourRoomsObj.getPosition()
        current_k = fourRoomsObj.getPackagesRemaining() # This will start at 4 for 'multi'
        current_state = (current_x, current_y, current_k)

        for step in range(max_steps_per_epoch):
            action = agent.choose_action(current_state)
            cellType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(action)

            # IMPORTANT: Call the new reward function for Scenario 2
            reward = calculate_reward_s2(cellType, newPos, packagesRemaining, isTerminal, current_k) # <-- Changed function call

            next_state = (newPos[0], newPos[1], packagesRemaining)
            agent.update_q_table(current_state, action, reward, next_state)

            current_state = next_state
            current_k = packagesRemaining # Make sure current_k is updated for the next iteration's reward calculation

            if isTerminal:
                break
        
        agent.decay_epsilon(epoch)
        if (epoch + 1) % (num_epochs // 10) == 0:
            print(f"Epoch {epoch+1}/{num_epochs} complete. Current epsilon: {agent.epsilon:.4f}")

    print("Training complete for Scenario 2.")


    print("\nDemonstrating learned policy for Scenario 2...")
    
    fourRoomsObj.newEpoch() # Reset environment for a fresh run to visualize the learned path.
    current_x, current_y = fourRoomsObj.getPosition()
    current_k = fourRoomsObj.getPackagesRemaining()
    current_state = (current_x, current_y, current_k)

    print('Agent starts at: {0}'.format(fourRoomsObj.getPosition()))
    for step in range(max_steps_per_epoch):
    
        action = agent.get_greedy_action(current_state)
        
        cellType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(action)

        current_state = (newPos[0], newPos[1], packagesRemaining)
        
        if isTerminal:
            print(f"Agent successfully collected package in {step+1} steps.")
            break # Episode finished, exit policy run.
    

    fourRoomsObj.showPath(-1)
    # fourRoomsObj.showPath(-1, savefig='scenario1_final_path.png')
    

if __name__ == "__main__":
    main()