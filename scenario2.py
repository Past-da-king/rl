import argparse
import sys 
from FourRooms import FourRooms
from q_agent import QAgent # Corrected import from 'Q_agent' to 'q_agent' if your filename is 'q_agent.py'

def calculate_reward_s2(cell_type: int, new_pos: tuple, packages_remaining: int, is_terminal: bool, old_k: int) -> float:
    
    reward = -1 # Cost per step

    # Reward for collecting any package
    if packages_remaining < old_k:
        reward += 50 

    # Additional large reward for collecting ALL packages and episode termination
    if packages_remaining == 0 and is_terminal:
        reward += 100 
        
    return reward

def main():

    parser = argparse.ArgumentParser(description="Run Four-Rooms RL Scenario 2.")
    parser.add_argument(
        '--stochastic',
        action='store_true', # If '--stochastic' is present, args.stochastic will be True
        help="Enable stochastic actions where the agent's intended movement has a 20% chance of random deviation."
    )
    args = parser.parse_args() # Parse command-line arguments

    # Create FourRooms Object for 'multi' scenario
    fourRoomsObj = FourRooms('multi', stochastic=args.stochastic)

    # Hyperparameters
    num_epochs = 5000 
    max_steps_per_epoch = 1000 

    learning_rate = 0.1     
    discount_factor = 0.9   

    epsilon_start = 1.0     
    epsilon_end = 0.01      
    
    epsilon_decay_rate = (epsilon_start - epsilon_end) / num_epochs

    # Initialize QAgent for Scenario 2 (max_k=4 for 4 packages)
    agent = QAgent(num_x=11, num_y=11, max_k=4, num_actions=4,
                   alpha=learning_rate, gamma=discount_factor,
                   epsilon_start=epsilon_start, epsilon_end=epsilon_end,
                   epsilon_decay_rate=epsilon_decay_rate)

    # Training Loop
    print(f"Starting Q-learning training for Scenario 2 (Stochastic: {args.stochastic})...")
    for epoch in range(num_epochs):
        fourRoomsObj.newEpoch()
        current_x, current_y = fourRoomsObj.getPosition()
        current_k = fourRoomsObj.getPackagesRemaining() 
        current_state = (current_x, current_y, current_k)

        for step in range(max_steps_per_epoch):
            action = agent.choose_action(current_state)
            cellType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(action)

            reward = calculate_reward_s2(cellType, newPos, packagesRemaining, isTerminal, current_k)

            next_state = (newPos[0], newPos[1], packagesRemaining)
            agent.update_q_table(current_state, action, reward, next_state)

            current_state = next_state
            current_k = packagesRemaining 

            if isTerminal:
                print(f"Agent successfully collected all packages in {step+1} steps during training epoch {epoch+1}.")
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
            print(f"Agent successfully collected all packages in {step+1} steps.")
            break
    fourRoomsObj.showPath(-1) 
    # fourRoomsObj.showPath(-1, savefig='scenario2_final_path.png') # Saves after window closed

if __name__ == "__main__":
    main()