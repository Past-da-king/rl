import argparse
import sys 
from FourRooms import FourRooms
from q_agent import QAgent 

def calculate_reward_s1(cell_type: int, new_pos: tuple, packages_remaining: int, is_terminal: bool, old_k: int) -> float:

    reward = -1 

    # Check if the goal (collecting the single package) has been achieved
    if packages_remaining == 0 and is_terminal:
        reward += 100
    return reward

def main():

    parser = argparse.ArgumentParser(description="Run Four-Rooms RL Scenario 1.")
    parser.add_argument(
        '--stochastic',
        action='store_true', # If '--stochastic' is present, args.stochastic will be True mostly for future work
        help="Enable stochastic actions where the agent's intended movement has a 20% chance of random deviation."
    )
    args = parser.parse_args() # Parse command-line arguments

    
    fourRoomsObj = FourRooms('simple', stochastic=args.stochastic)

    num_epochs = 2000 # Total number of training episodes. More epochs allow for more learning.
    max_steps_per_epoch = 500 # Maximum steps an agent can take in one episode.
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

    agent = QAgent(num_x=11, num_y=11, max_k=1, num_actions=4,
                   alpha=learning_rate, gamma=discount_factor,
                   epsilon_start=epsilon_start, epsilon_end=epsilon_end,
                   epsilon_decay_rate=epsilon_decay_rate)

 
    print(f"Starting Q-learning training for Scenario 1 (Stochastic: {args.stochastic})...")
    for epoch in range(num_epochs):
        fourRoomsObj.newEpoch() # !!!Resets the environment for each new training episode.
                                
        current_x, current_y = fourRoomsObj.getPosition()
        current_k = fourRoomsObj.getPackagesRemaining()
        current_state = (current_x, current_y, current_k) # Form the agent's state tuple

   
        for step in range(max_steps_per_epoch):
            # Agent chooses an action based on its current state using the epsilon-greedy policy.
            action = agent.choose_action(current_state)

            # The environment executes the chosen action and provides feedback.
            # Returns: (cellType, newPos, packagesRemaining, isTerminal)
            cellType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(action)

            # Calculate the immediate reward for this specific state-action transition.
            # We pass current_k (old_k) for consistency with other scenarios, though not strictly used in S1.
            reward = calculate_reward_s1(cellType, newPos, packagesRemaining, isTerminal, current_k)

            # Define the next state based on the environment's response.
            next_state = (newPos[0], newPos[1], packagesRemaining)

            # **Q-Learning Update Step:** The core learning mechanism.
            # Agent updates its Q-value for the (current_state, action) pair.
            agent.update_q_table(current_state, action, reward, next_state)

            # Update the agent's current state for the next iteration of the inner loop.
            current_state = next_state
            current_k = packagesRemaining # Keep old_k updated for next step's reward calculation.

            # Check if the episode terminated (e.g., package collected).
            if isTerminal:
                # print(f"Epoch {epoch+1}/{num_epochs}: Terminated in {step+1} steps. Packages left: {packagesRemaining}")
                break # End this epoch, start a new one if num_epochs not reached.

        # Decay epsilon at the end of each epoch.
        # This gradually reduces exploration as training progresses.
        agent.decay_epsilon(epoch)

        # Optional: Print progress periodically for long training runs
        if (epoch + 1) % (num_epochs // 10) == 0: # Prints every 10% of total epochs
            print(f"Epoch {epoch+1}/{num_epochs} complete. Current epsilon: {agent.epsilon:.4f}")

    print("Training complete for Scenario 1.")


    print("\nDemonstrating learned policy for Scenario 1...")
    
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
    

if __name__ == "__main__":
    main()