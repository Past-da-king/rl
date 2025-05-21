import argparse
import sys
from FourRooms import FourRooms
from Q_agent import QAgent
def calculate_reward_s3(cell_type: int, new_pos: tuple, packages_remaining: int, is_terminal: bool, old_k: int) -> float:
    reward = -1

    expected_package_type_for_order = None
    if old_k == 4: # When 4 packages remain, expect RED (first ordered)
        expected_package_type_for_order = FourRooms.RED
    elif old_k == 3: # When 3 packages remain, expect GREEN (second ordered)
        expected_package_type_for_order = FourRooms.GREEN
    elif old_k == 2: # When 2 packages remain, expect BLUE (third ordered)
        expected_package_type_for_order = FourRooms.BLUE

    if packages_remaining < old_k: # A package was just collected
        if cell_type == expected_package_type_for_order:
            reward += 200 # Correct ordered package picked
        elif cell_type in [FourRooms.RED, FourRooms.GREEN, FourRooms.BLUE]:
            # Wrong ordered package picked (triggers early termination by environment)
            reward -= 1000
        else: # This is the 4th, unordered package
            reward += 0 # Neutral reward for picking the extra package

    if packages_remaining == 0 and is_terminal: # All packages collected (and ordered correctly)
        reward += 1000
    elif is_terminal and packages_remaining > 0: # Terminated early due to wrong ordered package
        pass 

    return reward

def main():
    parser = argparse.ArgumentParser(description="Run Four-Rooms RL Scenario 3.")
    parser.add_argument(
        '--stochastic',
        action='store_true',
        help="Enable stochastic actions where the agent's intended movement has a 20% chance of random deviation."
    )
    args = parser.parse_args()

    fourRoomsObj = FourRooms('rgb', stochastic=args.stochastic)

    num_epochs = 20000
    max_steps_per_epoch = 1500

    learning_rate = 0.1
    discount_factor = 0.9

    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay_rate = (epsilon_start - epsilon_end) / num_epochs

    # max_k changed to 4 because FourRooms.py creates 4 packages for 'rgb' scenario
    agent = QAgent(num_x=11, num_y=11, max_k=4, num_actions=4,
                   alpha=learning_rate, gamma=discount_factor,
                   epsilon_start=epsilon_start, epsilon_end=epsilon_end,
                   epsilon_decay_rate=epsilon_decay_rate)

    print(f"Starting Q-learning training for Scenario 3 (Stochastic: {args.stochastic})...")
    for epoch in range(num_epochs):
        fourRoomsObj.newEpoch()
        current_x, current_y = fourRoomsObj.getPosition()
        current_k = fourRoomsObj.getPackagesRemaining()
        current_state = (current_x, current_y, current_k)

        for step in range(max_steps_per_epoch):
            action = agent.choose_action(current_state)
            cellType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(action)

            reward = calculate_reward_s3(cellType, newPos, packagesRemaining, isTerminal, current_k)

            next_state = (newPos[0], newPos[1], packagesRemaining)
            agent.update_q_table(current_state, action, reward, next_state)

            current_state = next_state
            current_k = packagesRemaining 

            if isTerminal:
                if packagesRemaining == 0:
                    print(f"Agent collected all packages in order in {step+1} steps during training epoch {epoch+1}.")
                else:
                    print(f"Agent terminated early (wrong package) in {step+1} steps during training epoch {epoch+1}.")
                break
        
        agent.decay_epsilon(epoch)
        if (epoch + 1) % (num_epochs // 10) == 0:
            print(f"Epoch {epoch+1}/{num_epochs} complete. Current epsilon: {agent.epsilon:.4f}")

    print("Training complete for Scenario 3.")

    print("\nDemonstrating learned policy for Scenario 3...")
    
    fourRoomsObj.newEpoch()
    current_x, current_y = fourRoomsObj.getPosition()
    current_k = fourRoomsObj.getPackagesRemaining()
    current_state = (current_x, current_y, current_k)

    print('Agent starts at: {0}'.format(fourRoomsObj.getPosition()))
    for step in range(max_steps_per_epoch):
        action = agent.get_greedy_action(current_state)
        cellType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(action)
        current_state = (newPos[0], newPos[1], packagesRemaining)
        if isTerminal:
            if packagesRemaining == 0:
                print(f"Agent successfully collected all packages in order in {step+1} steps.")
            else:
                print(f"Agent failed to collect all packages in order (terminated early) in {step+1} steps.")
            break
    
    fourRoomsObj.showPath(-1)
    # fourRoomsObj.showPath(-1, savefig='scenario3_final_path.png')

if __name__ == "__main__":
    main()