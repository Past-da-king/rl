import numpy as np
import random
import sys
from FourRooms import FourRooms
from Q_agent import QAgent 

def get_reward_scenario3(current_k_packages, new_k_packages, grid_type_landed_on, env_is_terminal):
    """
    Calculates reward for Scenario 3.
    The sequence enforced by FourRooms.py for 'rgb' scenario:
    1. Collect P4 (Value 4) when current_k_packages=4 -> results in new_k_packages=3
    2. Collect RED (Value FourRooms.RED=1) when current_k_packages=3 -> results in new_k_packages=2
    3. Collect BLUE (Value FourRooms.BLUE=2) when current_k_packages=2 -> results in new_k_packages=1
    4. Collect GREEN (Value FourRooms.GREEN=3) when current_k_packages=1 -> results in new_k_packages=0
    """
    
    if env_is_terminal and new_k_packages > 0: 
        return -100.0 

    package_collected = grid_type_landed_on > 0
    
    if package_collected:
        if current_k_packages == 4 and grid_type_landed_on == 4: 
            return 50.0 
        elif current_k_packages == 3 and grid_type_landed_on == FourRooms.RED: 
            return 50.0 
        elif current_k_packages == 2 and grid_type_landed_on == FourRooms.BLUE:
            return 50.0
        elif current_k_packages == 1 and grid_type_landed_on == FourRooms.GREEN: 
            return 100.0 
        # else: # Collecting a package out of the P4->R->B->G sequence
            # No specific penalty here if FourRooms.py didn't terminate, 
            # but no positive reward either. The step penalty will apply.
            # The env_is_terminal check above handles the main penalty.
            pass

    step_penalty = -1.0
    return step_penalty

def main():
    stochastic_mode = False
    nightmare_mode = False 

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.lower() == 'stochastic':
                stochastic_mode = True
            elif arg.lower() == 'nightmare': 
                nightmare_mode = True
    
    if stochastic_mode:
        print("Running Scenario 3 in STOCHASTIC mode.")
    else:
        print("Running Scenario 3 in DETERMINISTIC mode.")
    if nightmare_mode:
        print("Nightmare mode: Paths will be saved to files (e.g., scenario3_final_path.png).")

    fourRoomsObj = FourRooms(scenario='rgb', stochastic=stochastic_mode)
    
    actions_map = {
        0: FourRooms.UP, 
        1: FourRooms.DOWN, 
        2: FourRooms.LEFT, 
        3: FourRooms.RIGHT
    }
    num_actions = len(actions_map)
    
    # Initialize Q-learning agent using the new class structure
    agent = QAgent(
        grid_dim_x=13,  # Q-table x-dimension (0-12 for FourRooms)
        grid_dim_y=13,  # Q-table y-dimension (0-12 for FourRooms)
        max_initial_packages=4, # 'rgb' scenario starts with 4 packages
        num_actions=num_actions,
        learning_rate=0.1, 
        discount_factor=0.99, 
        epsilon_start=1.0, 
        epsilon_end=0.01, # Changed from 0.05 to match previous S3 output
        epsilon_decay_rate=0.00015, # Adjusted for ~30k epochs to reach 0.01
                                   # (1.0 - 0.01) / 0.00015 approx 6600 epochs to reach min
                                   # Decay from 1.0 to 0.01: (1.0-0.01)/N = rate -> rate = 0.99/N
                                   # For N=10000, rate = 0.000099. For N=7000, rate=0.00014
                                   # Let's use epsilon_decay_rate = (1.0 - 0.01) / 9000  (to reach min by ~epoch 9000)
                                   # epsilon_decay_rate = 0.99 / 9000 = 0.00011
        q_initial_value=0.0 # Standard Q-learning (not optimistic)
    )
    # Recalculating epsilon_decay_rate to match the S3 output where epsilon reached 0.01 around epoch 9000-10000
    # If epsilon_start=1.0, epsilon_end=0.01, epochs_to_min_epsilon = 9000
    # epsilon_decay_rate = (epsilon_start - epsilon_end) / epochs_to_min_epsilon
    agent.epsilon_decay_rate = (agent.epsilon_start - agent.epsilon_end) / 9000.0


    num_epochs = 30000
    max_steps_per_epoch = 250

    print(f"Starting Q-learning training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        fourRoomsObj.newEpoch()
        
        current_pos = fourRoomsObj.getPosition() # (x,y) tuple, values 1-11 for traversable
        current_k = fourRoomsObj.getPackagesRemaining()
        
        # State for Q-table: (x, y, k) using direct coordinates
        state = (current_pos[0], current_pos[1], current_k)
        
        for step in range(max_steps_per_epoch):
            action_idx = agent.choose_action(state)
            env_action = actions_map[action_idx]
            
            grid_type, new_pos, new_k, is_terminal = fourRoomsObj.takeAction(env_action)
            
            reward = get_reward_scenario3(current_k, new_k, grid_type, is_terminal)
            
            next_state = (new_pos[0], new_pos[1], new_k)
            
            agent.update_q_table(state, action_idx, reward, next_state, is_terminal)
            
            state = next_state
            current_k = new_k
            
            if is_terminal:
                break
        
        agent.decay_epsilon(epoch) # Pass current epoch number for linear decay

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}. Epsilon: {agent.epsilon:.4f}.")

    print("Training complete.")
    
    print("Showing path for the last training epoch...")
    try:
        if nightmare_mode:
            fourRoomsObj.showPath(index=-1, savefig='scenario3_train_final_path.png')
            print("Path from last training epoch saved to scenario3_train_final_path.png")
        else:
            fourRoomsObj.showPath(index=-1)
    except Exception as e:
        print(f"Error displaying/saving training path: {e}")

    print("\nRunning a test epoch with the learned policy (epsilon=0)...")
    fourRoomsObj.newEpoch()
    agent.epsilon = 0.0 # Set to greedy for testing
    
    current_pos = fourRoomsObj.getPosition()
    current_k = fourRoomsObj.getPackagesRemaining()
    state = (current_pos[0], current_pos[1], current_k)
    
    print(f"Test Run Start State: Pos={current_pos}, Packages Left={current_k}")
    
    test_path_len = 0
    for step in range(max_steps_per_epoch):
        action_idx = agent.get_greedy_action(state) # Use get_greedy_action for pure exploitation
        env_action = actions_map[action_idx]
        
        grid_type, new_pos, new_k, is_terminal = fourRoomsObj.takeAction(env_action)
        
        state = (new_pos[0], new_pos[1], new_k)
        test_path_len +=1
        
        if is_terminal:
            print(f"Test epoch finished in {test_path_len} steps. Final packages left: {new_k}")
            break
            
    if not fourRoomsObj.isTerminal():
         print(f"Test epoch did not finish collecting all packages within {max_steps_per_epoch} steps. Packages left: {fourRoomsObj.getPackagesRemaining()}")

    print("Showing path for the test epoch...")
    try:

        fourRoomsObj.showPath(index=-1, savefig='3_test_final_path.png')
        print("Path from test epoch saved to scenario3_test_final_path.png")

    except Exception as e:
        print(f"Error displaying/saving test path: {e}")

if __name__ == "__main__":
    main()