
import argparse
import sys 
from FourRooms import FourRooms
from Q_agent import QAgent

def calculate_reward_s2(cell_type: int, new_pos: tuple, packages_remaining: int, is_terminal: bool, old_k: int) -> float:
    reward = -1 
    if packages_remaining < old_k: 
        reward += 50 
    if packages_remaining == 0 and is_terminal: 
        reward += 100 
    return reward

def main():
    parser = argparse.ArgumentParser(description="Run Four-Rooms RL Scenario 2.")
    parser.add_argument('--stochastic', action='store_true', help="Enable stochastic actions.")
    parser.add_argument('--nightmare', action='store_true', help="Save plots to files instead of displaying them.")
    args = parser.parse_args()

    fourRoomsObj = FourRooms('multi', stochastic=args.stochastic)

    num_epochs = 10000 
    if args.stochastic:
        num_epochs = 25000 

    max_steps_per_epoch = 1000 
    if args.stochastic:
        max_steps_per_epoch = 1500 

    # Define variables for agent parameters
    lr = 0.1     
    gamma_df = 0.99   
    ep_start = 1.0     
    ep_end = 0.01      
    if args.stochastic:
        ep_end = 0.05 

    epochs_for_decay = num_epochs * 0.9 
    ep_decay_rate = (ep_start - ep_end) / epochs_for_decay if epochs_for_decay > 0 else (ep_start - ep_end)

    # Corrected QAgent initialization using the parameter names from Q_agent.py
    agent = QAgent(grid_dim_x=13, grid_dim_y=13, 
                   max_initial_packages=4, 
                   num_actions=4,
                   learning_rate=lr,           
                   discount_factor=gamma_df,  
                   epsilon_start=ep_start, 
                   epsilon_end=ep_end,
                   epsilon_decay_rate=ep_decay_rate,
                   q_initial_value=0.0 
                   )

    print(f"Starting Q-learning for Scenario 2 (Stochastic: {args.stochastic})...")
    print(f"Epochs: {num_epochs}, Max Steps/Ep: {max_steps_per_epoch}, LR: {lr}, Gamma: {gamma_df}")
    print(f"Epsilon: start={ep_start:.2f}, end={ep_end:.2f}, decay_rate={agent.epsilon_decay_rate:.6f}")

    successful_epoch_log_count = 0
    epochs_since_log = 0

    for epoch in range(num_epochs):
        fourRoomsObj.newEpoch()
        current_pos = fourRoomsObj.getPosition()
        current_k = fourRoomsObj.getPackagesRemaining() 
        current_state = (current_pos[0], current_pos[1], current_k)
        
        steps_this_epoch = 0
        collected_all_this_epoch = False

        for step in range(max_steps_per_epoch):
            steps_this_epoch += 1
            action = agent.choose_action(current_state)
            k_before_action = current_k 
            cellType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(action)
            reward = calculate_reward_s2(cellType, newPos, packagesRemaining, isTerminal, k_before_action)
            next_state = (newPos[0], newPos[1], packagesRemaining)
            agent.update_q_table(current_state, action, reward, next_state, isTerminal) 
            current_state = next_state
            current_k = packagesRemaining 

            if isTerminal:
                if packagesRemaining == 0:
                    successful_epoch_log_count += 1
                    collected_all_this_epoch = True
                break
        
        agent.decay_epsilon(epoch)
        epochs_since_log +=1

        log_interval = num_epochs // 20 if num_epochs >= 20 else 1
        if (epoch + 1) % log_interval == 0 or epoch == num_epochs - 1:
            success_rate = (successful_epoch_log_count / epochs_since_log * 100) if epochs_since_log > 0 else 0.0
            print(f"E {epoch+1}/{num_epochs}. Eps: {agent.epsilon:.3f}. Steps: {steps_this_epoch if collected_all_this_epoch else 'DNF'}. Success Log Batch: {success_rate:.1f}%")
            successful_epoch_log_count = 0 
            epochs_since_log = 0
            
    print("Training complete.")

    print("\nDemonstrating learned policy...")
    fourRoomsObj.newEpoch()
    current_pos = fourRoomsObj.getPosition()
    current_k = fourRoomsObj.getPackagesRemaining()
    current_state = (current_pos[0], current_pos[1], current_k)
    
    test_path_steps = 0
    print(f'Agent starts: {fourRoomsObj.getPosition()}, Pkgs: {current_k}')
    test_max_steps = max_steps_per_epoch * 2 if args.stochastic else max_steps_per_epoch
    for step in range(test_max_steps): 
        test_path_steps += 1
        action = agent.get_greedy_action(current_state)
        _, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(action)
        current_state = (newPos[0], newPos[1], packagesRemaining)
        if isTerminal:
            if packagesRemaining == 0:
                print(f"TEST: All packages collected in {test_path_steps} steps.")
            else:
                print(f"TEST: Terminated early. Pkgs left: {packagesRemaining}, Steps: {test_path_steps}")
            break
    if not fourRoomsObj.isTerminal():
         print(f"TEST: Did not finish in {test_max_steps} steps. Pkgs left: {fourRoomsObj.getPackagesRemaining()}")

    path_filename = 'scenario2_stochastic_final_path.png' if args.stochastic else 'scenario2_final_path.png'
    
    try:
        fourRoomsObj.showPath(-1, savefig=path_filename) 
        print(f"Final path saved to {path_filename}")
        if not args.nightmare and not args.stochastic:
             fourRoomsObj.showPath(-1) 
    except Exception as e:
        print(f"Error during showPath: {e}")

if __name__ == "__main__":
    main()