
import numpy as np
import random
import sys
from FourRooms import FourRooms
from Q_agent import QAgent 

def get_reward_scenario3(current_k_packages, new_k_packages, grid_type_landed_on, env_is_terminal):
    if env_is_terminal and new_k_packages > 0: 
        return -100.0 
    package_collected = grid_type_landed_on > 0
    reward_for_correct_package = 0.0
    if package_collected:
        if current_k_packages == 4 and grid_type_landed_on == 4: reward_for_correct_package = 50.0 
        elif current_k_packages == 3 and grid_type_landed_on == FourRooms.RED: reward_for_correct_package = 50.0 
        elif current_k_packages == 2 and grid_type_landed_on == FourRooms.BLUE: reward_for_correct_package = 50.0
        elif current_k_packages == 1 and grid_type_landed_on == FourRooms.GREEN: reward_for_correct_package = 100.0 
    step_penalty = -1.0
    return step_penalty + reward_for_correct_package

def main():
    stochastic_mode = False
    if '--stochastic' in sys.argv: 
        stochastic_mode = True

    print(f"Running Scenario 3 in {'STOCHASTIC' if stochastic_mode else 'DETERMINISTIC'} mode.")

    fourRoomsObj = FourRooms(scenario='rgb', stochastic=stochastic_mode)
    
    actions_map = {0: FourRooms.UP, 1: FourRooms.DOWN, 2: FourRooms.LEFT, 3: FourRooms.RIGHT}
    num_actions = len(actions_map)
    
    num_epochs = 30000
    max_steps_per_epoch = 300 
    lr = 0.1
    gamma_df = 0.99
    ep_start = 1.0
    ep_end = 0.01 
    epochs_for_decay = num_epochs * 0.30 

    if stochastic_mode:
        print("INFO: Adapting parameters for STOCHASTIC mode.")
        num_epochs = 75000      
        max_steps_per_epoch = 500 
        ep_end = 0.05           
        epochs_for_decay = num_epochs * 0.80 
    
    ep_decay_rate = (ep_start - ep_end) / epochs_for_decay if epochs_for_decay > 0 else (ep_start - ep_end)
    
    agent = QAgent(
        grid_dim_x=13, grid_dim_y=13,
        max_initial_packages=4, 
        num_actions=num_actions,
        learning_rate=lr, 
        discount_factor=gamma_df, 
        epsilon_start=ep_start, 
        epsilon_end=ep_end, 
        epsilon_decay_rate=ep_decay_rate, 
        q_initial_value=0.0
    )

    print(f"Config: Epochs={num_epochs}, MaxSteps/Ep={max_steps_per_epoch}, LR={lr}, Gamma={gamma_df}")
    print(f"Epsilon: Start={ep_start:.2f}, End={ep_end:.2f}, DecayRate={agent.epsilon_decay_rate:.7f} (Decay over ~{epochs_for_decay:.0f} ep)")

    successful_epoch_log_count = 0
    epochs_since_log = 0
    log_interval = num_epochs // 25 if num_epochs >= 25 else 1

    for epoch in range(num_epochs):
        fourRoomsObj.newEpoch()
        current_pos = fourRoomsObj.getPosition()
        current_k = fourRoomsObj.getPackagesRemaining()
        state = (current_pos[0], current_pos[1], current_k)
        steps_this_epoch = 0
        collected_all_this_epoch = False

        for step in range(max_steps_per_epoch):
            steps_this_epoch += 1
            action_idx = agent.choose_action(state)
            env_action = actions_map[action_idx]
            k_before_action = current_k
            grid_type, new_pos, new_k, is_terminal = fourRoomsObj.takeAction(env_action)
            reward = get_reward_scenario3(k_before_action, new_k, grid_type, is_terminal)
            next_state = (new_pos[0], new_pos[1], new_k)
            agent.update_q_table(state, action_idx, reward, next_state, is_terminal)
            state = next_state
            current_k = new_k
            if is_terminal:
                if new_k == 0:
                    successful_epoch_log_count += 1
                    collected_all_this_epoch = True
                break
        
        agent.decay_epsilon(epoch)
        epochs_since_log +=1

        if (epoch + 1) % log_interval == 0 or epoch == num_epochs - 1:
            success_rate = (successful_epoch_log_count / epochs_since_log * 100) if epochs_since_log > 0 else 0.0
            print(f"E {epoch+1}/{num_epochs}. Eps:{agent.epsilon:.3f}. LastSteps:{steps_this_epoch if collected_all_this_epoch else 'DNF'}. RecentSuccess:{success_rate:.1f}%")
            successful_epoch_log_count = 0 
            epochs_since_log = 0

    print("Training complete.")
    
    train_path_filename = f'scenario3_{"stochastic" if stochastic_mode else "deterministic"}_train_final_path.png'
    try:
        fourRoomsObj.showPath(index=-1, savefig=train_path_filename)
        print(f"Last training epoch path saved to {train_path_filename}")
    except Exception as e:
        print(f"Error saving training path: {e}")

    print("\nDemonstrating learned policy (greedy)...")
    fourRoomsObj.newEpoch() 
    agent.epsilon = 0.0 
    current_pos = fourRoomsObj.getPosition()
    current_k = fourRoomsObj.getPackagesRemaining()
    state = (current_pos[0], current_pos[1], current_k)
    print(f'Test Run Start: Pos={current_pos}, Pkgs Left={current_k}')
    test_path_steps = 0
    test_max_steps = max_steps_per_epoch * 2 
    for step in range(test_max_steps):
        test_path_steps += 1
        action_idx = agent.get_greedy_action(state)
        env_action = actions_map[action_idx]
        _, new_pos, new_k, is_terminal = fourRoomsObj.takeAction(env_action)
        state = (new_pos[0], new_pos[1], new_k)
        if is_terminal:
            if new_k == 0: print(f"TEST: Success in {test_path_steps} steps.")
            else: print(f"TEST: Terminated early. Pkgs left: {new_k}, Steps: {test_path_steps}")
            break
    if not fourRoomsObj.isTerminal():
         print(f"TEST: Did not finish in {test_max_steps} steps. Pkgs left: {fourRoomsObj.getPackagesRemaining()}")

    test_path_filename = f'scenario3_{"stochastic" if stochastic_mode else "deterministic"}_test_final_path.png'
    try:
        fourRoomsObj.showPath(index=-1, savefig=test_path_filename)
        print(f"Test epoch path saved to {test_path_filename}")
    except Exception as e:
        print(f"Error saving test path: {e}")

if __name__ == "__main__":
    main()