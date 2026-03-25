#generation of 'expert data'
import gymnasium as gym
import numpy as np
import simulation
import os

class ExpertPolicy:

    def __init__(self):
        self.waiting = False        #currently not waiting for oncoming vehicle cuz its False
        self.steer_steps = 0        #steering steps

    def act(self, state):
        us_left      = state[0]
        us_center    = state[1]
        us_right     = state[2]
        speed        = state[3]
        lateral      = state[4]
        static_det   = state[5]
        static_dist  = state[6]
        oncoming_det = state[7]
        oncoming_dist= state[8]
        oncoming_spd = state[9]
        left_free    = state[10]
        right_free   = state[11]
        traffic      = state[12]
        passed       = state[13]
        stop_sign    = state[14]

        #Rule 1: traffic light and stop sign
        '''traffic = 0.0 → red  
        traffic = 0.5 → yellow  
        traffic = 1.0 → green
        hence if traffic is more than 0.3 need not wait'''
        if traffic < 0.3 or stop_sign > 0.5:
            self.waiting = False
            return 0  #STOP

        #Rule 2: Emergency stop after collision
        if us_center < 0.06:
            return 0

        #Rule 3: After passing obstacle, return to center lane
        if passed > 0.5:
            self.waiting = False
            self.steer_steps = 0
            if lateral < 0.35:      # too far left
                return 4            # steer right back to center
            elif lateral > 0.65:    # too far right
                return 3            # steer left back to center
            return 1                # centered, go forward

        #Rule 4: Obstacle detected ahead
        if static_det > 0.5 and static_dist < 0.65:

            # Oncoming is safe if: not detected, OR detected but far AND slow
            oncoming_safe = (
                oncoming_det < 0.5 or                          # no oncoming
                (oncoming_dist > 0.7) or                       # oncoming far
                (oncoming_dist > 0.5 and oncoming_spd < 0.3)  # oncoming slow and medium
            )

            if oncoming_safe and left_free > 0.5:
                #left lane is clear, move
                self.waiting = False
                self.steer_steps += 1
                return 3  # STEER_LEFT

            elif oncoming_safe and us_left > 0.25:
                #there is space on the left, and no oncoming also.
                self.waiting = False
                self.steer_steps += 1
                return 3  # STEER_LEFT

            elif oncoming_safe and right_free > 0.5:
                #right lane is clear, move
                self.waiting = False
                self.steer_steps += 1
                return 4  # STEER_RIGHT

            elif oncoming_safe and us_right > 0.25:
                #there is space on the right, and no oncoming also.
                self.waiting = False
                self.steer_steps += 1
                return 4  # STEER_RIGHT

            else:
                #Oncoming is fast, wait
                self.waiting = True
                if static_dist < 0.15:
                    return 0  # STOP 
                else:
                    return 2  # FORWARD_SLOW while waiting

        # Rule 5: keep steering left until we clear the obstacle
        if self.steer_steps > 0 and self.steer_steps < 20:
            if us_center > 0.1:   #obstacle not passed 
                self.steer_steps += 1
                return 3          # keep steering left
            else:
                self.steer_steps = 0

        # Rule 6: Default move forward
        self.waiting = False
        return 1  # FORWARD


def generate(num_episodes=500, save_dir="data/expert"):
    os.makedirs(save_dir, exist_ok=True)

    env = gym.make("autonomous-driving-v1", domain_rand=True)

    all_states = []
    all_actions = []
    success_count = 0

    for ep in range(num_episodes):
        expert = ExpertPolicy()   #fresh state each episode
        state, _ = env.reset()
        ep_states = []
        ep_actions = []

        for step in range(800):
            action = expert.act(state)
            ep_states.append(state.copy())
            ep_actions.append(action)

            state, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                if info.get("passed_obstacle", False):
                    all_states.extend(ep_states)
                    all_actions.extend(ep_actions)
                    success_count += 1
                break

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{num_episodes} | "
                  f"Success: {success_count} | "
                  f"Data points: {len(all_states)}")

    env.close()

    states  = np.array(all_states,  dtype=np.float32)
    actions = np.array(all_actions, dtype=np.int64)

    np.save(os.path.join(save_dir, "states.npy"),  states)
    np.save(os.path.join(save_dir, "actions.npy"), actions)

    print(f"\nDone.")
    print(f"Successful episodes : {success_count}/{num_episodes}")
    print(f"Total data points   : {len(states)}")
    print(f"Saved to            : {save_dir}/")


if __name__ == "__main__":
    generate(num_episodes=500)