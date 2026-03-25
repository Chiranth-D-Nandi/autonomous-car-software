import os
import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    EvalCallback, 
    CheckpointCallback,
    BaseCallback,
)
from torch.utils.tensorboard import SummaryWriter
import simulation
from transformer import TransformerFeatures

class Loggingclbk(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        self.episode_rewards = [] #list to track rewards, lengths and successes
        self.episode_lengths = []
        self.successes = []
    
    def _on_step(self): #override basecallback function
        for info in self.locals.get("infos", []): #using get instead of self.locals["infos"] to prevent crashes in case of absense
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)

                success = info.get("passed_obstacle", False) #get passed_obstacle, if not there, default is False.
                self.successes.append(float(success))

                self.writer.add_scalar("episode/reward", ep_reward, self.num_timesteps)
                self.writer.add_scalar("episode/length", ep_length, self.num_timesteps)

                if len(self.successes) >= 100:
                    success_rate = np.mean(self.successes[-100:])
                    self.writer.add_scalar("episode/success_rate", success_rate, self.num_timesteps)

        return True
    
    def _on_training_end(self):
        self.writer.close()
        print(f"\nTraining complete. TensorBoard logs at: {self.log_dir}")
        print(f"Run: tensorboard --logdir {self.log_dir}")
        print(f"Total episodes: {len(self.episode_rewards)}")
        if self.episode_rewards:
            print(f"Final avg reward (last 100): "
                  f"{np.mean(self.episode_rewards[-100:]):.1f}")
        if self.successes:
            print(f"Final success rate (last 100): "
                  f"{np.mean(self.successes[-100:])*100:.1f}%")

def make_env(rank, seed=0, domain_rand=True):
    def _init():
        env = gym.make("autonomous-driving-v1", domain_rand=domain_rand)
        env.reset(seed=seed + rank) 
        return env
    return _init

def train():
    NUM_ENVS = 8
    TOTAL_TIMESTEPS = 2_000_000
    LOG_DIR = "runs/ppo"
    MODEL_DIR = "models/rl_ppo"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(f"{MODEL_DIR}/checkpoints", exist_ok=True)
    print(f"creating {NUM_ENVS} number of parallel game environments")
    env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    env = VecMonitor(env, LOG_DIR) #random training envs, but evaluation remains same, to obtain different scores
    eval_env = SubprocVecEnv([
        make_env(i + 100, domain_rand=False)  #for random seed and unidentical environments
        for i in range(4)
    ])
    policy_kwargs = dict(
        features_extractor_class=TransformerFeatures,
        features_extractor_kwargs=dict(
            features_dim=128,
            d_model=32,
            nhead=4,
            num_layers=2,
            dim_ff=128,
            dropout=0.1
        ),
        lstm_hidden_size=64,
        n_lstm_layers=1,
        shared_lstm=True,
        enable_critic_lstm=False,
        net_arch=dict(pi=[64], vf=[64]),
    )
    print("building recurrent ppo model")
    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=env,
        learning_rate=1e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=5,
        gae_lambda=0.95,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=LOG_DIR,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    checkpoint_loaded = False
    try:
        checkpoint_dir = f"{MODEL_DIR}/checkpoints"
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("ppo_") and f.endswith(".zip")]
            if checkpoints:
                def get_steps(filename):
                    try:
                        return int(filename.split("_")[1])
                    except (IndexError, ValueError):
                        return 0
                checkpoints.sort(key=get_steps)
                latest_checkpoint = checkpoints[-1]
                checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint[:-4])
                model = RecurrentPPO.load(checkpoint_path=checkpoint_path, env=env, device=device)
                print(f"Loaded checkpoint: {latest_checkpoint} - resuming training")
                checkpoint_loaded = True
    except Exception as e:
        print(f"Could not load checkpoint: {e}")
    
    if not checkpoint_loaded:
        print("No checkpoint found, trying BC initialization")
        try:
            bc_checkpoint = torch.load("models/bc/best_bc.pth", map_location="cpu")
            bc_features_state = bc_checkpoint["features_state_dict"]
            model.policy.features_extractor.load_state_dict(bc_features_state)
            print("BC weights loaded into PPO feature extractor [OK]")
        except FileNotFoundError:
            print("Warning: BC checkpoint not found, starting PPO from random weights")
    
    #gamma 0 means greedy, thinks only about currnt rewards. higher gamma thinks ahead
    #clip range determines change in policy. smaller clipping is better
    total_params = sum(p.numel() for p in model.policy.parameters())
    trainable = sum(
        p.numel() for p in model.policy.parameters() if p.requires_grad
    )
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable:,}")
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=f"{LOG_DIR}/eval",
        eval_freq=10000,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=50000,
        save_path=f"{MODEL_DIR}/checkpoints",
        name_prefix="ppo",
    )
    logging_cb = Loggingclbk(f"{LOG_DIR}/detailed")
    print(f"Starting training for {TOTAL_TIMESTEPS:,} timesteps")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_cb, checkpoint_cb, logging_cb],
    )
    model.save(f"{MODEL_DIR}/final_model")
    env.close()
    eval_env.close()
    
    return model

def watch(model_path="models/rl_ppo/final_model", episodes=5):
    env = gym.make(
        "autonomous-driving-v1",
        render_mode="human",
        domain_rand=False
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RecurrentPPO.load(model_path, device=device)
    
    for ep in range(episodes):
        obs, _ = env.reset()
        lstm_states = None
        episode_start = np.array([True])
        total_reward = 0
        
        while True:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=True,  # always pick best action
            )
            episode_start = np.array([False])
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            env.render()
            
            if terminated or truncated:
                print(f"Episode {ep+1}: reward={total_reward:.1f} "
                      f"passed={info.get('passed_obstacle', False)}")
                break
    env.close()
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--model", default="models/rl_policy/final_model")
    args = parser.parse_args()
    
    if args.watch:
        watch(args.model)
    else:
        train()


                         
    

