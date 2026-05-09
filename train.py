# train.py
# This file trains the PPO agent
# It runs the agent through many episodes
# Agent learns to scale S-Cache queues optimally
# This implements Algorithm 2 from the paper
# Training takes 8-12 hours - run overnight!

import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time
from environment import ServerlessEnvironment
from ppo_agent import PPOAgent
from simulator import AzureDataLoader
from config import *

# ─────────────────────────────────────────
# TRAINING LOGGER
# Records everything during training
# ─────────────────────────────────────────

class TrainingLogger:
    """
    Records training progress.
    Saves results so you can plot them later.
    """

    def __init__(self, save_path=RESULTS_PATH):
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

        # Lists to track metrics per episode
        self.episodes        = []
        self.rewards         = []
        self.cold_start_rates = []
        self.wmts            = []
        self.actor_losses    = []
        self.critic_losses   = []
        self.queue_caps      = []

        # Track start time
        self.start_time = time.time()

    def log_episode(self, episode, reward,
                    cold_rate, wmt,
                    actor_loss, critic_loss,
                    queue_capacities):
        """Log one episode of training"""
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.cold_start_rates.append(cold_rate)
        self.wmts.append(wmt)
        self.actor_losses.append(
            actor_loss if actor_loss else 0)
        self.critic_losses.append(
            critic_loss if critic_loss else 0)
        self.queue_caps.append(queue_capacities)

    def save_logs(self):
        """Save all logs to JSON file"""
        logs = {
            'episodes':          self.episodes,
            'rewards':           self.rewards,
            'cold_start_rates':  self.cold_start_rates,
            'wmts':              self.wmts,
            'actor_losses':      self.actor_losses,
            'critic_losses':     self.critic_losses,
            'queue_caps':        self.queue_caps
        }

        path = os.path.join(self.save_path,
                           'training_logs.json')
        with open(path, 'w') as f:
            json.dump(logs, f, indent=2)

        print(f"Logs saved to {path}")

    def plot_training_progress(self):
        """
        Plots training curves.
        Shows reward, cold start rate, and WMT
        over training episodes.
        Matches Figure 6 from the paper.
        """
        fig, axes = plt.subplots(1, 3,
                                  figsize=(15, 5))
        fig.suptitle('Training Progress',
                    fontsize=14)

        # Plot 1: Reward convergence
        axes[0].plot(self.episodes, self.rewards,
                    color='blue', alpha=0.7)
        axes[0].set_title('(a) Reward Convergence')
        axes[0].set_xlabel('Episodes')
        axes[0].set_ylabel('Average Reward')
        axes[0].grid(True, alpha=0.3)

        # Add smoothed line
        if len(self.rewards) > 10:
            smoothed = self._smooth(self.rewards, 10)
            axes[0].plot(self.episodes,
                        smoothed,
                        color='darkblue',
                        linewidth=2,
                        label='Smoothed')
            axes[0].legend()

        # Plot 2: Cold start rate
        axes[1].plot(self.episodes,
                    self.cold_start_rates,
                    color='red', alpha=0.7)
        axes[1].set_title('(b) Cold Starts Rate')
        axes[1].set_xlabel('Episodes')
        axes[1].set_ylabel('Cold Start Rate (%)')
        axes[1].grid(True, alpha=0.3)

        if len(self.cold_start_rates) > 10:
            smoothed = self._smooth(
                self.cold_start_rates, 10)
            axes[1].plot(self.episodes,
                        smoothed,
                        color='darkred',
                        linewidth=2,
                        label='Smoothed')
            axes[1].legend()

        # Plot 3: Wasted memory time
        axes[2].plot(self.episodes,
                    self.wmts,
                    color='green', alpha=0.7)
        axes[2].set_title('(c) Average WMT')
        axes[2].set_xlabel('Episodes')
        axes[2].set_ylabel('Avg WMT (sec)')
        axes[2].grid(True, alpha=0.3)

        if len(self.wmts) > 10:
            smoothed = self._smooth(self.wmts, 10)
            axes[2].plot(self.episodes,
                        smoothed,
                        color='darkgreen',
                        linewidth=2,
                        label='Smoothed')
            axes[2].legend()

        plt.tight_layout()

        # Save figure
        path = os.path.join(self.save_path,
                           'training_progress.png')
        plt.savefig(path, dpi=150,
                   bbox_inches='tight')
        print(f"Training plot saved to {path}")
        plt.close()

    def _smooth(self, values, window):
        """Smooths a list of values using moving average"""
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window)
            smoothed.append(
                np.mean(values[start:i+1]))
        return smoothed

    def print_progress(self, episode,
                       reward, cold_rate,
                       wmt, elapsed):
        """Prints training progress to terminal"""
        print(f"Episode {episode:>4}/{MAX_EPISODES} | "
              f"Reward: {reward:>7.4f} | "
              f"Cold%: {cold_rate:>6.2f}% | "
              f"WMT: {wmt:>8.2f}s | "
              f"Time: {elapsed:>6.1f}s")


# ─────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────

def train(theta=THETA, quick_test=False):
    """
    Main training loop.
    Trains PPO agent to scale S-Cache queues.
    
    theta: balance between cold starts and memory
    quick_test: if True runs only 5 episodes for testing
    
    Implements Algorithm 2 from paper.
    """

    print("=" * 60)
    print("CASR Training Starting")
    print("=" * 60)
    print(f"Theta:        {theta}")
    print(f"Max episodes: {MAX_EPISODES}")
    print(f"Delta:        {DELTA} calls per step")
    print("=" * 60)

    # ── Load Training Data ──
    print("\nLoading training data...")
    loader     = AzureDataLoader()
    train_data = []

    for day in TRAIN_DAYS:
        day_calls = loader.load_day(day)
        train_data.extend(day_calls)

    print(f"Total training calls: {len(train_data)}")

    # ── Create Agent ──
    state_dim  = NUM_QUEUES * 7
    action_dim = 3 ** NUM_QUEUES
    agent      = PPOAgent(state_dim, action_dim)

    print(f"\nPPO Agent created")
    print(f"State dim:  {state_dim}")
    print(f"Action dim: {action_dim}")

    # ── Create Logger ──
    logger = TrainingLogger()

    # ── Training Loop ──
    print("\nStarting training...")
    print("-" * 60)

    max_episodes = 5 if quick_test else MAX_EPISODES

    best_reward     = float('-inf')
    episode_rewards = []

    for episode in range(1, max_episodes + 1):

        episode_start = time.time()

        # Use subset of data per episode
        # Paper: 500,000 calls per episode
        calls_per_ep = (50000 if quick_test
                       else CALLS_PER_EPISODE)

        # Sample random subset of training data
        if len(train_data) > calls_per_ep:
            start_idx = np.random.randint(
                0,
                len(train_data) - calls_per_ep)
            episode_calls = train_data[
                start_idx:start_idx + calls_per_ep]
        else:
            episode_calls = train_data

        # Create fresh environment for this episode
        env   = ServerlessEnvironment(
            episode_calls, theta=theta)
        state = env.reset()

        # Episode tracking
        episode_reward     = 0.0
        episode_cold_rates = []
        episode_wmts       = []
        done               = False
        step_count         = 0

        # ── Episode Loop ──
        while not done:

            # Agent chooses action
            action, log_prob = agent.choose_action(state)

            # Environment processes DELTA calls
            next_state, reward, done = env.step(action)

            # Store experience
            agent.store_experience(
                state, action, log_prob,
                reward, next_state, done)

            # Update networks when buffer is ready
            if agent.buffer.is_ready():
                actor_loss, critic_loss = agent.update()
            else:
                actor_loss  = None
                critic_loss = None

            # Track metrics
            metrics = env.get_current_metrics()
            episode_cold_rates.append(
                metrics['cold_start_rate'])
            episode_wmts.append(metrics['wmt'])

            episode_reward += reward
            state           = next_state
            step_count     += 1

        # ── End of Episode ──
        episode_time = time.time() - episode_start

        # Calculate episode averages
        avg_cold_rate = np.mean(episode_cold_rates)
        avg_wmt       = np.mean(episode_wmts)

        # Get final losses
        final_actor_loss  = (agent.actor_losses[-1]
                            if agent.actor_losses
                            else 0)
        final_critic_loss = (agent.critic_losses[-1]
                            if agent.critic_losses
                            else 0)

        # Get queue capacities
        queue_caps = [q.capacity
                     for q in env.scache.queues]

        # Log this episode
        logger.log_episode(
            episode       = episode,
            reward        = episode_reward,
            cold_rate     = avg_cold_rate,
            wmt           = avg_wmt,
            actor_loss    = final_actor_loss,
            critic_loss   = final_critic_loss,
            queue_capacities = queue_caps)

        episode_rewards.append(episode_reward)

        # Print progress every PRINT_EVERY episodes
        if episode % PRINT_EVERY == 0:
            elapsed = time.time() - logger.start_time
            logger.print_progress(
                episode    = episode,
                reward     = episode_reward,
                cold_rate  = avg_cold_rate,
                wmt        = avg_wmt,
                elapsed    = elapsed)
            print(f"         Queue caps: {queue_caps} | "
                  f"Steps: {step_count}")

        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(MODEL_SAVE_PATH + "best/")

        # Save checkpoint every 50 episodes
        if episode % 50 == 0:
            agent.save(
                MODEL_SAVE_PATH +
                f"checkpoint_ep{episode}/")
            logger.save_logs()
            logger.plot_training_progress()
            print(f"\nCheckpoint saved at "
                  f"episode {episode}\n")

    # ── Training Complete ──
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    # Save final model and logs
    agent.save(MODEL_SAVE_PATH + "final/")
    logger.save_logs()
    logger.plot_training_progress()

    total_time = time.time() - logger.start_time
    print(f"\nTotal training time: "
          f"{total_time/3600:.2f} hours")
    print(f"Best reward achieved: {best_reward:.4f}")
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    print(f"Logs saved to:  {RESULTS_PATH}")

    return agent, logger


# ─────────────────────────────────────────
# QUICK TEST FUNCTION
# Tests training with just 5 episodes
# Run this first before full training
# ─────────────────────────────────────────

def quick_test():
    """
    Runs training for just 5 episodes.
    Use this to verify everything works
    before starting the 8-12 hour full training.
    """
    print("=" * 60)
    print("QUICK TEST MODE (5 episodes)")
    print("This should finish in 5-10 minutes")
    print("=" * 60)

    agent, logger = train(
        theta=THETA,
        quick_test=True)

    print("\n✅ Quick test passed!")
    print("Ready for full training.")
    print("\nTo start full training run:")
    print("  python train.py full")

    return agent, logger


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "full":
        # Full training
        print("Starting FULL training...")
        print("This will take 8-12 hours.")
        print("Keep laptop plugged in!")
        agent, logger = train(theta=THETA)
    else:
        # Quick test by default
        agent, logger = quick_test()