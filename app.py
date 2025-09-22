import streamlit as st
from ple.games.pixelcopter import Pixelcopter
from ple import PLE
import torch
import imageio
import numpy as np
import pandas as pd

from reinforce_agent import PolicyGradientAgent, train_reinforce
from ppo_agent import PPOAgent, train_ppo

def record_video(p, agent, filename, max_steps=1000):
    p.display_screen = True
    p.reset_game()
    state = p.getGameState()
    frames = []
    for t in range(max_steps):
        if p.game_over():
            break
        action = agent.select_action(state)
        p.act(action)
        frame = p.getScreenRGB()
        frames.append(frame)
        state = p.getGameState()
    imageio.mimsave(filename, frames, fps=30)
    print(f"Saved video to {filename}")

def main():
    st.title("PPO vs. REINFORCE for Pixelcopter")

    # Game setup
    game = Pixelcopter(width=48, height=48)
    p = PLE(game, fps=30, display_screen=False)
    p.init()
    action_set = p.getActionSet()
    state_keys = list(p.getGameState().keys())

    # Agent initialization
    reinforce_agent = PolicyGradientAgent(allowed_actions=action_set, state_keys=state_keys)
    ppo_agent = PPOAgent(allowed_actions=action_set, state_keys=state_keys)

    nb_episodes = st.sidebar.slider("Number of episodes", 50, 2000, 500)
    max_steps = st.sidebar.slider("Max steps per episode", 100, 2000, 1000)
    update_timestep = st.sidebar.slider("PPO Update Timestep", 100, 5000, 2000)

    if st.button("Train Agents"):
        st.header("Training in progress...")

        # Combined chart
        st.subheader("Training Rewards")
        chart_placeholder = st.empty()
        rewards_df = pd.DataFrame(columns=["episode", "REINFORCE", "PPO"])

        # REINFORCE training
        st.subheader("REINFORCE")
        def reinforce_callback(episode, reward):
            rewards_df.loc[episode, "episode"] = episode
            rewards_df.loc[episode, "REINFORCE"] = reward
            chart_placeholder.line_chart(rewards_df.set_index("episode"))

        train_reinforce(p, reinforce_agent, nb_episodes, max_steps, callback=reinforce_callback)
        st.write("REINFORCE training complete.")
        record_video(p, reinforce_agent, "reinforce_agent.mp4", max_steps)

        # PPO training
        st.subheader("PPO")
        def ppo_callback(episode, reward):
            rewards_df.loc[episode, "PPO"] = reward
            chart_placeholder.line_chart(rewards_df.set_index("episode"))

        train_ppo(p, ppo_agent, nb_episodes, max_steps, update_timestep, callback=ppo_callback)
        st.write("PPO training complete.")
        record_video(p, ppo_agent, "ppo_agent.mp4", max_steps)

        st.header("Training Complete")
        st.subheader("Final Videos")
        col1, col2 = st.columns(2)
        with col1:
            st.video("reinforce_agent.mp4")
        with col2:
            st.video("ppo_agent.mp4")

if __name__ == '__main__':
    main()
