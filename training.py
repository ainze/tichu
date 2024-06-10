from stable_baselines3 import PPO

from tichuEnv import TichuEnv

# Create the environment
env = TichuEnv()

# Define the model
model = PPO('MlpPolicy', env, verbose=1)

# Train the model
model.learn(total_timesteps=100000)

# Save the model
model.save("ppo_tichu")

# Load the model
model = PPO.load("ppo_tichu")

# Evaluate the trained agent
episodes = 10
for episode in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()