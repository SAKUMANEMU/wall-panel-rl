"""Show one episode: select RRT or PPO once with independent probability 0.5."""
from demo_autonomous_ppo import main

if __name__=='__main__':main(default_controller='mixture')
