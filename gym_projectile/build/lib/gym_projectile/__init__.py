from gymnasium.envs.registration import register

register(
    id='Projectile-v0',
    entry_point='gym_projectile.envs:ProjectileEnv',
)
