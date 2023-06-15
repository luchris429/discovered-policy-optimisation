import optax
import jax.numpy as jnp
from jax import jit, vmap, pmap
import jax
from flax import linen as nn
import os
import functools
import os.path as osp
from datetime import datetime
from brax import envs
from brax import jumpy as jp
from evosax import OpenES, SimpleGA, CMA_ES, ParameterReshaper, FitnessShaper, NetworkMapper
from evosax.utils import ESLog
import wandb
import argparse

from brax_drift import train

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, required=True)
parser.add_argument("--save-dir", type=str, required=True)
parser.add_argument("--outer_algo", type=str, default="OPEN_ES")
parser.add_argument("--pop-size", type=int, default=32)
parser.add_argument("--pmap", action="store_true")
parser.add_argument("--log", action="store_true")
parser.add_argument("--ppo-init", action="store_true")
parser.add_argument("--end-only", action="store_true")
args = parser.parse_args()


if __name__ == "__main__":
    """
    INIT INNER AGENT THINGS
    """
    env_fn = envs.create_fn(env_name=args.env)
    env = env_fn()
    state = env.reset(rng=jp.random_prngkey(seed=0))

    class DriftNet(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=128, name="hidden", use_bias=False)(x)
            x = nn.tanh(x)
            out = nn.Dense(features=1, name="out", use_bias=False)(x)
            return out

    rng = jax.random.PRNGKey(0)
    pholder = jnp.zeros((8,))

    drift_network = DriftNet()
    params = drift_network.init(
        rng,
        x=pholder,
    )

    param_reshaper = ParameterReshaper(params["params"])

    train_fn = {
        "ant": functools.partial(
            train,
            environment_fn=env_fn,
            num_timesteps=30_000_000,
            log_frequency=20,
            reward_scaling=10,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=5,
            num_minibatches=32,
            num_update_epochs=4,
            discounting=0.97,
            learning_rate=3e-4,
            entropy_cost=1e-2,
            num_envs=2048,
            batch_size=1024,
            drift_apply=drift_network.apply,
            ppo_init=args.ppo_init,
        ),
        "humanoid": functools.partial(
            train,
            environment_fn=env_fn,
            num_timesteps=50_000_000,
            log_frequency=20,
            reward_scaling=0.1,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=10,
            num_minibatches=32,
            num_update_epochs=8,
            discounting=0.97,
            learning_rate=3e-4,
            entropy_cost=1e-3,
            num_envs=2048,
            batch_size=1024,
            drift_apply=drift_network.apply,
            ppo_init=args.ppo_init,
        ),
        "fetch": functools.partial(
            train,
            environment_fn=env_fn,
            num_timesteps=100_000_000,
            log_frequency=20,
            reward_scaling=5,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=20,
            num_minibatches=32,
            num_update_epochs=4,
            discounting=0.997,
            learning_rate=3e-4,
            entropy_cost=0.001,
            num_envs=2048,
            batch_size=256,
            drift_apply=drift_network.apply,
            ppo_init=args.ppo_init,
        ),
        "grasp": functools.partial(
            train,
            environment_fn=env_fn,
            num_timesteps=600_000_000,
            log_frequency=10,
            reward_scaling=10,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=20,
            num_minibatches=32,
            num_update_epochs=2,
            discounting=0.99,
            learning_rate=3e-4,
            entropy_cost=0.001,
            num_envs=2048,
            batch_size=256,
            drift_apply=drift_network.apply,
            ppo_init=args.ppo_init,
        ),
        "halfcheetah": functools.partial(
            train,
            environment_fn=env_fn,
            num_timesteps=100_000_000,
            log_frequency=10,
            reward_scaling=1,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=20,
            num_minibatches=32,
            num_update_epochs=8,
            discounting=0.95,
            learning_rate=3e-4,
            entropy_cost=0.001,
            num_envs=2048,
            batch_size=512,
            drift_apply=drift_network.apply,
            ppo_init=args.ppo_init,
        ),
        "ur5e": functools.partial(
            train,
            environment_fn=env_fn,
            num_timesteps=20_000_000,
            log_frequency=20,
            reward_scaling=10,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=5,
            num_minibatches=32,
            num_update_epochs=4,
            discounting=0.95,
            learning_rate=2e-4,
            entropy_cost=1e-2,
            num_envs=2048,
            batch_size=1024,
            drift_apply=drift_network.apply,
            ppo_init=args.ppo_init,
        ),
        "reacher": functools.partial(
            train,
            environment_fn=env_fn,
            num_timesteps=100_000_000,
            log_frequency=20,
            reward_scaling=5,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=4,
            unroll_length=50,
            num_minibatches=32,
            num_update_epochs=8,
            discounting=0.95,
            learning_rate=3e-4,
            entropy_cost=1e-3,
            num_envs=2048,
            batch_size=256,
            drift_apply=drift_network.apply,
            ppo_init=args.ppo_init,
        ),
        "walker2d": functools.partial(
            train,
            environment_fn=env_fn,
            num_timesteps=50_000_000,
            log_frequency=10,
            reward_scaling=0.1,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=20,
            num_minibatches=32,
            num_update_epochs=8,
            discounting=0.97,
            learning_rate=0.0003,
            entropy_cost=0.001,
            num_envs=2048,
            batch_size=256,
            drift_apply=drift_network.apply,
            ppo_init=args.ppo_init,
        ),
    }[args.env]
    config = {}
    config["OUTER_ALGO"] = args.outer_algo
    config["DRIFT"] = True
    config["ENV"] = args.env
    config["N_DEVICES"] = jax.local_device_count()
    config["POP_SIZE"] = args.pop_size
    config["PMAP"] = args.pmap
    config["PPO_INIT"] = args.ppo_init

    def single_rollout(rng_input, drift_params):
        params, metrics = train_fn(key=rng_input, drift_params={"params": drift_params})
        return metrics.completed_episodes_metrics["reward"][-1] / metrics.completed_episodes[-1]

    vmap_rollout = jax.vmap(single_rollout, in_axes=(0, None))
    if args.pmap:
        rollout = jax.pmap(jax.jit(jax.vmap(vmap_rollout, in_axes=(None, param_reshaper.vmap_dict))))
    else:
        rollout = jax.jit(jax.vmap(vmap_rollout, in_axes=(None, param_reshaper.vmap_dict)))

    popsize = int(config["POP_SIZE"])
    if args.outer_algo == "OPEN_ES":
        strategy = OpenES(popsize=popsize, num_dims=param_reshaper.total_params, opt_name="adam")
    elif args.outer_algo == "SIMPLE_GA":
        strategy = SimpleGA(popsize=popsize, num_dims=param_reshaper.total_params, elite_ratio=0.5)
    elif args.outer_algo == "CMA_ES":
        strategy = CMA_ES(popsize=popsize, num_dims=param_reshaper.total_params, elite_ratio=0.5)
    else:
        raise NotImplementedError

    es_params = strategy.default_params

    num_generations = 1025
    num_rollouts = 1
    save_every_k_gens = 16

    es_logging = ESLog(param_reshaper.total_params, num_generations, top_k=5, maximize=True)
    log = es_logging.initialize()

    fit_shaper = FitnessShaper(centered_rank=False, z_score=True, w_decay=0.0, maximize=True)

    state = strategy.initialize(rng, es_params)

    config["SAVE_DIR"] = args.save_dir

    if args.log:
        wandb.init()

    for gen in range(num_generations):
        rng, rng_init, rng_ask, rng_eval = jax.random.split(rng, 4)
        x, state = strategy.ask(rng_ask, state, es_params)
        reshaped_params = param_reshaper.reshape(x)
        batch_rng = jax.random.split(rng_eval, num_rollouts)
        if args.pmap:
            batch_rng_pmap = jnp.tile(batch_rng, (jax.local_device_count(), 1, 1))
            fitness = rollout(batch_rng_pmap, reshaped_params).reshape(-1, num_rollouts).mean(axis=1)
        else:
            fitness = rollout(batch_rng, reshaped_params).mean(axis=1)
        fit_re = fit_shaper.apply(x, fitness)
        state = strategy.tell(x, fit_re, state, es_params)
        log = es_logging.update(log, x, fitness)

        print(f"Generation: {gen}, Best: {log['log_top_1'][gen]}, Fitness: {fitness.mean()}")
        if args.log:
            wandb.log(
                {
                    "Best Training Score": log["log_top_1"][gen],
                    "Fitness": fitness.mean(),
                }
            )
        if gen % save_every_k_gens == 0:
            jnp.save(osp.join(args.save_dir, f"curr_param_{gen}.npy"), state["mean"])
