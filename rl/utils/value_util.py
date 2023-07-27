import torch
from types import SimpleNamespace
from rl.networks.network import ValueNetwork

def monte_carlo_returns(
        config: SimpleNamespace,
        state: torch.FloatTensor,
        next_state: torch.FloatTensor,
        reward: torch.FloatTensor,
        done: torch.int,
        critic: ValueNetwork = None) -> torch.FloatTensor:

    # Total time steps
    n_steps = reward.shape[-2]

    # Monte Carlo estimate of returns
    returns = reward.clone()

    # view transformation (3D -> 2D)
    returns = returns.view(-1, n_steps)      # [n_steps,1] -> [1,n_steps]
    done = done.view(-1, n_steps)          # [n_steps,1] -> [1,n_steps]

    for t in reversed(range(n_steps-1)):
        returns[:, t] += (1-done[:, t])*config.gamma*returns[:, t+1]

    # Normalizing the rewards
    if config.return_standardization:
        returns = (returns - returns.mean(dim=1, keepdim=True)) / \
                  (returns.std(dim=1, keepdim=True) + config.epsilon)

    returns = returns.view_as(reward)  # [1,n_steps] -> [n_steps,1]
    advantage = returns

    if critic is not None:
        with torch.no_grad():
            value = critic(state)
        advantage = returns - value

    return returns, advantage

def n_step_return(
        config: SimpleNamespace,
        state: torch.FloatTensor,
        next_state: torch.FloatTensor,
        reward: torch.FloatTensor,
        done: torch.int,
        critic: ValueNetwork,
    ) -> (torch.FloatTensor, torch.FloatTensor):

    # Total time steps
    n_steps = reward.shape[-2]

    with torch.no_grad():
        value = critic(state)

    # n step returns
    returns = reward.clone()
    returns[-1, :] = value[-1, :]  # bootstrap V(s_t_n)

    # view transformation (3D -> 2D)
    returns = returns.view(-1, n_steps)         # [n_steps,1] -> [1,n_steps]
    done = done.view(-1, n_steps)           # [n_steps,1] -> [1,n_steps]

    for t in reversed(range(n_steps-1)):
        returns[:, t] += (1 - done[:, t]) * config.gamma * returns[:, t+1]

    # Normalizing the rewards
    if config.return_standardization:
        returns = (returns - returns.mean(dim=1, keepdim=True)) / \
                  (returns.std(dim=1, keepdim=True) + config.epsilon)

    returns = returns.view_as(reward)  # [1,n_steps] -> [n_steps,1]
    advantage = returns - value

    return returns, advantage

def gae_advantages(
        config: SimpleNamespace,
        state: torch.FloatTensor,
        next_state: torch.FloatTensor,
        reward: torch.FloatTensor,
        done: torch.int,
        critic: ValueNetwork,
        ) -> (torch.FloatTensor, torch.FloatTensor):

    # Total time steps
    n_steps = state.shape[-2]

    with torch.no_grad():
        value = critic(state)
        next_value = critic(next_state)

    # Compute TD-residual
    delta = reward + (1 - done)*config.gamma*next_value - value

    # Generalized Advantage Estimation
    gae = delta.clone()
    gae = gae.view(-1, n_steps)      # [n_steps,1] -> [1,n_steps]
    done = done.view(-1, n_steps)  # [n_steps,1] -> [1,n_steps]

    for t in reversed(range(n_steps-1)):
        gae[:, t] += (1-done[:, t])*config.gamma*config.gae_lambda*gae[:, t+1]

    if config.gae_standardization:
        gae = (gae - gae.mean(dim=1, keepdim=True)) / \
              (gae.std(dim=1, keepdim=True) + config.epsilon)

    gae = gae.view_as(delta)   # [1,n_steps] -> [n_steps,1]
    returns = gae + value
    return returns, gae

REGISTRY = {}

REGISTRY["mc"] = monte_carlo_returns
REGISTRY["n_step"] = n_step_return
REGISTRY["gae"] = gae_advantages
