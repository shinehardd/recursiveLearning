from rl.networks.network import GaussianMLPPolicy
from rl.networks.network import CategoricalMLPPolicy

REGISTRY = {}

REGISTRY["gaussian_mlp_policy"] = GaussianMLPPolicy
REGISTRY["categorical_mlp_policy"] = CategoricalMLPPolicy