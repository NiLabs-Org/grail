"""Load a LeRobot VLA policy with Grail's options and call it with fixed noise.

Requires PyTorch and LeRobot (``pip install "grail-vla[lerobot]"``).

Baseline means the checkpoint as published: whatever dtype and compile settings
its own config sets. Grail's options are applied on top and recorded, so every
comparison says exactly what changed.
"""

from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass

DEFAULT_TASK = "put the black bowl on the plate"


@dataclass
class Options:
    """What Grail changes relative to the checkpoint's own config.

    ``None`` means leave the checkpoint's setting alone.
    """

    dtype: str | None = None  # "bfloat16" or "float32"
    compile: bool | None = None
    compile_mode: str | None = None  # e.g. "max-autotune", "reduce-overhead"
    denoise_steps: int | None = None  # num_inference_steps for flow-matching policies
    quantize: str | None = None  # "int8wo": int8 weight-only via torchao

    def changed(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


def _set_dtype(cfg, name: str) -> None:
    import torch

    if not hasattr(cfg, "dtype"):
        raise ValueError(f"{type(cfg).__name__} has no dtype option")
    current = cfg.dtype
    cfg.dtype = name if isinstance(current, str) else getattr(torch, name)


def _quantize_int8_weight_only(module) -> None:
    try:
        from torchao.quantization import quantize_
    except ImportError as e:  # pragma: no cover - depends on environment
        raise SystemExit("--quantize int8wo needs torchao: pip install torchao") from e
    try:
        from torchao.quantization import Int8WeightOnlyConfig

        quantize_(module, Int8WeightOnlyConfig())
    except ImportError:  # older torchao
        from torchao.quantization import int8_weight_only

        quantize_(module, int8_weight_only())


class LeRobotModel:
    """A loaded policy plus its pre/post processors, ready to time and compare."""

    def __init__(self, model_id: str, policy: str = "pi0", device: str = "cuda",
                 options: Options | None = None, task: str = DEFAULT_TASK):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.factory import get_policy_class

        self.torch = torch
        self.model_id = model_id
        self.policy_type = policy
        self.device = device
        self.options = options or Options()
        self.task = task

        cfg = PreTrainedConfig.from_pretrained(model_id)
        o = self.options
        if o.dtype is not None:
            _set_dtype(cfg, o.dtype)
        if o.compile is not None:
            cfg.compile_model = o.compile
        if o.compile_mode is not None:
            cfg.compile_mode = o.compile_mode
        if o.denoise_steps is not None:
            if not hasattr(cfg, "num_inference_steps"):
                raise ValueError(f"{policy} has no num_inference_steps option")
            cfg.num_inference_steps = o.denoise_steps
        self.effective = {
            "dtype": str(getattr(cfg, "dtype", None)),
            "compile_model": getattr(cfg, "compile_model", None),
            "compile_mode": getattr(cfg, "compile_mode", None),
            "num_inference_steps": getattr(cfg, "num_inference_steps", None),
            "quantize": o.quantize,
        }

        policy_cls = get_policy_class(policy)
        self.policy = policy_cls.from_pretrained(model_id, config=cfg).to(device).eval()
        if o.quantize == "int8wo":
            _quantize_int8_weight_only(self.policy.model)
        elif o.quantize is not None:
            raise ValueError(f"unknown quantize option {o.quantize!r}")

        self.pre, self.post = make_pre_post_processors(
            self.policy.config,
            pretrained_path=model_id,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.config = self.policy.config

    # -- inputs -------------------------------------------------------------

    def observation(self, seed: int = 0) -> dict:
        """A synthetic observation shaped like the policy's inputs (no batch dim)."""
        from lerobot.configs.types import FeatureType

        torch = self.torch
        g = torch.Generator().manual_seed(seed)
        obs = {}
        for key, feat in self.config.input_features.items():
            if feat.type == FeatureType.VISUAL:
                obs[key] = torch.rand(*feat.shape, generator=g)  # [C, H, W] in [0, 1]
            elif feat.type == FeatureType.STATE:
                obs[key] = torch.randn(*feat.shape, generator=g)
        obs["task"] = self.task
        return obs

    def noise(self, seed: int = 0):
        """Diffusion noise for one chunk, identical across models for a given seed."""
        torch = self.torch
        dim = getattr(self.config, "max_action_dim", None)
        if dim is None:
            return None
        g = torch.Generator().manual_seed(10_000 + seed)
        n = torch.randn(1, self.config.chunk_size, dim, generator=g, dtype=torch.float32)
        return n.to(self.device)

    # -- calls --------------------------------------------------------------

    def _sync(self) -> None:
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()

    def predict(self, obs: dict, noise=None):
        """Unnormalized actions for one chunk, shape [1, n_action_steps, action_dim]."""
        batch = self.pre(dict(obs))
        with self.torch.inference_mode():
            if noise is not None:
                actions = self.policy.predict_action_chunk(batch, noise=noise)
            else:
                actions = self.policy.predict_action_chunk(batch)
        return self.post(actions[:, : self.config.n_action_steps])

    def timed_call(self, obs: dict) -> tuple[float, float]:
        """One inference call. Returns (end-to-end seconds, model-only seconds)."""
        self._sync()
        t0 = time.perf_counter()
        batch = self.pre(dict(obs))
        self._sync()
        t1 = time.perf_counter()
        with self.torch.inference_mode():
            actions = self.policy.predict_action_chunk(batch)
        self._sync()
        t2 = time.perf_counter()
        self.post(actions[:, : self.config.n_action_steps])
        self._sync()
        return time.perf_counter() - t0, t2 - t1

    @property
    def inner(self):
        return getattr(self.policy, "model", None)

    def close(self) -> None:
        """Free GPU memory so a second model can be loaded in the same process."""
        del self.policy, self.pre, self.post
        gc.collect()
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()
            self.torch.cuda.reset_peak_memory_stats()
