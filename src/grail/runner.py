"""Asynchronous action-chunk execution.

A VLA returns a chunk of k actions per inference call. Run synchronously, the
robot stops while the next chunk is computed. ``ChunkRunner`` instead executes
the current chunk at the control rate while the next one is computed in the
background, and handles the two things that go wrong:

* **Staleness.** A chunk computed from an observation taken at step s0 and
  ready at step s1 is already s1 - s0 steps old. Its first s1 - s0 actions
  describe moments that have passed, so they are dropped and execution
  continues from action s1 - s0.
* **Deadline misses.** If the queue runs dry before the next chunk is ready,
  the runner counts a miss and returns a fallback action (by default it holds
  the last action) instead of blocking the control loop.

The runner is pure Python and model-agnostic: ``predict`` is any callable that
maps an observation to a sequence of actions.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

Action = Any
Observation = Any


@dataclass
class RunnerStats:
    steps: int = 0
    misses: int = 0  # control steps with no fresh action available
    chunks: int = 0  # chunks integrated into the queue
    stale_actions_dropped: int = 0
    fully_stale_chunks: int = 0  # chunks that arrived with every action already in the past
    inference_s: list[float] = field(default_factory=list)
    tick_overruns: int = 0  # control-loop ticks that started late (only from run())

    @property
    def miss_rate(self) -> float:
        return self.misses / self.steps if self.steps else 0.0

    def as_dict(self) -> dict:
        from .stats import summarize_ms

        out = {
            "steps": self.steps,
            "misses": self.misses,
            "miss_rate": self.miss_rate,
            "chunks": self.chunks,
            "stale_actions_dropped": self.stale_actions_dropped,
            "fully_stale_chunks": self.fully_stale_chunks,
            "tick_overruns": self.tick_overruns,
        }
        if self.inference_s:
            out["inference_ms"] = summarize_ms(self.inference_s)
        return out


def hold_last(last: Action | None) -> Action | None:
    """Default fallback: repeat the last action that was sent."""
    return last


class ChunkRunner:
    """Run a chunked policy at a fixed control rate with background inference.

    Args:
        predict: maps an observation to a sequence of actions; action 0 is the
            action for the step at which the observation was taken.
        control_hz: control loop rate. Used by ``run`` and to size the refill
            threshold.
        refill_at: start the next inference when this many actions or fewer are
            left in the queue. ``None`` sizes it automatically from measured
            inference latency (p90 plus one step of margin).
        fallback: called with the last sent action when the queue is empty;
            its return value is sent instead. Use it to hold position, slow
            down, or trigger a safe stop.
    """

    def __init__(
        self,
        predict: Callable[[Observation], Sequence[Action]],
        *,
        control_hz: float,
        refill_at: int | None = None,
        fallback: Callable[[Action | None], Action | None] = hold_last,
        clock: Callable[[], float] = time.perf_counter,
    ):
        if control_hz <= 0:
            raise ValueError("control_hz must be positive")
        self.predict = predict
        self.control_hz = control_hz
        self.refill_at = refill_at
        self.fallback = fallback
        self.clock = clock
        self.stats = RunnerStats()
        self._queue: deque = deque()
        self._step = 0
        self._last: Action | None = None
        self._pending: Future | None = None
        self._pending_step = 0
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="grail-infer")
        self._chunk_len = 0

    # -- inference ----------------------------------------------------------

    def _infer(self, obs: Observation) -> tuple[list, float]:
        t0 = self.clock()
        actions = list(self.predict(obs))
        return actions, self.clock() - t0

    def _launch(self, obs: Observation) -> None:
        self._pending_step = self._step
        self._pending = self._pool.submit(self._infer, obs)

    def _integrate(self, actions: list, elapsed: float, taken_at: int) -> None:
        self.stats.inference_s.append(elapsed)
        self.stats.chunks += 1
        self._chunk_len = max(self._chunk_len, len(actions))
        delay = self._step - taken_at
        if delay >= len(actions):
            self.stats.fully_stale_chunks += 1
            self.stats.stale_actions_dropped += len(actions)
            return  # keep whatever is left of the previous chunk
        self.stats.stale_actions_dropped += delay
        # The new chunk supersedes the rest of the old one.
        self._queue = deque(actions[delay:])

    def _threshold(self) -> int:
        if self.refill_at is not None:
            return self.refill_at
        if not self.stats.inference_s:
            return max(1, self._chunk_len // 2)
        from .stats import percentile

        p90 = percentile(self.stats.inference_s, 90)
        return max(1, math.ceil(p90 * self.control_hz) + 1)

    # -- public API ---------------------------------------------------------

    def start(self, obs: Observation) -> None:
        """Compute the first chunk synchronously, so step() has actions to send."""
        actions, elapsed = self._infer(obs)
        self._integrate(actions, elapsed, taken_at=self._step)

    def step(self, obs: Observation) -> Action | None:
        """Advance one control step and return the action to send now.

        Call once per control tick with the latest observation. Never blocks
        on inference.
        """
        if self._pending is not None and self._pending.done():
            actions, elapsed = self._pending.result()
            self._pending = None
            self._integrate(actions, elapsed, self._pending_step)

        if self._pending is None and len(self._queue) <= self._threshold():
            self._launch(obs)

        self.stats.steps += 1
        if self._queue:
            action = self._queue.popleft()
            self._last = action
        else:
            self.stats.misses += 1
            action = self.fallback(self._last)
        self._step += 1
        return action

    def run(
        self,
        get_observation: Callable[[], Observation],
        send_action: Callable[[Action | None], None],
        *,
        duration_s: float | None = None,
        steps: int | None = None,
        stop: threading.Event | None = None,
    ) -> RunnerStats:
        """Fixed-rate control loop: observe, step, send, sleep until the next tick."""
        if duration_s is None and steps is None and stop is None:
            raise ValueError("give duration_s, steps or stop")
        period = 1.0 / self.control_hz
        if not self._queue and self._pending is None:
            self.start(get_observation())
        t_next = self.clock()
        t_end = t_next + duration_s if duration_s is not None else math.inf
        n = 0
        while True:
            if steps is not None and n >= steps:
                break
            if stop is not None and stop.is_set():
                break
            now = self.clock()
            if now >= t_end:
                break
            if now - t_next > period * 0.5:
                self.stats.tick_overruns += 1
            send_action(self.step(get_observation()))
            n += 1
            t_next += period
            sleep = t_next - self.clock()
            if sleep > 0:
                time.sleep(sleep)
            else:
                t_next = self.clock()  # fell behind; do not try to catch up in a burst
        return self.stats

    def close(self) -> None:
        self._pool.shutdown(wait=True)

    def __enter__(self) -> ChunkRunner:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
