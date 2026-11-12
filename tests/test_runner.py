import threading
import time

import pytest

from grail import ChunkRunner


def chunk_from(obs, k):
    """Action i of a chunk computed from the observation at step s is (s, i)."""
    return [(obs, i) for i in range(k)]


def test_stale_actions_are_dropped_and_chunk_aligns_to_now():
    gate = threading.Event()

    def predict(obs):
        if obs > 0:  # the first chunk (start) returns at once; later ones wait for the test
            gate.wait(5)
        return chunk_from(obs, 10)

    # Observations are labelled with the step they are taken at.
    with ChunkRunner(predict, control_hz=100, refill_at=6) as r:
        r.start(0)  # chunk from step 0; its action i is for step i
        sent = [r.step(s) for s in range(4)]
        assert sent == [(0, 0), (0, 1), (0, 2), (0, 3)]
        assert r._pending is None  # 6 actions left, checked before popping, so not yet
        sent.append(r.step(4))  # 6 left -> inference starts on the step-4 observation
        assert r._pending is not None and r._pending_step == 4
        sent.append(r.step(5))  # inference still running: keep executing the old chunk
        assert sent[-2:] == [(0, 4), (0, 5)]
        gate.set()
        r._pending.result(5)
        a = r.step(6)  # chunk from step 4 arrives at step 6: drop actions 0-1, send action 2
        assert a == (4, 2)
        assert r.step(7) == (4, 3)
        assert r.stats.stale_actions_dropped == 2
        assert r.stats.misses == 0


def test_misses_fall_back_to_last_action_without_blocking():
    def predict(obs):
        if obs > 0:
            time.sleep(0.5)  # far slower than the queue lasts
        return chunk_from(obs, 3)

    with ChunkRunner(predict, control_hz=100, refill_at=1) as r:
        r.start(0)
        t0 = time.perf_counter()
        out = [r.step(s) for s in range(1, 8)]
        assert time.perf_counter() - t0 < 0.2, "step() must not wait for inference"
    assert out[:3] == [(0, 0), (0, 1), (0, 2)]
    assert out[3:] == [(0, 2)] * 4  # held the last action
    assert r.stats.misses == 4


def test_custom_fallback_is_used():
    with ChunkRunner(lambda o: [] if o else ["a"], control_hz=10, refill_at=0,
                     fallback=lambda last: "SAFE_STOP") as r:
        r.start(0)
        assert r.step(1) == "a"
        assert r.step(2) == "SAFE_STOP"


def test_run_loop_has_no_misses_when_inference_fits_the_budget():
    # 20 actions at 200 Hz = 100 ms per chunk; inference takes ~20 ms.
    def predict(obs):
        time.sleep(0.02)
        return chunk_from(obs, 20)

    sent = []
    with ChunkRunner(predict, control_hz=200) as r:
        stats = r.run(lambda: len(sent), sent.append, duration_s=0.6)
    assert stats.steps > 80
    assert stats.misses == 0
    assert stats.chunks >= 4


def test_run_loop_reports_misses_when_inference_is_too_slow():
    # 5 actions at 100 Hz = 50 ms per chunk; inference takes 150 ms.
    def predict(obs):
        time.sleep(0.15)
        return chunk_from(obs, 5)

    sent = []
    with ChunkRunner(predict, control_hz=100) as r:
        stats = r.run(lambda: len(sent), sent.append, duration_s=0.6)
    assert stats.misses > 0
    assert stats.stale_actions_dropped > 0


def test_auto_threshold_tracks_latency():
    with ChunkRunner(lambda o: list(range(50)), control_hz=50) as r:
        assert r._threshold() == 1  # nothing known yet, chunk length 0
        r.stats.inference_s = [0.1] * 10  # 100 ms at 50 Hz = 5 steps, + 1 margin
        assert r._threshold() == 6


def test_validation():
    with pytest.raises(ValueError):
        ChunkRunner(lambda o: [], control_hz=0)
    with ChunkRunner(lambda o: [1], control_hz=10) as r, pytest.raises(ValueError):
        r.run(lambda: 0, lambda a: None)
