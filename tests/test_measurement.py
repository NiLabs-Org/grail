import math

import pytest

from grail.cli import main
from grail.power import parse_tegrastats, total_board_watts
from grail.stats import percentile, summarize_ms
from grail.verify import action_drift, summarize_drift


def test_percentile_and_summary():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 99) == 5
    assert percentile([3, 1, 2], 100) == 3
    with pytest.raises(ValueError):
        percentile([], 50)
    s = summarize_ms([0.1, 0.2, 0.3])
    assert s["p50"] == pytest.approx(200) and s["max"] == pytest.approx(300) and s["n"] == 3


def test_tegrastats_new_and_old_formats():
    new = ("RAM 9000/62841MB VDD_GPU_SOC 4321mW/4100mW VDD_CPU_CV 1200mW/1100mW "
           "VIN_SYS_5V0 5000mW/4900mW VDD_IN 18350mW/17000mW")
    old = "RAM 2000/7764MB VDD_IN 5123/5000 VDD_CPU_GPU_CV 1500/1400 VDD_SOC 1100/1000"
    r_new, r_old = parse_tegrastats(new), parse_tegrastats(old)
    assert r_new["VDD_GPU_SOC"] == pytest.approx(4.321)
    assert total_board_watts(r_new) == pytest.approx(18.35)  # VDD_IN preferred
    assert total_board_watts(r_old) == pytest.approx(5.123)
    assert total_board_watts({"VDD_SOC": 1.0}) is None


def test_drift_identical_and_scaled():
    ref = [[[0.1, -0.2, 0.3], [0.0, 0.5, -0.5]]]
    same = action_drift(ref, ref)
    assert same["max_abs"] == 0 and same["rel_l2"] == 0 and same["cosine"] == pytest.approx(1)
    scaled = action_drift(ref, [[[v * 1.1 for v in row] for row in ref[0]]])
    assert scaled["rel_l2"] == pytest.approx(0.1)
    assert scaled["cosine"] == pytest.approx(1)
    with pytest.raises(ValueError):
        action_drift([1, 2], [1, 2, 3])


def test_drift_summary_uses_worst_case():
    s = summarize_drift([{"max_abs": 0.1, "mean_abs": 0.01, "rel_l2": 0.02, "cosine": 0.999},
                         {"max_abs": 0.3, "mean_abs": 0.02, "rel_l2": 0.08, "cosine": 0.99}],
                        max_rel_l2=0.05)
    assert s["worst"]["rel_l2"] == 0.08 and s["worst"]["cosine"] == 0.99
    assert s["within_tolerance"] is False


def test_drift_zero_reference():
    d = action_drift([0, 0], [0, 0])
    assert d["rel_l2"] == 0 and d["cosine"] == 1
    assert math.isinf(action_drift([0, 0], [1, 0])["rel_l2"])


def test_compare_requires_an_option(capsys):
    assert main(["compare"]) == 2
    assert "needs at least one Grail option" in capsys.readouterr().err


def test_stage_timer_splits_pi0_style_calls_and_restores_methods():
    import types

    from grail.bench import StageTimer

    clock = [0.0]

    class Event:
        def __init__(self, **kwargs):
            self.t = None

        def record(self):
            self.t = clock[0]

        def elapsed_time(self, other):
            return other.t - self.t

    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(Event=Event, synchronize=lambda: None))

    class Inner:
        def embed_prefix(self):
            clock[0] += 5

        def denoise_step(self):
            clock[0] += 2

        def sample_actions(self):
            self.embed_prefix()
            clock[0] += 30  # backbone prefill
            for _ in range(10):
                self.denoise_step()

    m = Inner()
    t = StageTimer(m, fake_torch)
    assert t.complete
    m.sample_actions()
    assert t.collect_ms() == {"vision_and_embed": 5, "action_expert": 20,
                              "language_backbone_prefill": 30, "denoise_steps": 10}
    t.remove()
    assert not ({"embed_prefix", "denoise_step", "sample_actions"} & set(vars(m)))
