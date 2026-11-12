"""Power sampling: tegrastats on Jetson (whole-board input), nvidia-smi elsewhere."""

from __future__ import annotations

import re
import shutil
import statistics
import subprocess
import threading
import time

# tegrastats prints rails as  NAME current/averagemW  (newer JetPack) or
# NAME current/average  (older). VDD_IN is total board input on AGX Orin.
_RAIL = re.compile(
    r"(VDD_[A-Z0-9_]+|VIN_[A-Z0-9_]+)\s+(\d+)mW/(\d+)mW|(VDD_[A-Z0-9_]+|VIN_[A-Z0-9_]+)\s+(\d+)/(\d+)"
)
_BOARD_RAILS = ("VDD_IN", "VIN_SYS_5V0", "POM_5V_IN")


def parse_tegrastats(line: str) -> dict[str, float]:
    """Return {rail: watts} for the instantaneous readings in one tegrastats line."""
    rails = {}
    for m in _RAIL.finditer(line):
        name = m.group(1) or m.group(4)
        mw = m.group(2) or m.group(5)
        rails[name] = int(mw) / 1000.0
    return rails


def total_board_watts(rails: dict[str, float]) -> float | None:
    for key in _BOARD_RAILS:
        if key in rails:
            return rails[key]
    return None


class PowerSampler:
    """Context manager that samples power in a background thread.

    ``source`` is None when no supported tool is installed; ``mean()`` is then None.
    """

    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self.samples: list[float] = []
        self.gpu_samples: list[float] = []
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        if shutil.which("tegrastats"):
            self.source: str | None = "tegrastats (board input)"
        elif shutil.which("nvidia-smi"):
            self.source = "nvidia-smi (GPU only)"
        else:
            self.source = None

    @property
    def _jetson(self) -> bool:
        return bool(self.source and self.source.startswith("tegrastats"))

    def _read(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._jetson:
                rails = parse_tegrastats(line)
                total = total_board_watts(rails)
                if total is not None:
                    self.samples.append(total)
                if "VDD_GPU_SOC" in rails:
                    self.gpu_samples.append(rails["VDD_GPU_SOC"])
            else:
                try:
                    self.samples.append(float(line.strip()))
                except ValueError:
                    pass

    def __enter__(self) -> PowerSampler:
        if self.source is None:
            return self
        if self._jetson:
            cmd = ["tegrastats", "--interval", str(self.interval_ms)]
        else:
            cmd = ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits",
                   "-lms", str(self.interval_ms)]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        time.sleep(0.5)  # let the first samples arrive, then start clean
        self.samples.clear()
        self.gpu_samples.clear()
        return self

    def __exit__(self, *exc) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def mean(self) -> float | None:
        return statistics.fmean(self.samples) if self.samples else None

    def gpu_mean(self) -> float | None:
        return statistics.fmean(self.gpu_samples) if self.gpu_samples else None
