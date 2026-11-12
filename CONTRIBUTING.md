# Contributing to Grail

Grail is early. The most useful contributions right now are measurements.

## Benchmark results

If you run the [baseline script](benchmarks/baseline/) on hardware we have not
covered, open an issue with the **Benchmark result** template and attach the
JSON it writes. Include the power mode and clock settings; numbers without them
cannot be compared.

## Bugs

Open an issue with the **Bug report** template. Include the exact command, the
full output, and your hardware, OS, CUDA/JetPack, PyTorch and LeRobot versions.

## Pull requests

- Open an issue first for anything larger than a small fix, so we can agree on
  the approach before you spend time on it.
- Any change that claims a speedup must include before-and-after numbers from
  the benchmark scripts on the same hardware, and must not reduce task success.
- Keep pull requests focused on one change.

## License

By contributing you agree that your contributions are licensed under the
Apache License 2.0, the same license as the project.
