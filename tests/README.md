# Release preparation tests

Run from the repository root on macOS with Python 3.10 or later:

```sh
python3 -B -m unittest discover -s tests -p 'test_*.py' -v
python3 -B tests/test_final_archive_integration.py
```

Build the release assets with `python3 tools/build_release_zip.py` before the second command. The final archive integration script has a standalone `main()` and is **not executed by unittest discovery**. Both commands report in the terminal and do not create repository audit logs.

The discovery suite contains 94 test methods, including parameterized failure cases. The archive integration adds one separate scenario. Fixtures use temporary work packages and Support directories, synthetic input/result sentinels, local release transport, and mock Python/Fiji/model components. Every maintenance and installer child runs with a macOS network-deny sandbox. Fixture shells receive temporary `HOME` and `ZDOTDIR` values to avoid user startup files, plus a fixed system `PATH`.

Coverage includes fixed-tag release identity and digests, program path restrictions, corrupted/missing program recovery, executable permissions, transaction rollback and interrupted recovery, environment and workspace locks, maintenance integrity, installation without overwriting existing targets, and retention of input/results/run state. The final archive scenario restores 95 program files from the actual release ZIP using a local transport fixture and reaches the real maintenance shell chain.

These tests do not run image analysis, real model inference, a live Fiji GUI, dependency installation, or live GitHub release downloads. Published asset availability and live anonymous download checks require a separate post-publication verification.
