# SatQuery AI — Change Analysis Specialist (Member 3)

Autonomous, reproducible, and demo-ready multimodal remote-sensing change analysis system designed for the **SatQuery AI** project.

Please see the full documentation in [`change_analysis/README.md`](change_analysis/README.md).

---

## Quick Start (Windows PowerShell)

```powershell
# 1. Run environment and hardware inspection
python change_analysis/hardware_check.py

# 2. Run the automated 4-case benchmark demonstration
python demo.py

# 3. Run the automated test suite
python -m pytest change_analysis/tests -v

# 4. Run the CLI tool on any bitemporal pair
python -m change_analysis.cli `
  --before change_analysis/examples/case_01/before.png `
  --after change_analysis/examples/case_01/after.png `
  --question "Has construction increased?"

# 5. Start the local FastAPI microservice
python change_analysis/api.py
```

All generated visual artifacts (change maps, overlays, 4-panel comparison figures) are saved to [`change_analysis/outputs/`](change_analysis/outputs/).
