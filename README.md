# ConvectionMLT

Reference mixing-length convection kernel for HELIOS/VULCAN coupling work.

## Active package

The maintained implementation lives in [`convection_mlt/`](convection_mlt/):

```bash
cd convection_mlt
python -m pip install -e ".[test,plot]"
python -m pytest
mlt-baseline --output stage0/baseline-result.json
python stage1/plots/make_all.py --smoke   # CI-scale plot campaign
```

Stage 0 fixtures/tests and Stage 1 validation plots/evidence are under `convection_mlt/stage0/` and `convection_mlt/stage1/`.

## Legacy

Earlier exploratory solvers, sweeps, and notebooks are archived under [`Legacy/`](Legacy/).
