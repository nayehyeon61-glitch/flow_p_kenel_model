# Vendored: physical_guide_flowmatching (PG-CFM)

- Source: `nayehyeon61-glitch/physical_guide_flowmatching`, branch `claude/project-thread-c97r6n`, commit `5d38e56`
  ("docs 4.13: the core at stages 2 and 3 only ...").
- Copied unchanged: `pgcfm/` (tendency core `DynCore`, primitive-equation operator `OperatorFeatures` = G(x),
  KAI data contract), `third_party/climode` (ClimODE ResNet used by the core), `third_party/kai` (imported by
  `pgcfm/kai_tend.py`), `scripts/run_kai_tendency.py` (core training, data loading, `build_core`),
  `scripts/wb2_to_kai.py` (ERA5 from WeatherBench2), `scripts/local_era5_to_kai.py` (ERA5 from a local 0.25 deg
  archive, e.g. the daily means on the GPU server: `--src-daily --daily-mean --grid 1.5`; added for the + version).
- Why vendored: cores trained with PG-CFM (`runs/kaitend/<tag>.pt` + `stats.npz`) load here unchanged
  (`fpk/env.py`). To update, copy the same paths from a newer commit and change the hash above.
