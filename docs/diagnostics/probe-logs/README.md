# gfx1201 probe logs (June 2026)

Raw `AMD_LOG_LEVEL=3` output from `scripts/probe_gfx1201_tile.py`,
`scripts/repro_gfx1201_dense.py` and the production kernel capture, compressed
with xz (about 70 MB raw, 1.5 MB here). These are the logs that
`docs/HANDOFF_GFX1201_UPSTREAM_AMD.md` refers to by bare file name.

Read one with `xz -dc probe_msweep.log.xz | less`.

Not included: `diag-sync-probe-step1230-20260605-124008.log`, the 26 GB
serialised capture of the faulting training step. It is too large for git and
exists only on the training host under `docs/diagnostics/`.
