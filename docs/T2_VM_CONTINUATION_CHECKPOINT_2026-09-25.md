# T2-GovTrust — VM Continuation Checkpoint — 2026-09-25

## Purpose

This checkpoint freezes the authoritative scientific execution state before migrating the remaining LOCO matrix from Google Colab to a persistent GPU VM. Colab sessions have repeatedly disconnected/stopped during long sequential runs. The migration is operational only; the scientific contract is unchanged.

## Frozen scientific contract

- Scientific execution SHA: `998a0e17b03986a7adf2edffec61d1e7e5618d96`
- Design report SHA256: `ebfdc18dbfeebd9d760868cae6e438f1e138cc4356e5093d3cf4b1cf9fddc628`
- Preprocessor fingerprint: `ab286be51a209e1e2cb91913cc40adae184a4c270b7491fff37eedffc6246dac`
- Matrix size: 48 cells = 4 held-out cities × 3 seeds × 4 conditions
- Conditions: nonprivate, clipped_no_noise, private epsilon=5, private epsilon=1
- Rounds: 20
- Batch size: 1024
- Learning rate: 0.02
- Max grad norm: 1.0
- Total effective epochs target: 1.0
- Selection: source-city 2024 validation only
- Internal evaluation: source-city 2025 only
- External evaluation: held-out city 2025 only
- Device: CUDA

## Authoritative matrix state

`MATRIX_COMPLETED=15/48`
`MATRIX_REMAINING=33`

NYC fold:
- 12/12 accepted
- `NYC_FULL_FOLD_REPLICATION_GATE=PASS`

Chicago fold accepted cells:
1. seed0 nonprivate — PASS
   - UUID: `14f01b60-5649-4bf1-a62c-ce1424b68ca3`
   - checkpoint SHA256: `0560ef8422b9eda6faf2556835ddc6096e5f63138bec566517e278ea4006fa82`
   - external MAE log1p-hours: `2.671832994138678`
   - source macro internal MAE log1p-hours: `1.869949624522053`

2. seed0 private epsilon=5 — PASS
   - UUID: `4dcc1921-5a79-4f44-a056-6ec826c4d5ad`
   - checkpoint SHA256: `d0147cd02f3447c262a84a86adb55d91e96529f7a41b2fef60e04b44ee8de231`
   - external MAE log1p-hours: `2.9852883550789464`
   - source macro internal MAE log1p-hours: `1.792394052326925`
   - fold realized epsilon: `4.997480635399884`

3. seed0 private epsilon=1 — PASS
   - UUID: `51bb0e01-a609-4c4a-9643-268ad0a6ef6f`
   - checkpoint SHA256: `c8365bcc6f1c1b3f8d6562637a388ffc3a2eae4b8b378143dfbcab79b9432632`
   - external MAE log1p-hours: `2.9855600092825796`
   - source macro internal MAE log1p-hours: `1.7919723354726182`
   - fold realized epsilon: `0.9966398141937871`

The Chicago seed0 epsilon=1 run independently satisfies the frozen execution SHA, clean Git state, source-only selection, 2025 held-out scope, checkpoint hash, and formal per-client privacy-accounting step equality.

## Interrupted / incomplete operational attempt

`MATRIX_chicago_seed0_clipped_no_noise_epsinf_attempt1.log`

The log contains the run specification and design/unlock gate only. There is no accepted completed matrix cell for Chicago seed0 clipped/no-noise in the authoritative progress file. Therefore it is **not counted**.

Do not delete the log. Preserve it as operational provenance.

## Exact next matrix key

`chicago|seed=0|clipped_no_noise|eps=inf`

The VM must start by rescanning the Drive result folder, skip all 15 accepted cells, and execute this missing cell only if no valid completed artifact already exists at VM start.

After that:
- Chicago seed1: 4 conditions
- Chicago seed2: 4 conditions
- Chicago target state: 12/12, global matrix 24/48
- then Boston 12 cells -> 36/48
- then Los Angeles 12 cells -> 48/48

## VM continuation requirements

1. Use exact scientific SHA `998a0e17b03986a7adf2edffec61d1e7e5618d96`.
2. Verify design-report SHA256 before any execution.
3. Use the same harmonized data and output root already used by Colab.
4. Scan and validate all existing UUID run folders before launching any new cell.
5. Never rerun a valid accepted matrix cell.
6. Fail closed on duplicate keys, incomplete artifacts, wrong SHA, dirty Git state, checkpoint hash mismatch, target leakage, or privacy-accounting mismatch.
7. Persist one log per attempt and update matrix progress only after the new cell passes post-run validation.
8. Preserve interrupted Colab logs.
9. Do not change seeds, hyperparameters, privacy settings, chronology, predictors, or estimands during the VM migration.
10. Stop at fold boundaries for audit: Chicago 24/48, Boston 36/48, Los Angeles 48/48.

## Continuation state

`EXECUTION_ENVIRONMENT_NEXT=GPU_VM`
`COLAB_LONG_RUNS_DEPRECATED_FOR_MATRIX_CONTINUATION=YES`
`SCIENTIFIC_CONTRACT_CHANGED=NO`
`AUTHORITATIVE_ACCEPTED_COUNT=15`
`NEXT_MATRIX_KEY=chicago|seed=0|clipped_no_noise|eps=inf`
