# EKF front-end smoke test — 2026-08-20

1× A100-SXM4-40GB, us-east-1, ~50 minutes wall clock, **~$1.70**. Instance self-terminated
via an on-box watchdog; verified zero instances running afterwards.

## Verdict: PASS. The config trains, and the throughput estimate was wrong in our favour.

| check | result |
|---|---|
| config loads, data resolves | ✅ 52 files, 554 MB rsynced; all corpora found |
| torch / CUDA | ✅ 2.11.0+**cu128**, CUDA live (cu130 trap avoided) |
| FA2 | ✅ `kernels` installed, **no non-finite loss** across 586 steps |
| loss descending | ✅ 15.47 → 12.63 |
| **throughput** | **18.5 samples/s**, stable |
| skipped samples | 225 (alignment + gold-capacity; expected and documented) |
| only "error" in the log | `KeyboardInterrupt` — our own time cap. Nothing else. |

`rc=124` is the `timeout` cap firing, which is the intended end of a smoke run.

## The measurement that matters

**18.5 samples/s**, against an extrapolated 12.1 for a 40 GB A100 — so **every cost estimate
made before this run was ~34% too high.** Full run is 6 epochs over 70,968 steps ≈ 17.0 h on
one A100.

| instance | $/hr | GPUs | train | +setup | **cost** |
|---|--:|--:|--:|--:|--:|
| `gpu_1x_a100_sxm4` | 1.99 | 1 | 17.0 h | 17.5 h | **$35** |
| `gpu_2x_h100_sxm5` | 8.38 | 2 | 5.9 h | 6.4 h | **$54** |
| `gpu_8x_a100_80gb_sxm4` | 22.32 | 8 | 2.4 h | 2.9 h | **$65** |

## Two findings worth keeping

**`num_workers` is not the bottleneck — do not re-try it.** 18.4 samples/s at 0 workers,
18.5 at 4, same box and config. GPU utilisation swings 28–81% while memory stays flat at
10,599 MiB, so the idling is variable sequence length under sliding-window, not input
starvation. Gradient checkpointing would *raise* utilisation, which is what ruled that out
first. `run_nw0.log` is the 0-worker baseline, `run.log` the 4-worker arm.

**Memory is the untested lever.** 10.6 GB of 40 (26%), flat across the whole run.
`batch_size` is where the headroom is, and raising it is the first thing to try if the full
run needs to be cheaper — but it was left alone here, one variable at a time.

## Watchdog

The box terminated itself. `box_run.sh` issues the Lambda terminate API call **from the box**,
so losing network on the operator's side cannot leave a $1.99/hr instance running. Three
independent stops: a `timeout` on training so the log survives, a hard deadline that fires
even if the script wedges, and a bounded pull window. `watchdog.log` records the call.

One operational note: `pkill -f box_run` matches the incoming ssh command string and kills
your own session mid-command. Collect PIDs with `pgrep -f "[b]ox_run"` and kill by PID.
