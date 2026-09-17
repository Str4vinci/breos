# Run logs

Runs are listed in the order they happened. Each driver writes its own status
file and refuses to start unless HEAD is the expected commit and the tracked
worktree is clean.

| driver | log | status | outcome |
|---|---|---|---|
| `driver.sh` | `rerun.log` | `status.json` | complete, exit 0. Full pipeline at `fa5779c` on the **superseded** candidate set and the absolute 4352 W cap. Bundle: `article1_062_rerun`. |
| `superseded-driver-cross.sh` | `superseded-cross.log` | `superseded-cross-status.json` | **killed deliberately.** Same superseded config. Outputs deleted; nothing from this attempt is shipped. |
| `driver-task3.sh` | `task3.log` | `status-task3.json` | complete, exit 0. End-of-life sweep, 27/27 checks. Re-run afterwards at `30c35cb` once its script was committed; the re-run's outputs are byte-identical. |
| `driver-rerun2.sh` | `rerun2.log` | `status-rerun2.json` | complete, exit 0. Full pipeline plus the four cost x degradation cells at `b3d0034`, on the **accepted** candidate set and the 1 C limit. Bundle: `article1_062_rerun_1c`. |

The killed attempt is kept so the record is complete, not because anything in
it is used.
