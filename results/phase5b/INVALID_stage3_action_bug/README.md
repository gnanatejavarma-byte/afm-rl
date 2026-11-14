# INVALID STAGE 3 RUN (ACTION INVERSION & OBSERVATION SCRAMBLE BUG)

## Reason for Invalidation
During Phase 5a Section A audit, direct side-by-side inspection between `src/level2_env.py` and `src/megadepth_eval_env.py` revealed that action semantics were inverted in the initial MegaDepth evaluation harness:
- Training (`Level2Env`): Action 0 = +500, Action 1 = +1000, Action 2 = -500, Action 3 = STOP
- Initial MegaDepth Eval (`MegaDepthEvalEnv`): Action 0 = STOP, Action 1 = +500, Action 2 = -500, Action 3 = +1000
Additionally, observation dimensions 26-32 were scrambled, and `MAX_STEPS` was set to 5 instead of 15.

Consequently, when the trained Level 2 PPO policies emitted Action 3 intending to STOP immediately (60-78% of pairs), the uncorrected environment interpreted Action 3 as `+1000` and failed to terminate, causing 88.3% to 98.1% of episodes to truncate at step 5 at maximum budget (mean cumulative KP ~9,600).

## Archived Affected Files & Tables
1. `megadepth_stage3_afm_rl_rollouts.csv`: Raw rollout logs for 7,500 episodes (5 folds x 1,500 pairs) under inverted semantics.
2. `megadepth_afm_rl_folds_summary.csv`: Summary table reporting degraded AFM-RL v1 performance (Succ@10° ~71.87%, AUC@10 ~49.64%).
3. `megadepth_primary_comparisons.csv`: Comparisons involving AFM-RL v1 vs baselines.
4. `a3_rollout_diagnostics.csv`: Action histogram showing 97.8% Action 3.
5. `e_statistical_comparisons.csv`: Bootstrap comparisons of AFM-RL v1 under invalid rollouts.

## Replacement in Phase 5b
All Stage 3 rollouts are rerun using the corrected `MegaDepthEvalEnv` (which imports constants and the observation builder directly from `src/level2_env.py`), saved to `results/phase5b/megadepth_stage3_afm_rl_v1_fixed.csv`.
