# Model selection - run `5fold_seed0_split42_675a0c14`

Criterion: mean outer-CV RMSE (lower is better).  Tie rule: winner is 'not clearly separated' from a rival if the mean paired per-fold RMSE gap is below 1 paired SE.

## Best direct model: `Direct|VotingOriginal`
mean RMSE 3.3483 (std over folds 0.0412), MAE 2.6425, R2 0.3805, Pearson 0.6181, mean training time 31.9s

- vs `Direct|VotingET`: RMSE gap +0.0036 +- 0.0037 (paired SE), winner better on 3/5 folds (**not clearly separated**)
- vs `Direct|XGBoost`: RMSE gap +0.0134 +- 0.0035 (paired SE), winner better on 5/5 folds
- vs `Direct|HistGradientBoosting`: RMSE gap +0.0228 +- 0.0024 (paired SE), winner better on 5/5 folds
- Not a clear winner: within 1 paired SE of ['Direct|VotingET']. The rule still selects the lowest mean RMSE; the cheapest of the winner and its near-ties is `Direct|VotingOriginal` - weigh stability, complexity and cost before preferring it.

## Best Two-Stage configuration: `TwoStage[S1=HistGradientBoosting]|VotingET`
mean RMSE 3.3437 (std over folds 0.0549), MAE 2.6314, R2 0.3821, Pearson 0.6188, mean training time 189.5s

- vs `TwoStage[S1=HistGradientBoosting]|VotingOriginal`: RMSE gap +0.0102 +- 0.0029 (paired SE), winner better on 5/5 folds
- vs `TwoStage[S1=HistGradientBoosting]|RandomForest`: RMSE gap +0.0213 +- 0.0070 (paired SE), winner better on 4/5 folds
- vs `TwoStage[S1=HistGradientBoosting]|ExtraTrees`: RMSE gap +0.0218 +- 0.0080 (paired SE), winner better on 4/5 folds

## Best overall: `TwoStage[S1=HistGradientBoosting]|VotingET`
mean RMSE 3.3437 (std over folds 0.0549), MAE 2.6314, R2 0.3821, Pearson 0.6188, mean training time 189.5s

- vs `Direct|VotingOriginal`: RMSE gap +0.0046 +- 0.0103 (paired SE), winner better on 3/5 folds (**not clearly separated**)
- vs `Direct|VotingET`: RMSE gap +0.0083 +- 0.0088 (paired SE), winner better on 3/5 folds (**not clearly separated**)
- vs `TwoStage[S1=HistGradientBoosting]|VotingOriginal`: RMSE gap +0.0102 +- 0.0029 (paired SE), winner better on 5/5 folds
- Not a clear winner: within 1 paired SE of ['Direct|VotingOriginal', 'Direct|VotingET']. The rule still selects the lowest mean RMSE; the cheapest of the winner and its near-ties is `Direct|VotingOriginal` - weigh stability, complexity and cost before preferring it.

## All candidates (mean +- std over outer folds)

|                                                        |   RMSE_mean |   RMSE_std |   MAE_mean |   R2_mean |   R2_std |   Pearson_mean |   seconds_mean |
|:-------------------------------------------------------|------------:|-----------:|-----------:|----------:|---------:|---------------:|---------------:|
| Direct|RandomForest                                    |      3.4101 |     0.0399 |     2.6893 |    0.3574 |   0.0392 |         0.5985 |         17.894 |
| Direct|XGBoost                                         |      3.3617 |     0.0478 |     2.6543 |    0.3755 |   0.0397 |         0.6136 |          4.458 |
| Direct|HistGradientBoosting                            |      3.3712 |     0.0414 |     2.6588 |    0.372  |   0.0385 |         0.6101 |          9.532 |
| Direct|ExtraTrees                                      |      3.4512 |     0.0409 |     2.7188 |    0.342  |   0.036  |         0.5857 |         23.934 |
| Direct|VotingOriginal                                  |      3.3483 |     0.0412 |     2.6425 |    0.3805 |   0.0376 |         0.6181 |         31.882 |
| Direct|VotingET                                        |      3.352  |     0.0386 |     2.6455 |    0.3792 |   0.0362 |         0.6172 |         55.816 |
| TwoStage[S1=HistGradientBoosting]|RandomForest         |      3.365  |     0.043  |     2.6507 |    0.3742 |   0.0389 |         0.6123 |        151.808 |
| TwoStage[S1=HistGradientBoosting]|XGBoost              |      3.3682 |     0.0685 |     2.6485 |    0.3727 |   0.0471 |         0.6117 |        137.362 |
| TwoStage[S1=HistGradientBoosting]|HistGradientBoosting |      3.3788 |     0.0641 |     2.6548 |    0.3689 |   0.0461 |         0.6085 |        142.104 |
| TwoStage[S1=HistGradientBoosting]|ExtraTrees           |      3.3655 |     0.0473 |     2.6507 |    0.3742 |   0.0364 |         0.6124 |        155.972 |
| TwoStage[S1=HistGradientBoosting]|VotingOriginal       |      3.354  |     0.058  |     2.6383 |    0.3782 |   0.0434 |         0.6157 |        166.096 |
| TwoStage[S1=HistGradientBoosting]|VotingET             |      3.3437 |     0.0549 |     2.6314 |    0.3821 |   0.0414 |         0.6188 |        189.488 |

## Limitation: selection bias

Selecting the minimum of several candidates from the same outer-CV scores makes the winner's CV score optimistic (selection bias). Nested CV was NOT run (computational cost). The validation-set score is an independent secondary check on data never used for fitting or ranking, but it does not remove the selection bias and is itself only independent while the configuration is not changed after seeing it.
