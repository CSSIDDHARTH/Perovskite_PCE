# Model selection - run `5fold_seed0_split42_60e10f71`

Criterion: mean outer-CV RMSE (lower is better).  Tie rule: winner is 'not clearly separated' from a rival if the mean paired per-fold RMSE gap is below 1 paired SE.

## Best direct model: `Direct|VotingET`
mean RMSE 3.3523 (std over folds 0.0522), MAE 2.6399, R2 0.4931, Pearson 0.7035, mean training time 71.4s

- vs `Direct|VotingOriginal`: RMSE gap +0.0045 +- 0.0038 (paired SE), winner better on 3/5 folds
- vs `Direct|XGBoost`: RMSE gap +0.0208 +- 0.0056 (paired SE), winner better on 5/5 folds
- vs `Direct|HistGradientBoosting`: RMSE gap +0.0350 +- 0.0083 (paired SE), winner better on 5/5 folds

## Best Two-Stage configuration: `TwoStage[S1=HistGradientBoosting]|VotingET`
mean RMSE 3.3324 (std over folds 0.0574), MAE 2.6086, R2 0.4991, Pearson 0.7073, mean training time 229.4s

- vs `TwoStage[S1=HistGradientBoosting]|VotingOriginal`: RMSE gap +0.0084 +- 0.0020 (paired SE), winner better on 5/5 folds
- vs `TwoStage[S1=HistGradientBoosting]|XGBoost`: RMSE gap +0.0187 +- 0.0035 (paired SE), winner better on 5/5 folds
- vs `TwoStage[S1=HistGradientBoosting]|RandomForest`: RMSE gap +0.0228 +- 0.0042 (paired SE), winner better on 5/5 folds

## Best overall: `TwoStage[S1=HistGradientBoosting]|VotingET`
mean RMSE 3.3324 (std over folds 0.0574), MAE 2.6086, R2 0.4991, Pearson 0.7073, mean training time 229.4s

- vs `TwoStage[S1=HistGradientBoosting]|VotingOriginal`: RMSE gap +0.0084 +- 0.0020 (paired SE), winner better on 5/5 folds
- vs `TwoStage[S1=HistGradientBoosting]|XGBoost`: RMSE gap +0.0187 +- 0.0035 (paired SE), winner better on 5/5 folds
- vs `Direct|VotingET`: RMSE gap +0.0199 +- 0.0038 (paired SE), winner better on 5/5 folds

## All candidates (mean +- std over outer folds)

|                                                        |   RMSE_mean |   RMSE_std |   MAE_mean |   R2_mean |   R2_std |   Pearson_mean |   seconds_mean |
|:-------------------------------------------------------|------------:|-----------:|-----------:|----------:|---------:|---------------:|---------------:|
| Direct|RandomForest                                    |      3.4179 |     0.0515 |     2.6945 |    0.4732 |   0.026  |         0.6888 |         23.442 |
| Direct|XGBoost                                         |      3.373  |     0.06   |     2.6559 |    0.4868 |   0.029  |         0.6987 |          5.274 |
| Direct|HistGradientBoosting                            |      3.3873 |     0.0653 |     2.6699 |    0.4824 |   0.0305 |         0.6955 |         11.156 |
| Direct|ExtraTrees                                      |      3.4383 |     0.0385 |     2.69   |    0.467  |   0.0222 |         0.6846 |         31.518 |
| Direct|VotingOriginal                                  |      3.3568 |     0.0571 |     2.6457 |    0.4917 |   0.0277 |         0.7025 |         39.87  |
| Direct|VotingET                                        |      3.3523 |     0.0522 |     2.6399 |    0.4931 |   0.0264 |         0.7035 |         71.388 |
| TwoStage[S1=HistGradientBoosting]|RandomForest         |      3.3552 |     0.0584 |     2.63   |    0.4922 |   0.0281 |         0.7023 |        183.604 |
| TwoStage[S1=HistGradientBoosting]|XGBoost              |      3.3511 |     0.0569 |     2.6219 |    0.4935 |   0.0267 |         0.7036 |        164.086 |
| TwoStage[S1=HistGradientBoosting]|HistGradientBoosting |      3.3667 |     0.0643 |     2.6363 |    0.4887 |   0.0295 |         0.7    |        169.466 |
| TwoStage[S1=HistGradientBoosting]|ExtraTrees           |      3.3639 |     0.0468 |     2.6324 |    0.4897 |   0.0244 |         0.7006 |        187.324 |
| TwoStage[S1=HistGradientBoosting]|VotingOriginal       |      3.3408 |     0.0599 |     2.6156 |    0.4966 |   0.0279 |         0.7055 |        200.406 |
| TwoStage[S1=HistGradientBoosting]|VotingET             |      3.3324 |     0.0574 |     2.6086 |    0.4991 |   0.0271 |         0.7073 |        229.354 |

## Limitation: selection bias

Selecting the minimum of several candidates from the same outer-CV scores makes the winner's CV score optimistic (selection bias). Nested CV was NOT run (computational cost). The validation-set score is an independent secondary check on data never used for fitting or ranking, but it does not remove the selection bias and is itself only independent while the configuration is not changed after seeing it.
