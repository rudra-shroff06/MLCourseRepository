import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score


DATA_FILE = "IMT2024032_train_var2.csv"

NUM_FEATURES = 3

DEGREES = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

OUTER_FOLDS = 5
INNER_FOLDS = 5

RANDOM_STATE = 42


PARAM_GRID = [
    {
        "poly__degree": DEGREES,
        "model": [Ridge()],
        "model__alpha": [
            1e-4,
            1e-3,
            1e-2,
            1e-1,
            1.0,
            10.0
        ]
    },

    {
        "poly__degree": DEGREES,
        "model": [
            Lasso(
                max_iter=10000,
                tol=1e-6
            )
        ],
        "model__alpha": [
            1e-4,
            1e-3,
            1e-2,
            1e-1,
            1.0
        ]
    },

    {
        "poly__degree": DEGREES,
        "model": [
            ElasticNet(
                max_iter=10000,
                tol=1e-6
            )
        ],
        "model__alpha": [
            1e-4,
            1e-3,
            1e-2,
            1e-1,
            1.0
        ],
        "model__l1_ratio": [
            0.25,
            0.50,
            0.75
        ]
    }
]


df = pd.read_csv(DATA_FILE)

X = df[
    [f"x{i}" for i in range(1, NUM_FEATURES + 1)]
].to_numpy(dtype=np.float64)

y = df["y"].to_numpy(dtype=np.float64)


print("=" * 70)
print("DATASET")
print("=" * 70)

print(f"Number of samples : {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")


y_mean = np.mean(y)

baseline_predictions = np.full(
    y.shape,
    y_mean,
    dtype=np.float64
)

baseline_mse = mean_squared_error(
    y,
    baseline_predictions
)

baseline_rmse = np.sqrt(baseline_mse)


print("\n" + "=" * 70)
print("TARGET STATISTICS & BASELINE")
print("=" * 70)

print(
    f"Target range : "
    f"[{np.min(y):.6f}, {np.max(y):.6f}]"
)

print(
    f"Target mean  : "
    f"{np.mean(y):.6f}"
)

print(
    f"Target std   : "
    f"{np.std(y):.6f}"
)

print(
    f"Baseline MSE : "
    f"{baseline_mse:.6f}"
)

print(
    f"Baseline RMSE: "
    f"{baseline_rmse:.6f}"
)


pipeline = Pipeline([
    (
        "poly",
        PolynomialFeatures(
            include_bias=False
        )
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        Ridge()
    )
])


outer_cv = KFold(
    n_splits=OUTER_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)

outer_mses = []

outer_predictions = np.empty_like(
    y,
    dtype=np.float64
)

outer_selected_parameters = []


print("\n" + "=" * 70)
print("STARTING NESTED CROSS-VALIDATION")
print("=" * 70)


for outer_fold, (
    outer_train_idx,
    outer_test_idx
) in enumerate(
    outer_cv.split(X),
    start=1
):

    print("\n" + "-" * 70)
    print(
        f"OUTER FOLD {outer_fold}/{OUTER_FOLDS}"
    )
    print("-" * 70)

    X_outer_train = X[outer_train_idx]
    y_outer_train = y[outer_train_idx]

    X_outer_test = X[outer_test_idx]
    y_outer_test = y[outer_test_idx]

    print(
        f"Outer training samples: "
        f"{len(outer_train_idx)}"
    )

    print(
        f"Outer validation samples: "
        f"{len(outer_test_idx)}"
    )


    inner_cv = KFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRID,
        cv=inner_cv,
        scoring="neg_mean_squared_error",
        refit=True,
        n_jobs=-1,
        return_train_score=False
    )

    grid_search.fit(
        X_outer_train,
        y_outer_train
    )

    best_params = grid_search.best_params_
    best_inner_mse = -grid_search.best_score_


    print("\nBest inner parameters:")
    print(best_params)

    print(
        f"Best inner CV MSE: "
        f"{best_inner_mse:.6f}"
    )


    outer_predictions_fold = (
        grid_search.best_estimator_.predict(
            X_outer_test
        )
    )

    outer_predictions[
        outer_test_idx
    ] = outer_predictions_fold


    outer_mse = mean_squared_error(
        y_outer_test,
        outer_predictions_fold
    )

    outer_mses.append(
        outer_mse
    )

    outer_selected_parameters.append(
        best_params
    )

    print(
        f"Outer validation MSE: "
        f"{outer_mse:.6f}"
    )


nested_mse = mean_squared_error(
    y,
    outer_predictions
)

nested_rmse = np.sqrt(
    nested_mse
)

nested_r2 = r2_score(
    y,
    outer_predictions
)


print("\n" + "=" * 70)
print("NESTED CROSS-VALIDATION RESULTS")
print("=" * 70)

print("\nOuter-fold MSEs:")

for i, mse in enumerate(
    outer_mses,
    start=1
):
    print(
        f"  Fold {i}: {mse:.6f}"
    )

print()

print(
    f"Mean outer-fold MSE : "
    f"{np.mean(outer_mses):.6f}"
)

print(
    f"Pooled outer MSE    : "
    f"{nested_mse:.6f}"
)

print(
    f"Outer MSE std       : "
    f"{np.std(outer_mses):.6f}"
)

print(
    f"Pooled outer RMSE   : "
    f"{nested_rmse:.6f}"
)

print(
    f"Pooled outer R²     : "
    f"{nested_r2:.6f}"
)

print(
    f"Variance explained  : "
    f"{nested_r2 * 100:.2f}%"
)


print("\n" + "=" * 70)
print("PARAMETERS SELECTED IN EACH OUTER FOLD")
print("=" * 70)

for i, params in enumerate(
    outer_selected_parameters,
    start=1
):

    print(f"\nFold {i}:")

    print(
        f"  Degree    : "
        f"{params['poly__degree']}"
    )

    print(
        f"  Model     : "
        f"{params['model']}"
    )

    print(
        f"  Alpha     : "
        f"{params['model__alpha']}"
    )

    if "model__l1_ratio" in params:
        print(
            f"  L1 ratio  : "
            f"{params['model__l1_ratio']}"
        )


improvement = (
    (baseline_mse - nested_mse)
    / baseline_mse
    * 100
)


print("\n" + "=" * 70)
print("COMPARISON WITH BASELINE")
print("=" * 70)

print(
    f"Baseline MSE  : "
    f"{baseline_mse:.6f}"
)

print(
    f"Nested CV MSE : "
    f"{nested_mse:.6f}"
)

print(
    f"Improvement   : "
    f"{improvement:.2f}%"
)


if nested_mse < baseline_mse:
    print(
        "\nModel performs better than baseline."
    )
else:
    print(
        "\nModel performs worse than baseline."
    )


print("\n" + "=" * 70)
print("FINAL HYPERPARAMETER SEARCH USING ALL DATA")
print("=" * 70)


final_cv = KFold(
    n_splits=INNER_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)

final_search = GridSearchCV(
    estimator=pipeline,
    param_grid=PARAM_GRID,
    cv=final_cv,
    scoring="neg_mean_squared_error",
    refit=True,
    n_jobs=-1,
    return_train_score=False
)

final_search.fit(
    X,
    y
)

final_params = final_search.best_params_
final_cv_mse = -final_search.best_score_


print("\nFinal selected parameters:")
print(final_params)

print(
    f"\nFinal CV MSE: "
    f"{final_cv_mse:.6f}"
)


final_model = final_search.best_estimator_

final_degree = final_params[
    "poly__degree"
]

final_model_type = final_params[
    "model"
]

final_alpha = final_params[
    "model__alpha"
]


print("\n" + "=" * 70)
print("FINAL MODEL")
print("=" * 70)

print(
    f"Polynomial degree : "
    f"{final_degree}"
)

print(
    f"Model             : "
    f"{final_model_type}"
)

print(
    f"Alpha             : "
    f"{final_alpha}"
)

if "model__l1_ratio" in final_params:
    print(
        f"L1 ratio          : "
        f"{final_params['model__l1_ratio']}"
    )


fitted_poly = final_model.named_steps[
    "poly"
]

num_polynomial_features = (
    fitted_poly.n_output_features_
)


print(
    f"Original features  : "
    f"{NUM_FEATURES}"
)

print(
    f"Polynomial features: "
    f"{num_polynomial_features}"
)

print(
    f"Training samples   : "
    f"{len(X)}"
)

print("=" * 70)
