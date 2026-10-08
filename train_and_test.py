import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet


TRAIN_FILE = "IMT2024032_train_var2.csv"
TEST_FILE = "IMT2024032_test_var2.csv"
OUTPUT_FILE = "IMT2024032_pred_var2.csv"

FEATURE_COLUMNS = ["x1", "x2", "x3"]
TARGET_COLUMN = "y"

DEGREE = 12
MODEL_TYPE = "elasticnet"
ALPHA = 0.001
L1_RATIO = 0.5


train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df[TARGET_COLUMN]

X_test = test_df[FEATURE_COLUMNS]


if MODEL_TYPE.lower() == "ridge":
    model = Ridge(alpha=ALPHA)

elif MODEL_TYPE.lower() == "lasso":
    model = Lasso(
        alpha=ALPHA,
        max_iter=100000,
        tol=1e-6
    )

elif MODEL_TYPE.lower() == "elasticnet":
    model = ElasticNet(
        alpha=ALPHA,
        l1_ratio=L1_RATIO,
        max_iter=100000,
        tol=1e-6
    )

else:
    raise ValueError(
        "MODEL_TYPE must be 'ridge', 'lasso', or 'elasticnet'"
    )


pipeline = Pipeline([
    ("poly", PolynomialFeatures(
        degree=DEGREE,
        include_bias=False
    )),
    ("scaler", StandardScaler()),
    ("model", model)
])


pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

output_df = pd.DataFrame({
    "y": predictions
})

output_df.to_csv(OUTPUT_FILE, index=False)

print(f"Model trained using {len(X_train)} training samples.")
print(f"Generated {len(predictions)} predictions.")
print(f"Predictions saved to: {OUTPUT_FILE}")