import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

CSV_PATH = "posture_data.csv"
MODEL_PATH = "posture_model.pkl"

def main():
    # 1. load dataset
    df = pd.read_csv(CSV_PATH)
    X = df.drop("label", axis=1).values
    y = df["label"].values

    # 2. split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. pipeline: scaling + RandomForest classifier
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    random_state=42,
                ),
            ),
        ]
    )

    # 4. train
    print("[INFO] training model...")
    clf.fit(X_train, y_train)

    # 5. evaluate
    y_pred = clf.predict(X_test)
    print("\n[RESULT] classification report:")
    print(classification_report(y_test, y_pred))
    print("[RESULT] confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 6. save model
    joblib.dump(clf, MODEL_PATH)
    print(f"\n[INFO] model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
