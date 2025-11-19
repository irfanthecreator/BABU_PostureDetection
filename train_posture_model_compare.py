import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

CSV_PATH = "posture_data.csv"
MODEL_PATH = "posture_model_best.pkl"

def main():
    # 1. load dataset
    df = pd.read_csv(CSV_PATH)
    X = df.drop("label", axis=1).values
    y = df["label"].values

    # 2. split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. define several models to compare
    models = {
        # tuned a bit compared to your original
        "RandomForest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=300,      # number of trees
                        max_depth=None,       # let trees grow deep
                        min_samples_split=2,  # try 2, can tune to 4
                        min_samples_leaf=1,   # minimum samples per leaf
                        random_state=42,
                    ),
                ),
            ]
        ),
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        max_iter=1000,
                        solver="lbfgs",
                        multi_class="auto",
                    ),
                ),
            ]
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svm",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        probability=True,  # so we can get predict_proba if we want
                    ),
                ),
            ]
        ),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "knn",
                    KNeighborsClassifier(
                        n_neighbors=5,
                        weights="distance",
                    ),
                ),
            ]
        ),
    }

    best_acc = -1.0
    best_name = None
    best_model = None

    # 4. train & evaluate each model
    for name, model in models.items():
        print("=" * 60)
        print(f"[INFO] training model: {name}")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"[RESULT] accuracy ({name}): {acc:.4f}")
        print("[RESULT] classification report:")
        print(classification_report(y_test, y_pred))
        print("[RESULT] confusion matrix:")
        print(confusion_matrix(y_test, y_pred))

        # remember best model
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = model

    print("=" * 60)
    print(f"[SUMMARY] best model: {best_name} with accuracy {best_acc:.4f}")

    # 5. save the best-performing model
    if best_model is not None:
        joblib.dump(best_model, MODEL_PATH)
        print(f"[INFO] best model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
