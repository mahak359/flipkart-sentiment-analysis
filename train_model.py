# ==================================================
# FLIPKART SENTIMENT ANALYSIS (OPTUNA + MLFLOW PIPELINE)
# ==================================================

import pandas as pd
import re
import nltk
import optuna
import mlflow
import mlflow.sklearn
from optuna.integration.mlflow import MLflowCallback

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, classification_report

import joblib
import warnings
warnings.filterwarnings("ignore")

# ----------------------------
# Download NLTK data
# ----------------------------
nltk.download("stopwords")
nltk.download("wordnet")

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("reviews_badminton/data.csv")

# ----------------------------
# Create sentiment label
# ----------------------------
df = df[df["Ratings"] != 3]
df["sentiment"] = df["Ratings"].apply(lambda x: 1 if x >= 4 else 0)

# ----------------------------
# Clean text
# ----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

df["clean_review"] = df["Review text"].apply(clean_text)

# ----------------------------
# Train test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_review"], df["sentiment"],
    test_size=0.2, stratify=df["sentiment"], random_state=42
)

# ----------------------------
# Pipeline
# ----------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("model", LogisticRegression())
])

# ==================================================
# OPTUNA OBJECTIVES
# ==================================================

def objective_lr(trial):
    pipeline.set_params(
        tfidf__max_features=trial.suggest_int("max_features", 3000, 8000, step=1000),
        tfidf__ngram_range=trial.suggest_categorical("ngram_range", [(1,1), (1,2)]),
        model=LogisticRegression(
            C=trial.suggest_float("C", 0.01, 10, log=True),
            max_iter=1000
        )
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    return cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1").mean()

def objective_svm(trial):
    pipeline.set_params(
        tfidf__max_features=trial.suggest_int("max_features", 3000, 8000, step=1000),
        tfidf__ngram_range=trial.suggest_categorical("ngram_range", [(1,1), (1,2)]),
        model=LinearSVC(
            C=trial.suggest_float("C", 0.01, 10, log=True)
        )
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    return cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1").mean()

def objective_nb(trial):
    pipeline.set_params(
        tfidf__max_features=trial.suggest_int("max_features", 3000, 8000, step=1000),
        tfidf__ngram_range=trial.suggest_categorical("ngram_range", [(1,1), (1,2)]),
        model=MultinomialNB(
            alpha=trial.suggest_float("alpha", 0.01, 1.0)
        )
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    return cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1").mean()

# ==================================================
# MODEL LOOP
# ==================================================
objectives = {
    "LogisticRegression": objective_lr,
    "SVM": objective_svm,
    "NaiveBayes": objective_nb
}

mlflow.set_experiment("FLIPKART_SENTIMENT_ANALYSIS")

for model_name, obj in objectives.items():
    print(f"\n--- Optimizing {model_name} ---")

    mlflow_cb = MLflowCallback(
        metric_name="cv_f1",
        mlflow_kwargs={"nested": True}
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=15, callbacks=[mlflow_cb])

    best_params = study.best_params
    print(f"Best CV F1: {study.best_value:.4f}")

    # ----------------------------
    # FIX PARAMETER NAMES FOR PIPELINE
    # ----------------------------
    fixed_params = {}
    for k, v in best_params.items():
        if k in ["max_features", "ngram_range"]:
            fixed_params[f"tfidf__{k}"] = v
        elif k == "C":
            fixed_params["model__C"] = v
        elif k == "alpha":
            fixed_params["model__alpha"] = v

    pipeline.set_params(**fixed_params)
    pipeline.fit(X_train, y_train)

    # ----------------------------
    # Evaluate
    # ----------------------------
    y_pred = pipeline.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    print(f"\nTest F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred))

    # ----------------------------
    # Save model
    # ----------------------------
    joblib.dump(pipeline, f"{model_name}_sentiment_model.pkl")
    mlflow.sklearn.log_model(pipeline, model_name)
    mlflow.log_metric("test_f1", f1)
    mlflow.end_run()

print("\n✅ TRAINING COMPLETED SUCCESSFULLY")
