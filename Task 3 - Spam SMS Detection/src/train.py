from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_baseline_models(X, y):
    """
    Train baseline NB, LR, and SVM models.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    pipelines = {
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("model", MultinomialNB())
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("model", LogisticRegression(max_iter=1000))
        ]),
        "SVM": Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("model", LinearSVC())
        ])
    }

    for name, model in pipelines.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        print("=" * 50)
        print(name)
        print("Accuracy :", accuracy_score(y_test, preds))
        print("Precision:", precision_score(y_test, preds))
        print("Recall   :", recall_score(y_test, preds))
        print("F1 Score :", f1_score(y_test, preds))

    return X_train, X_test, y_train, y_test


def train_tuned_svm(X_train, y_train):
    """
    Train tuned SVM using GridSearchCV.
    """
    svm_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("model", SVC(kernel="linear", probability=True))
    ])

    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__max_df": [0.9, 1.0],
        "model__C": [0.1, 1, 10]
    }

    grid = GridSearchCV(
        estimator=svm_pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    grid.fit(X_train, y_train)
    print("BEST PARAMETERS:", grid.best_params_)

    return grid.best_estimator_
