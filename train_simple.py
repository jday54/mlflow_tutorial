import mlflow

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

depth = 4
clf = DecisionTreeClassifier(max_depth=depth)
print(f"Training model with depth of {depth}...")
clf.fit(X_train, y_train)
print("Training complete")
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Start MLflow
with mlflow.start_run() as run:

    # Track parameters
    mlflow.log_param("depth", depth)

    # Track metrics
    mlflow.log_metric("accuracy", accuracy)

    # Track model
    mlflow.sklearn.log_model(clf, "classifier")

print("MLflow is complete")
