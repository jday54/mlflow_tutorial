import mlflow
from mlflow.tracking import MlflowClient

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

EXPERIMENT_NAME = "network-depth"
ACTIVE_EXPERIMENT = mlflow.set_experiment(EXPERIMENT_NAME)
EXPERIMENT_ID = ACTIVE_EXPERIMENT.experiment_id


for idx, depth in enumerate([1, 2, 5, 10, 20]):
    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Start MLflow
    RUN_NAME = f"run_{idx}"
    with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=RUN_NAME) as run:
        # Retrieve run id
        RUN_ID = run.info.run_id

        # Attach a description to this run
        description = f"Run {RUN_ID} for depth {depth}"
        MlflowClient().set_tag(RUN_ID, "mlflow.note.content", description)

        # Track parameters
        mlflow.log_param("depth", depth)

        # Track metrics
        mlflow.log_metric("accuracy", accuracy)

        # Track model
        mlflow.sklearn.log_model(clf, "classifier")

print("end")
