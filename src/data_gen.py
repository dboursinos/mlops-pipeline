from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pickle
import dvc.api
import subprocess

random_state = 42

X, y = make_classification(n_samples=1000, n_features=20, random_state=random_state)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state
)

with open("./data/X_train.pkl", "wb") as f:
    pickle.dump(X_train, f)
with open("./data/X_test.pkl", "wb") as f:
    pickle.dump(X_test, f)
with open("./data/y_train.pkl", "wb") as f:
    pickle.dump(y_train, f)
with open("./data/y_test.pkl", "wb") as f:
    pickle.dump(y_test, f)

try:
    subprocess.run(["dvc", "add", "./data"], check=True)
    subprocess.run(["dvc", "push"], check=True)
    print("Data pushed to remote")
except Exception as e:
    print(f"Error pushing data to remote with DVC: {e}")
