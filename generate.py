from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris(as_frame=True)
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.to_csv("data/iris_data.csv", index=False)
print("Iris dataset saved to data/iris_data.csv")