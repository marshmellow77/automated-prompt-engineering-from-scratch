from datasets import load_dataset
import pandas as pd

dataset = load_dataset("lukaemon/bbh", "geometric_shapes", cache_dir="./bbh_nshapes_cache")
data = dataset["test"]
data = data.shuffle(seed=1234)

training = data.select(range(100))
df_train = pd.DataFrame({"question": training["input"], "answer": training["target"]})

test = data.select(range(100, 200))
df_test = pd.DataFrame({"question": test["input"], "answer": test["target"]})

df_train.to_csv("train.csv", index=False)
df_test.to_csv("test.csv", index=False)
