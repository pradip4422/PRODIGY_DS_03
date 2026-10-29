import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("bank.csv", sep=";")

print("Dataset loaded:", df.shape)
print(df.head())

X = pd.get_dummies(df.drop(columns=["y"]), drop_first=True, dtype=int)
y = df["y"].map({"no": 0, "yes": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = DecisionTreeClassifier(
    max_depth=5, random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\nDecision Tree Accuracy: {accuracy:.2%}")
print(classification_report(
    y_test, predictions,
    labels=[0, 1],
    target_names=["No Purchase", "Purchase"],
    zero_division=0
))

plt.figure(figsize=(6, 4))
sns.heatmap(
    confusion_matrix(y_test, predictions, labels=[0, 1]),
    annot=True, fmt="d", cmap="Blues",
    xticklabels=["No", "Yes"], yticklabels=["No", "Yes"]
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.close()

plt.figure(figsize=(20, 10))
plot_tree(
    model, max_depth=3, filled=True, rounded=True,
    class_names=["No Purchase", "Purchase"], fontsize=7
)
plt.savefig("outputs/decision_tree.png", bbox_inches="tight")
plt.close()

df["y"].value_counts().plot(kind="bar")
plt.title("Purchase Distribution")
plt.xlabel("Purchase (no/yes)")
plt.ylabel("Customers")
plt.tight_layout()
plt.savefig("outputs/purchase_distribution.png")
plt.close()

with open("outputs/model_results.txt", "w", encoding="utf-8") as f:
    f.write(f"Decision Tree Accuracy: {accuracy:.2%}\n")
    f.write(classification_report(
        y_test, predictions,
        labels=[0, 1],
        target_names=["No Purchase", "Purchase"],
        zero_division=0
    ))

print("\nSuccess! Check the outputs folder for charts and results.")