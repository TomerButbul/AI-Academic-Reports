# plot_experiment_results.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the reduced results
df = pd.read_csv("results_reduced.csv")

# Setup
sns.set(style="whitegrid")
df['Reg'] = df.apply(lambda row: 'BN' if row.batch_norm else ('DO' if row.dropout else 'None'), axis=1)

# 1. Test Accuracy by Optimizer and Regularization
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="optimizer", y="test_accuracy", hue="Reg")
plt.title("Test Accuracy by Optimizer and Regularization")
plt.ylabel("Test Accuracy (%)")
plt.xlabel("Optimizer")
plt.legend(title="Regularization")
plt.tight_layout()
plt.savefig("plot1_accuracy_optimizer_regularization.png")
plt.close()

# 2. Test Accuracy vs Learning Rate (grouped by optimizer)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="lr", y="test_accuracy", hue="optimizer")
plt.title("Test Accuracy by Learning Rate and Optimizer")
plt.ylabel("Test Accuracy (%)")
plt.xlabel("Learning Rate")
plt.tight_layout()
plt.savefig("plot2_accuracy_learning_rate.png")
plt.close()

# 3. Train vs Test Accuracy (scatter)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="best_train_acc", y="test_accuracy", hue="optimizer", style="Reg")
plt.plot([50, 100], [50, 100], ls='--', color='gray')
plt.title("Train vs Test Accuracy")
plt.xlabel("Best Train Accuracy (%)")
plt.ylabel("Test Accuracy (%)")
plt.tight_layout()
plt.savefig("plot3_train_vs_test_accuracy.png")
plt.close()

# 4. Training Time by Batch Size
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="batch_size", y="train_time", hue="optimizer")
plt.title("Training Time by Batch Size")
plt.ylabel("Training Time (s)")
plt.xlabel("Batch Size")
plt.tight_layout()
plt.savefig("plot4_train_time_batch_size.png")
plt.close()

# 5. Epochs Run by Train/Test Split
df['split'] = df['train_pct'].apply(lambda x: f"{int(x*100)}% Train")
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="split", y="epochs_ran", hue="optimizer")
plt.title("Epochs Until Early Stopping by Data Split")
plt.ylabel("Epochs Run")
plt.xlabel("Data Split")
plt.tight_layout()
plt.savefig("plot5_epochs_split.png")
plt.close()

print("All plots saved to disk.")
