import time
import numpy as np
import matplotlib.pyplot as plt
from feature_extraction import load_images, load_labels
from naive_bayes import NaiveBayesClassifier
from perceptron import Perceptron

def train_and_evaluate_percentage(X_train, y_train, X_test, y_test, classifier, percentages, classifier_name, trials=100):
    print(f"\n--- Evaluating {classifier_name} ---")
    num_samples = X_train.shape[0]
    all_accuracies = {pct: [] for pct in percentages}
    all_times = {pct: [] for pct in percentages}

    for pct in percentages:
        for _ in range(trials):
            sample_size = int(num_samples * (pct / 100))
            indices = np.random.choice(num_samples, sample_size, replace=False)
            X_sample, y_sample = X_train[indices], y_train[indices]

            start_time = time.time()
            classifier.train(X_sample, y_sample)
            elapsed_time = time.time() - start_time

            predictions = classifier.predict(X_test)
            accuracy = classifier.accuracy(y_test, predictions)

            all_accuracies[pct].append(accuracy * 100)
            all_times[pct].append(elapsed_time)

        avg_accuracy = np.mean(all_accuracies[pct])
        avg_time = np.mean(all_times[pct])
        print(f"{pct}% Training Data: Average Accuracy = {avg_accuracy:.2f}% over {trials} trials, Average Time = {avg_time:.4f}s")

    plot_attempts_and_average(percentages, all_accuracies, classifier_name, "Accuracy (%)")
    plot_attempts_and_average(percentages, all_times, classifier_name, "Training Time (s)", ylabel="Time (s)")

def plot_attempts_and_average(percentages, all_results, classifier_name, metric, ylabel="Accuracy (%)"):
    plt.figure(figsize=(12, 8))

    for pct in percentages:
        plt.scatter([pct] * len(all_results[pct]), all_results[pct], alpha=0.2, label=f'{pct}% Attempts')

    avg_results = [np.mean(all_results[pct]) for pct in percentages]
    plt.plot(percentages, avg_results, color='red', marker='o', label='Average', linewidth=2)

    plt.title(f"{classifier_name} {metric} Across Trials", fontsize=14)
    plt.xlabel("Training Data Percentage (%)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(percentages)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc="upper right" if metric == "Training Time (s)" else "lower right", fontsize=10)
    plt.show()

def main():
    digit_data_folder = "digitdata"
    digit_train_images = f"{digit_data_folder}/trainingimages"
    digit_train_labels = f"{digit_data_folder}/traininglabels"
    digit_test_images = f"{digit_data_folder}/testimages"
    digit_test_labels = f"{digit_data_folder}/testlabels"

    face_data_folder = "facedata"
    face_train_images = f"{face_data_folder}/facedatatrain"
    face_train_labels = f"{face_data_folder}/facedatatrainlabels"
    face_test_images = f"{face_data_folder}/facedatatest"
    face_test_labels = f"{face_data_folder}/facedatatestlabels"

    digit_image_height, digit_image_width = 28, 28
    face_image_height, face_image_width = 70, 60

    print("\n--- DIGIT DATA ---")
    X_train_digits = load_images(digit_train_images, digit_image_height, digit_image_width, max_images=5000)
    y_train_digits = load_labels(digit_train_labels)
    X_test_digits = load_images(digit_test_images, digit_image_height, digit_image_width, max_images=1000)
    y_test_digits = load_labels(digit_test_labels)

    nb_classifier_digits = NaiveBayesClassifier(num_classes=10, alpha=0.5)
    percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    train_and_evaluate_percentage(X_train_digits, y_train_digits, X_test_digits, y_test_digits, nb_classifier_digits, percentages, "Naive Bayes on Digit Data", trials=100)

    perceptron_classifier_digits = Perceptron(num_features=X_train_digits.shape[1], num_classes=10)
    train_and_evaluate_percentage(X_train_digits, y_train_digits, X_test_digits, y_test_digits, perceptron_classifier_digits, percentages, "Perceptron on Digit Data", trials=100)

    print("\n--- FACE DATA ---")
    X_train_faces = load_images(face_train_images, face_image_height, face_image_width, max_images=451)
    y_train_faces = load_labels(face_train_labels)
    X_test_faces = load_images(face_test_images, face_image_height, face_image_width, max_images=150)
    y_test_faces = load_labels(face_test_labels)

    nb_classifier_faces = NaiveBayesClassifier(num_classes=2, alpha=0.5)
    train_and_evaluate_percentage(X_train_faces, y_train_faces, X_test_faces, y_test_faces, nb_classifier_faces, percentages, "Naive Bayes on Face Data", trials=100)

    perceptron_classifier_faces = Perceptron(num_features=X_train_faces.shape[1], num_classes=2)
    train_and_evaluate_percentage(X_train_faces, y_train_faces, X_test_faces, y_test_faces, perceptron_classifier_faces, percentages, "Perceptron on Face Data", trials=100)

if __name__ == "__main__":
    main()
