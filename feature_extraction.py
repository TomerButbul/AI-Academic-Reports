import numpy as np

def load_images(file_path, image_height, image_width, max_images=None):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    images = []
    current_image = []

    for line in lines:
        row = [1 if char == '+' else (1.5 if char == '#' else 0) for char in line.strip()]
        row = row + [0] * (image_width - len(row))  # Pad rows to match image width
        current_image.append(row)

        if len(current_image) == image_height:
            flattened_image = np.concatenate(current_image)
            images.append(flattened_image)
            current_image = []

    if max_images:
        images = images[:max_images]

    return np.array(images) / 1.5

def extract_grid_features(images, image_height, image_width, grid_size=(10, 10)):
    num_samples, _ = images.shape
    reshaped_images = images.reshape(num_samples, image_height, image_width)
    grid_height = image_height // grid_size[0]
    grid_width = image_width // grid_size[1]

    grid_features = []
    for image in reshaped_images:
        features = []
        for i in range(0, image_height, grid_height):
            for j in range(0, image_width, grid_width):
                subregion = image[i:i+grid_height, j:j+grid_width]
                features.append(np.mean(subregion))  # Use mean intensity of each grid
        grid_features.append(features)

    return np.array(grid_features)

def load_labels(file_path):
    with open(file_path, 'r') as file:
        labels = [int(line.strip()) for line in file if line.strip()]
    return np.array(labels)
