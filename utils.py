import os
import cv2
import json
import pandas as pd
import matplotlib.pyplot as plt

def get_img_from_selected_images(dataset_path, scenario, round, img_nr):
    img_path = dataset_path + f'{scenario}_0{round}_{img_nr}_original.png'
    if os.path.isfile(img_path):
        img = cv2.imread(img_path)
    else:
        print(f'Img {img_path} not available')
        img = None

    return img, img_path

def load_paths_from_config_file(config_path):
    """
    Load a JSON configuration file.

    Parameters:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Dictionary containing the loaded configuration.
    """
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        return config['paths']
    except FileNotFoundError:
        print(f"Error: File not found at {config_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")

def load_config_file(config_path):
    """
    Load a JSON configuration file.

    Parameters:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Dictionary containing the loaded configuration.
    """
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        print(f"Error: File not found at {config_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")

def plot_dataset_classes(annotations_path):
    df = pd.read_csv(annotations_path)
    true_counts = df[['is_field_boundary', 'is_field_marking', 'is_not_a_field_feature']].sum()

    # Plot the bar chart
    true_counts.plot(kind='bar', color=['blue', 'green', 'red'])
    plt.title('Count of True Values in Columns')
    plt.ylabel('Number of True Values')
    plt.xticks(rotation=0)
    plt.show()

if __name__ == "__main__":
    # Example usage
    config_path = "configs.json"
    config = load_paths_from_config_file(config_path)

    if config:
        print(config)

    plot_dataset_classes(config['segments_annotations'])