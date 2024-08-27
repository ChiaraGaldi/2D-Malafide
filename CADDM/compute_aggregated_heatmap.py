import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

def read_matrices_from_folder(folder_path, label):
    matrices = []
    for filename in os.listdir(folder_path):
        if filename.endswith(f'{label}.npy'):
            file_path = os.path.join(folder_path, filename)
            matrix = np.load(file_path)
            matrices.append(matrix)
    return matrices

def compute_average_matrix(matrices):
    if not matrices:
        raise ValueError("No matrices found in the folder.")
    return np.mean(matrices, axis=0)

def save_matrix(matrix, file_path):
    np.save(file_path, matrix)

def save_matrix_as_image(matrix, file_path):
    plt.imshow(matrix, cmap='viridis')  # You can change the colormap if needed
    plt.colorbar()
    plt.savefig(file_path)
    plt.close()

def main(folder_path, output_path):
    output_file = os.path.join(output_path, 'aggregated_heatmap_spoof.npy')
    matrices = read_matrices_from_folder(os.path.join(folder_path, 'grad_cam'), 'spoof')
    average_matrix = compute_average_matrix(matrices)
    save_matrix(average_matrix, output_file)
    
    image_file = output_file.replace('.npy', '.png')
    save_matrix_as_image(average_matrix, image_file)

    output_file = os.path.join(output_path, 'aggregated_heatmap_bonafide.npy')
    matrices = read_matrices_from_folder(os.path.join(folder_path, 'grad_cam'), 'bonafide')
    average_matrix = compute_average_matrix(matrices)
    save_matrix(average_matrix, output_file)
    
    image_file = output_file.replace('.npy', '.png')
    save_matrix_as_image(average_matrix, image_file)

    output_file = os.path.join(output_path, 'aggregated_heatmap_malafide.npy')
    matrices = read_matrices_from_folder(os.path.join(folder_path, 'grad_cam_malafide'), 'spoof')
    average_matrix = compute_average_matrix(matrices)
    save_matrix(average_matrix, output_file)
    
    image_file = output_file.replace('.npy', '.png')
    save_matrix_as_image(average_matrix, image_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute the average of numpy matrices in a folder and save the result as an image.")
    parser.add_argument('folder_path', type=str, help="Path to the folder containing .npy matrices")
    parser.add_argument('output_path', type=str, help="Path to save the average matrix")
    
    args = parser.parse_args()
    main(args.folder_path, args.output_path)
