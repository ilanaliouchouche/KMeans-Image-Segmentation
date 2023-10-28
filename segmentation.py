import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from pyclustertend import hopkins, ivat


class MyKMean:
    """KMeans class for image segmentation."""

    SEED = 42  # Seed for random operations

    def __init__(self, path: str, size_k: np.ndarray):
        """
        Initialize the image and cluster sizes.

        Parameters:
            path (str): Path to the image file.
            size_k (np.ndarray): Array of sizes for KMeans clustering.
        """
        self.path = path
        
        # Load the image and convert it to RGB format
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Check if the image is loaded successfully
        if img is None:
            raise Exception(f"Failed to load image from path: {path}")
        
        self.img = img
        self.size_k = size_k
        self.df_img = None  # Dataframe of flattened image
        self.h = None  # Hopkins statistic
        self.models = None  # Trained KMeans models
        self.all_silhouette = None  # Silhouette scores
        self.all_inertia_w = None  # Within-cluster sum of squares
        self.bench = None  # Benchmark results

    def display_image(self):
        """Display the image."""
        plt.imshow(self.img)
        plt.axis("off")
        plt.show()

    def crop_img(self, height: slice, width: slice):
        """
        Crop the image if needed (for time constraints).

        Parameters:
            height (slice): Height slice for cropping.
            width (slice): Width slice for cropping.
        """
        cropped_img = self.img[height, width, :]
        self.img = cropped_img

    def set_hob(self):
        """Compute the Hopkins statistic for clustering tendency."""
        if self.df_img is None:
            raise Exception("You must use set_DataFrame before!")
        
        hopkins_stat = hopkins(self.df_img, self.df_img.shape[0])
        self.h = hopkins_stat

    def display_ivat(self):
        """Display the iVAT visualisation for assessing cluster tendency."""
        if self.df_img is None:
            raise Exception("You must use set_DataFrame before!")
        
        plt.figure(figsize=(10, 8))
        ivat(self.df_img.values)
        plt.title("iVAT Visualisation")
        plt.show()

    def display_rgb(self):
        """Display the image channels in Red, Green, and Blue."""
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        channel_names = ["Reds", "Greens", "Blues"]

        for idx, color in enumerate(channel_names):
            axs[idx].imshow(self.img[:, :, idx], cmap=color)
            axs[idx].axis('off')

        plt.tight_layout()
        plt.show()

    def set_DataFrame(self):
        """Flatten the image and set it as a dataframe."""
        img_flat = self.img.reshape((-1, 3))
        self.df_img = pd.DataFrame(img_flat, columns=['R', 'G', 'B'])

    def display_3D(self, elev=30, azim=30):
        """Display a 3D scatter plot of RGB values of the image."""
        if self.df_img is None:
            raise Exception("You must use set_DataFrame before!")
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.df_img['R'], self.df_img['G'], self.df_img['B'], c='red', marker='o')

        ax.set_xlabel('R')
        ax.set_ylabel('G')
        ax.set_zlabel('B')
        ax.set_title('RGB 3D Scatter Plot')
        ax.view_init(elev=elev, azim=azim)
        plt.show()

    def compute_kmeans(self, k: int):
        """Train a KMeans model for a given number of clusters."""
        kmeans = KMeans(n_clusters=k, random_state=MyKMean.SEED)
        kmeans.fit(self.df_img)
        return kmeans

    def training(self):
        """Train KMeans models for all specified cluster sizes."""
        kmeans_list = Parallel(n_jobs=-1, verbose=10)(delayed(self.compute_kmeans)(k) for k in self.size_k)
        self.models = kmeans_list

    def display_centroids(self):
        """Display the color centroids for each KMeans clustering."""
        if not self.models:
            raise Exception("You must train before!")
        
        centroids_dict = {k: model.cluster_centers_ for k, model in zip(self.size_k, self.models)}

        for k, centroids in centroids_dict.items():
            plt.figure(figsize=(5, 1))
            for index, color in enumerate(centroids):
                plt.fill_between([index, index+1], 0, 1, color=color/255)  # Normalize to [0,1]
            plt.title(f'Colors for k = {k}')
            plt.xlim(0, len(centroids))
            plt.ylim(0, 1)
            plt.axis('off')
            plt.show()

    def compute_silhouette(self, model):
        """Compute the silhouette score for a given KMeans model."""
        if self.df_img is None:
            raise Exception("You must use set_DataFrame before!")
        return silhouette_score(self.df_img, model.labels_)

    def set_all_silhouette(self):
        """Compute silhouette scores for all trained KMeans models."""
        if self.models is None:
            raise Exception("You must train before!")
        
        sil_list = Parallel(n_jobs=-1, verbose=10)(delayed(self.compute_silhouette)(model) for model in self.models)
        self.all_silhouette = sil_list

    def compute_inertia_w(self, model):
        """Compute the within-cluster sum of squares for a given KMeans model."""
        return model.inertia_

    def set_all_inertiaw(self):
        """Compute the within-cluster sum of squares for all trained KMeans models."""
        if self.models is None:
            raise Exception("You must train before!")
        
        inertiaw_list = Parallel(n_jobs=-1, verbose=10)(delayed(self.compute_inertia_w)(model) for model in self.models)
        self.all_inertia_w = inertiaw_list

    def display_elbow_plot(self, idx_marker: int = None):
        """Display the elbow plot for determining the optimal number of clusters."""
        if self.all_inertia_w is None:
            raise Exception("You must use set_all_inertiaw before!")
        
        plt.plot(self.size_k, self.all_inertia_w)
        plt.title("Elbow Method")
        plt.ylabel("Within-cluster Sum of Squares")
        plt.xlabel("Number of Clusters")
        
        if idx_marker is not None:
            ki = {k: v for k, v in zip(self.size_k, self.all_inertia_w)}
            plt.plot(idx_marker, ki[idx_marker], marker='X', label="Optimal k")
            
        plt.legend()
        plt.show()

    def compute_segmented_image(self, model):
        """Get the segmented image based on clustering."""
        if self.df_img is None:
            raise Exception("You must use set_DataFrame before!")
        
        labels = model.predict(self.df_img)
        segmented_img_flat = model.cluster_centers_[labels]
        segmented_img = segmented_img_flat.reshape(self.img.shape).astype(np.uint8)
        return segmented_img

    def display_segmented_images(self, ax_grid=(2, 2), number=4):
        """Display segmented images for a specified number of cluster sizes."""
        fig, axs = plt.subplots(*ax_grid, figsize=(8, 10))
        axs = axs.flatten()

        for i, k in enumerate(self.size_k[:number]):
            segmented_img = self.compute_segmented_image(self.models[i])
            axs[i].imshow(segmented_img)
            axs[i].set_title(f"K = {k}")
            axs[i].axis("off")

        plt.tight_layout()
        plt.show()

    def benchmark(self):
        """Compare segmented images for different cluster sizes."""
        results = {}
        
        for i, k in enumerate(self.size_k):
            segmented_img = self.compute_segmented_image(self.models[i])
            temp_path = Path(f'temp_img_{k}.png')
            cv2.imwrite(str(temp_path), cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR))
            
            file_size = temp_path.stat().st_size / 1024  # Convert to KB
            
            temp_path.unlink()
            
            if self.all_inertia_w is None:
                raise Exception("You have to use set_all_inertiaw before!")
            
            inertia_w = self.all_inertia_w[i]
            
            if self.all_silhouette is not None:
                silhouette = self.all_silhouette[i]
                results[k] = {'inertia_w': inertia_w, 'file_size': file_size, 'silhouette': silhouette}
            else:
                results[k] = {'inertia_w': inertia_w, 'file_size': file_size}
                
        self.bench = results

    def display_benchmark(self, w=True):
        """Display the benchmark results."""
        if self.bench is None:
            raise Exception("Use benchmark before!")
        
        k_values = list(self.bench.keys())
        file_sizes = [entry['file_size'] for entry in self.bench.values()]

        if w:
            metric_values = [entry['inertia_w'] for entry in self.bench.values()]
            metric_label = "Within-cluster Sum of Squares"
        else:
            if 'silhouette' not in self.bench[k_values[0]]:
                raise Exception("Silhouette scores are not available. Ensure you've computed them with set_all_silhouette.")
            metric_values = [entry['silhouette'] for entry in self.bench.values()]
            metric_label = "Silhouette Score"

        point_sizes = [size * 70 for size in file_sizes]
        
        img_file = Path(self.path)
        img_size = img_file.stat().st_size / 1024
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(k_values, metric_values, s=point_sizes, c=file_sizes, cmap='viridis', alpha=0.6)
        cbar = plt.colorbar(scatter, label='File Size (KB)')
        cbar.ax.axhline(img_size, color='black', linestyle='-', linewidth=3)
        
        plt.xlabel('Number of Clusters')
        plt.ylabel(metric_label)
        plt.title(f'{metric_label} vs Number of Clusters (with File Size in KB)')
        plt.show()


