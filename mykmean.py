import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path

#classe Kmeans pour la ségmentation d'image

class MyKMean:

    SEED = 42

    #initialise l'image ainsi que le nombre de 
    def __init__(self, path: str, size_k :np.ndarray):
        self.path = path
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is None:
            raise Exception(f"Failed to load image from path: {path}")
        self.img = img
        self.size_k = size_k
        self.df_img = None
        self.models = None
        self.all_silhouette = None
        self.all_inertia_w = None
        self.bench = None
    
    def display_image(self):
        plt.imshow(self.img)
        plt.axis("off")
        plt.show()
    
    #si on souhaite reduire les dimensions de l'image (soucis de temps)
    def crop_img(self, height : slice, width : slice):
        croped_img = self.img[height,width,:]
        self.img = croped_img
    
    #critere de validation pour savoir si c'est pertinent de clusterer
    def hobkins(self):
        #TODO probleme avec pyclusteredtend
        pass
    
    #affiche l'image selon les filtres R G B
    def display_rgb(self):
        fig, axs = plt.subplots(1, 3, figsize=(10,5))
        channel_names = ["Reds", "Greens", "Blues"]

        for idx, color in enumerate(channel_names):  
            axs[idx].imshow(self.img[:,:,idx], cmap=color)
            axs[idx].axis('off')

        plt.tight_layout()
        plt.show()
    
    #on re cupere le dataframe avec l'image applatit
    def set_DataFrame(self):
        img_flat = self.img.reshape((-1, 3))
        self.df_img = pd.DataFrame(img_flat, columns = ['R', 'G', 'B'])
    
    #un entrainement de kmeans
    def compute_kmeans(self, k : int):
        kmeans = KMeans(n_clusters=k, random_state=MyKMean.SEED)
        kmeans.fit(self.df_img)
        return kmeans
    
    #entrainement pour tous les k
    def training(self):
        kmeans_list = Parallel(n_jobs=-1, verbose=10)(delayed(self.compute_kmeans)(k) for k in self.size_k)
        self.models = kmeans_list
    
    def display_centroids(self):
        if not self.models:
            raise Exception("You must train before !")
        centroids_dict = {}
        for k, model in zip(self.size_k, self.models):
            centroids = model.cluster_centers_
            centroids_dict[k] = centroids
        for k, centroids in centroids_dict.items():
            plt.figure(figsize=(5, 1))
            for index, color in enumerate(centroids):
                plt.fill_between([index, index+1], 0, 1, color=color/255)  # Divisez par 255 pour obtenir des valeurs RGB dans [0,1]
            plt.title(f'Colors for k = {k}')
            plt.xlim(0, len(centroids))
            plt.ylim(0, 1)
            plt.axis('off')
            plt.show()
        

    #metrique silhouette
    def compute_silhouette(self,model):
        if self.df_img is None:
            raise Exception("You must use set_DataFrame before !")
        return silhouette_score(self.df_img, model.labels_)
    
    #metrique silhouette pour tous les k
    def set_all_silhouette(self):
        if self.models is None:
            raise Exception("You must train before !")
        sil_list = Parallel(n_jobs=-1, verbose=10)(delayed(self.compute_silhouette)(model) for model in self.models)
        self.all_silhouette = sil_list
    
    #metrique inertie within
    def compute_inertia_w(self,model):
        return model.inertia_
    
    #pour tous les k
    def set_all_inertiaw(self):
        if self.models is None:
            raise Exception("You must train before !")
        inertiaw_list = Parallel(n_jobs=-1, verbose=10)(delayed(self.compute_inertia_w)(model) for model in self.models)
        self.all_inertia_w = inertiaw_list
    
    #methode du coude
    def display_elbow_plot(self, idx_marker : int = None):
        if self.all_inertia_w is None:
            raise Exception("You must use set_all_inertiaw before !")
        plt.plot(self.size_k, self.all_inertia_w)
        plt.title("Elbow")
        plt.ylabel("w")
        plt.xlabel("k")
        if idx_marker is not None:
            ki = {k:v for k,v in zip(self.size_k, self.all_inertia_w)}
            plt.plot(idx_marker, ki[idx_marker], marker = 'X', label = "best k")
        
        plt.legend()
        plt.show()
    
    #image ségmentée
    def compute_segmented_image(self,model):
        if self.df_img is None:
            raise Exception("You must use set_DataFrame before !")
        labels = model.predict(self.df_img)
        segmented_img_flat = model.cluster_centers_[labels]
        segmented_img = segmented_img_flat.reshape(self.img.shape).astype(np.uint8)
        return segmented_img
    
    #on affiche pour certains cas
    def display_segmented_images(self,ax_grid = (2,2) ,number=4):
        fig, axs = plt.subplots(*ax_grid, figsize=(8, 10))
        axs = axs.flatten()

        for i, k in enumerate(self.size_k[:number]):
            segmented_img = self.compute_segmented_image(self.models[i])
            axs[i].imshow(segmented_img)
            axs[i].set_title(f"K = {k}")
            axs[i].axis("off")

        plt.tight_layout() 
        plt.show()

    #benchmark pour comparer
    def benchmark(self):
        results = {}
        for i, k in enumerate(self.size_k):
           
            segmented_img = self.compute_segmented_image(self.models[i])
            temp_path = Path(f'temp_img_{k}.png')
            cv2.imwrite(str(temp_path), cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR))
            
            file_size = temp_path.stat().st_size / (1024)
            
            temp_path.unlink()
            if(self.all_inertia_w) is None:
                raise Exception("You have to use set_all_inertiaw before !")
            inertia_w = self.all_inertia_w[i]
            if(self.all_silhouette is not None):
                silhouette = self.all_silhouette[i]
                results[k] = {'inertia_w': inertia_w, 'file_size': file_size, 'silhouette': silhouette}
            else:
                results[k] = {'inertia_w': inertia_w, 'file_size': file_size}           
        self.bench = results

    #on affiche le benchmark
    def display_benchmark(self, w: bool = True):
        if self.bench is None:
            raise Exception("Use benchmark before !")

        k_values = list(self.bench.keys())
        file_sizes = [entry['file_size'] for entry in self.bench.values()]

        if w:
            metric_values = [entry['inertia_w'] for entry in self.bench.values()]
            metric_label = "Inertia W"
        else:
            if 'silhouette' not in self.bench[k_values[0]]:
                raise Exception("Silhouette scores are not available. Ensure you've computed them with set_all_silhouette.")
            metric_values = [entry['silhouette'] for entry in self.bench.values()]
            metric_label = "Silhouette Score"

        point_sizes = [size * 70 for size in file_sizes]
        
        img_file = Path(self.path)
        img_size = img_file.stat().st_size / (1024)
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(k_values, metric_values, s=point_sizes, c=file_sizes, cmap='viridis', alpha=0.6)
        
        cbar = plt.colorbar(scatter, label='File Size (KB)')
        cbar.ax.axhline(img_size, color='black', linestyle='-', linewidth=3) 
        
        plt.xlabel('K')
        plt.ylabel(metric_label)
        plt.title(f'{metric_label} vs K (with File Size(KB))')
        plt.show()


    


            


        


