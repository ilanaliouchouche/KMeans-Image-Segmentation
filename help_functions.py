import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def load_image(path : str, display : bool =True):
    img = cv2.imread("ping.jpg", cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if display:
        plt.imshow(img)
        plt.axis("off")
        plt.show()
    return img



def crop_img(img : np.ndarray, height : slice, width : slice, display : bool =True):
    croped_img = img[height,width,:]
    if display:
        plt.imshow(croped_img)
        plt.axis(False)
        plt.show()
    return croped_img

def display_rgb(img : np.ndarray):
    fig, axs = plt.subplots(1, 3, figsize=(10,5))
    channel_names = ["Reds", "Greens", "Blues"]

    for idx, color in enumerate(channel_names):  
        axs[idx].imshow(img[:,:,idx], cmap=color)
        axs[idx].axis('off')

    plt.tight_layout()
    plt.show()

def flatten(img : np.ndarray, dataframe=True):
    img_flat = img.reshape((-1, 3))
    if dataframe:
        return pd.DataFrame(img_flat, columns = ['R', 'G', 'B'])
    return img_flat

def compute_kmeans(k : int,df_img : pd.DataFrame | np.ndarray, seed=42):
    kmeans = KMeans(n_clusters=k, random_state=seed)
    kmeans.fit(df_img)
    return kmeans

def compute_silhouette(model, data : pd.DataFrame | np.ndarray):
    return silhouette_score(data, model.labels_)

def compute_inertia_w(model):
    return model.inertia_

def compute_segmented_image(model,df_img : pd.DataFrame | np.ndarray, img : np.ndarray, display=False):
    labels = model.predict(df_img)
    segmented_img_flat = model.cluster_centers_[labels]
    segmented_img = segmented_img_flat.reshape(img.shape).astype(np.uint8)
    if (display):
        plt.imshow(segmented_img)
        plt.show()
    return segmented_img

def display_segmented_images(model_list,df_img : pd.DataFrame | np.ndarray, img : np.ndarray, k_values = np.arange(2,10), number=4):
    fig, axs = plt.subplots(2, 2, figsize=(8, 10))
    axs = axs.flatten()

    for i, k in enumerate(k_values[:number]):
        segmented_img = compute_segmented_image(model_list[i], df_img, img)
        axs[i].imshow(segmented_img)
        axs[i].set_title(f"K = {k}")
        axs[i].axis("off")

    plt.tight_layout() 
    plt.show()

def training(df_img : np.ndarray | pd.DataFrame, k_values=np.arange(2,10)):
    kmeans_list = Parallel(n_jobs=-1, verbose=10)(delayed(compute_kmeans)(k, df_img) for k in k_values)
    return kmeans_list

def get_all_inertiaw(kmeans_list):
    inertiaw_list = Parallel(n_jobs=-1, verbose=10)(delayed(compute_inertia_w)(model) for model in kmeans_list)
    return inertiaw_list

def display_elbow_plot(inertiaw_list,k_values=np.arange(2,10)):
    plt.plot(k_values, inertiaw_list)
    plt.title("Elbow")
    plt.ylabel("Inertie w")
    plt.xlabel("k")
    ki = {k:v for k,v in zip(k_values, inertiaw_list)}
    plt.plot(5, ki[5], marker = 'X')

    plt.show()