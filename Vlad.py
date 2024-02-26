import torch
import numpy as np
from SuperGluePretrainedNetwork.models.matching import Matching
from SuperGluePretrainedNetwork.models.utils import frame2tensor, make_matching_plot_fast
from tqdm import tqdm
import cv2
import os
import warnings
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree

# Note: made some adjustments to implementation already done here: https://github.com/IrvingF7/VPR_VLAD/tree/master

# Define configuration parameters for the SuperPoint and SuperGlue models
class SuperOpt:
    def __init__(self):
        self.nms_radius = 4
        self.keypoint_threshold = 0.005
        self.max_keypoints = 80

        self.superglue = 'outdoor'  ## change this based on the conditions of your environment i.e indoor or outdoor
        self.sinkhorn_iterations = 20
        self.match_threshold = 0.3

# Main class that orchestrates feature extraction, VLAD computation, and similarity search
class Main:
    def __init__(self):
        # Initialize SuperOpt configuration, device, and SuperGlue model
        self.super_opt = SuperOpt()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.super_config = {
            'superpoint': {
                'nms_radius': self.super_opt.nms_radius,
                'keypoint_threshold': self.super_opt.keypoint_threshold,
                'max_keypoints': self.super_opt.max_keypoints
            },
            'superglue': {
                'weights': self.super_opt.superglue,
                'sinkhorn_iterations': self.super_opt.sinkhorn_iterations,
                'match_threshold': self.super_opt.match_threshold,
            }
        }
        self.matching = Matching(self.super_config).eval().to(self.device)
        self.super_keys = ['keypoints', 'scores', 'descriptors']
        self.img_data_list = []
        self.img_tensor_list = []

    # Method to find SuperPoint feature descriptors in an image
    def find_feature_points_superpoint(self, img):
        img_tensor = frame2tensor(img, self.device)
        img_data = self.matching.superpoint({'image': img_tensor})
        self.img_data_list.append(img_data)
        self.img_tensor_list.append(img_tensor)
        return img_data['descriptors']

    # Method to extract and aggregate feature descriptors from a folder of images
    def extract_aggregate_feature(self, folder_path):
        all_descriptors = []
        
        # Iterate through images in the folder
        for img_name in tqdm(os.listdir(folder_path)):
            img_path = os.path.join(folder_path, img_name)
            
            try:
                # Read and process the image
                image = cv2.imread(img_path)
                grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                des = self.find_feature_points_superpoint(grayscale_image)
                all_descriptors.extend(des)
            except Exception as e:
                # Handle errors when processing images
                print(f"Error processing image {img_name}: {e}")

        # Detach tensors and convert to NumPy arrays
        all_descriptors = np.asarray([desc.detach().numpy() for desc in all_descriptors])
        return all_descriptors
    
    # Method to compute the VLAD (Vector of Locally Aggregated Descriptors) feature
    def get_VLAD(self, descriptors, codebook):
        predicted_labels = codebook.predict(descriptors)
        centroids = codebook.cluster_centers_
        num_clusters = codebook.n_clusters

        m, d = descriptors.shape
        VLAD_feature = np.zeros([num_clusters, d])

        # Compute the differences for all clusters (visual words)
        for i in range(num_clusters):
            # Check if there is at least one descriptor in that cluster
            if np.sum(predicted_labels == i) > 0:
                # Add the differences
                VLAD_feature[i] = np.sum(descriptors[predicted_labels == i, :] - centroids[i], axis=0)

        VLAD_feature = VLAD_feature.flatten()

        # Power normalization, also called square-rooting normalization
        VLAD_feature = np.sign(VLAD_feature) * np.sqrt(np.abs(VLAD_feature))

        # L2 normalization
        VLAD_feature = VLAD_feature / np.linalg.norm(VLAD_feature)

        return VLAD_feature

# Instantiate the Main class
vlad = Main()

# Set the path to the database folder and extract aggregated feature descriptors
# database_path = r"C:\Users\ifeda\ROB-GY-Computer-Vision\HW2\task_5\database"
all_des = vlad.extract_aggregate_feature(database_path).reshape(30 * 80, 256)

# Perform k-means clustering on the entire bag of descriptors
kmeans_codebook = KMeans(n_clusters=16, init='k-means++', n_init=1, verbose=1).fit(all_des)

# Initialize lists to store VLAD representations and image names from the database
database_VLAD = []
database_name = []

# Iterate through images in the database folder
for img_name in tqdm(os.listdir(database_path)):
    img_path = os.path.join(database_path, img_name)
    image = cv2.imread(img_path)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    des = vlad.find_feature_points_superpoint(grayscale_image)
    all_descriptors = np.asarray([desc.detach().numpy() for desc in des]).reshape(80, 256)    
    VLAD = vlad.get_VLAD(all_descriptors, kmeans_codebook)
    database_VLAD.append(VLAD)
    database_name.append(img_name)
    
# Convert the lists to NumPy arrays
database_VLAD = np.asarray(database_VLAD)

# Build a BallTree for efficient nearest neighbor search
tree = BallTree(database_VLAD, leaf_size=60)

# Set the number of closest images to retrieve
num_of_imgs = 1

# Set the path to the query folder
# query_path = r"C:\Users\ifeda\ROB-GY-Computer-Vision\HW2\task_5\query"
img_value_list = []

# Iterate through images in the query folder
for img_name in tqdm(os.listdir(query_path)):
    img_path = os.path.join(query_path, img_name)
    image = cv2.imread(img_path)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find SuperPoint feature descriptors in the query image
    q_des = vlad.find_feature_points_superpoint(grayscale_image)
    all_descriptors = np.asarray([desc.detach().numpy() for desc in q_des]).reshape(80, 256)
    
    # Compute the VLAD representation for the query image
    query_VLAD = vlad.get_VLAD(all_descriptors, kmeans_codebook).reshape(1, -1)
    
    # Retrieve the index of the closest image(s) in the database
    dist, index = tree.query(query_VLAD, num_of_imgs)
    
    # Index is an array of arrays of size 1
    # Get the name of the closest image and append it to the list
    value_name = database_name[index[0][0]]
    img_value_list.append(value_name)
    
# Print the list of names of the closest images in the database to the query images
print(img_value_list)
