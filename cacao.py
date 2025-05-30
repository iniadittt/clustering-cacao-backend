import os
import cv2
import uuid
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class CacaoColorSegmentation:
    def __init__(self, dataset_path, n_clusters=5):
        """
        Initialize the color segmentation class
        
        Args:
            dataset_path: Path to the dataset folder containing subfolders
            n_clusters: Number of clusters for K-means
        """
        self.dataset_path = Path(dataset_path)
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        
    def load_images_from_folder(self, folder_path):
        """Load all images from a specific folder"""
        images = []
        image_paths = []
        
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        
        for img_path in Path(folder_path).glob('*'):
            if img_path.suffix.lower() in supported_formats:
                img = cv2.imread(str(img_path))
                if img is not None:
                    images.append(img)
                    image_paths.append(str(img_path))
                    
        return images, image_paths
    
    def rgb_to_lab(self, rgb_image):
        """Convert RGB image to CIELAB color space"""
        # Convert BGR to RGB (OpenCV loads as BGR)
        rgb_img = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        # Convert RGB to LAB
        lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
        return lab_img
    
    def extract_lab_features(self, images):
        """Extract LAB color features from images"""
        all_lab_pixels = []
        
        for img in images:
            # Convert to LAB
            lab_img = self.rgb_to_lab(img)
            
            # Reshape to get all pixels
            lab_pixels = lab_img.reshape(-1, 3)
            
            # Sample pixels to reduce computation (optional)
            # You can adjust sampling rate based on your needs
            sample_size = min(1000, len(lab_pixels))
            indices = np.random.choice(len(lab_pixels), sample_size, replace=False)
            sampled_pixels = lab_pixels[indices]
            
            all_lab_pixels.append(sampled_pixels)
        
        # Combine all pixels
        combined_pixels = np.vstack(all_lab_pixels)
        return combined_pixels
    
    def perform_kmeans_clustering(self, lab_features):
        """Perform K-means clustering on LAB features"""
        # Normalize the features
        lab_features_scaled = self.scaler.fit_transform(lab_features)
        
        # Apply K-means
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(lab_features_scaled)
        
        return kmeans, cluster_labels
    
    def get_dominant_colors(self, kmeans_model, lab_features):
        """Get dominant colors from cluster centers"""
        # Get cluster centers in scaled space
        centers_scaled = kmeans_model.cluster_centers_
        
        # Inverse transform to get original LAB values
        centers_lab = self.scaler.inverse_transform(centers_scaled)
        
        # Convert LAB centers to RGB for visualization
        centers_rgb = []
        for center in centers_lab:
            # Create a single pixel image with the LAB color
            lab_pixel = np.uint8([[center]])
            rgb_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2RGB)
            centers_rgb.append(rgb_pixel[0][0])
        
        return centers_lab, centers_rgb
    
    def visualize_dominant_colors(self, colors_rgb, title="Dominant Colors"):
        """Visualize dominant colors"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 2))
        
        # Create color palette
        palette = np.array(colors_rgb).reshape(1, -1, 3) / 255.0
        ax.imshow(palette, aspect='auto')
        ax.set_xlim(0, len(colors_rgb))
        ax.set_xticks(range(len(colors_rgb)))
        ax.set_xticklabels([f'Cluster {i}' for i in range(len(colors_rgb))])
        ax.set_yticks([])
        ax.set_title(title)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_single_category(self, category_name, category_path):
        """Analyze color segmentation for a single category"""
        
        # Load images
        images, image_paths = self.load_images_from_folder(category_path)
        
        if len(images) == 0:
            return None
        
        # Extract LAB features
        lab_features = self.extract_lab_features(images)
        
        # Perform clustering
        kmeans_model, cluster_labels = self.perform_kmeans_clustering(lab_features)
        
        # Get dominant colors
        centers_lab, centers_rgb = self.get_dominant_colors(kmeans_model, lab_features)
        
        # Visualize
        self.visualize_dominant_colors(centers_rgb, f"Dominant Colors - {category_name}")
        
        return {
            'category': category_name,
            'kmeans_model': kmeans_model,
            'centers_lab': centers_lab,
            'centers_rgb': centers_rgb,
            'cluster_labels': cluster_labels,
            'lab_features': lab_features
        }
    
    def analyze_all_categories(self):
        """Analyze all categories in the dataset"""
        categories = {
            'Kakao Matang': self.dataset_path / 'Kakao Matang',
            'Kakao Belum Matang': self.dataset_path / 'Kakao Belum Matang',
            'Bukan Kakao': self.dataset_path / 'Bukan Kakao'
        }
        
        results = {}
        
        for category_name, category_path in categories.items():
            if category_path.exists():
                result = self.analyze_single_category(category_name, category_path)
                if result:
                    results[category_name] = result
        
        return results
    
    def compare_categories(self, results):
        """Compare color characteristics between categories"""
        if len(results) < 2:
            return
        
        # Create comparison visualization
        fig, axes = plt.subplots(len(results), 1, figsize=(12, 4 * len(results)))
        if len(results) == 1:
            axes = [axes]
        
        for idx, (category, result) in enumerate(results.items()):
            palette = np.array(result['centers_rgb']).reshape(1, -1, 3) / 255.0
            axes[idx].imshow(palette, aspect='auto')
            axes[idx].set_xlim(0, len(result['centers_rgb']))
            axes[idx].set_xticks(range(len(result['centers_rgb'])))
            axes[idx].set_xticklabels([f'C{i}' for i in range(len(result['centers_rgb']))])
            axes[idx].set_yticks([])
            axes[idx].set_title(f'{category} - Dominant Colors')
        
        plt.tight_layout()
        plt.show()
        
        # Create LAB comparison dataframe
        comparison_data = []
        for category, result in results.items():
            for i, lab_color in enumerate(result['centers_lab']):
                comparison_data.append({
                    'Category': category,
                    'Cluster': i,
                    'L': lab_color[0],
                    'A': lab_color[1],
                    'B': lab_color[2]
                })
        
        df = pd.DataFrame(comparison_data)
        
        # Visualize LAB distribution
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, channel in enumerate(['L', 'A', 'B']):
            sns.boxplot(data=df, x='Category', y=channel, ax=axes[idx])
            axes[idx].set_title(f'{channel} Channel Distribution')
            axes[idx].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return df
    
    def segment_image_colors(self, image_path, kmeans_model):
        """Segment colors in a single image using trained model"""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert to LAB
        lab_img = self.rgb_to_lab(img)
        original_shape = lab_img.shape[:2]
        
        # Reshape for prediction
        lab_pixels = lab_img.reshape(-1, 3)
        lab_pixels_scaled = self.scaler.transform(lab_pixels)
        
        # Predict clusters
        labels = kmeans_model.predict(lab_pixels_scaled)
        
        # Reshape back to image shape
        segmented = labels.reshape(original_shape)
        
        return segmented, img
    
    def visualize_cluster_segmentation(self, image_path, kmeans_model, save_path=None):
        """Create cluster segmentation visualization like the example"""
        # Get segmentation and original image
        result = self.segment_image_colors(image_path, kmeans_model)
        if result is None:
            return None
            
        segmented, original_img = result
        
        # Create color map for clusters
        # Use distinct colors for each cluster
        cluster_colors = [
            [255, 0, 0],      # Red
            [0, 255, 0],      # Green  
            [0, 0, 255],      # Blue
            [255, 255, 0],    # Yellow
            [255, 0, 255],    # Magenta
            [0, 255, 255],    # Cyan
            [255, 128, 0],    # Orange
            [128, 0, 255],    # Purple
            [0, 128, 255],    # Light Blue
            [255, 255, 255]   # White
        ]
        
        # Create individual cluster images
        cluster_images = []
        height, width = segmented.shape
        
        for cluster_id in range(self.n_clusters):
            # Create blank image
            cluster_img = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Get cluster color
            color = cluster_colors[cluster_id % len(cluster_colors)]
            
            # Fill pixels belonging to this cluster
            mask = segmented == cluster_id
            cluster_img[mask] = color
            
            cluster_images.append(cluster_img)
        
        # Create visualization
        fig, axes = plt.subplots(self.n_clusters + 1, 1, figsize=(8, 2 * (self.n_clusters + 1)))
        
        # Show original image
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(original_rgb)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Show each cluster
        for i, cluster_img in enumerate(cluster_images):
            cluster_rgb = cv2.cvtColor(cluster_img, cv2.COLOR_BGR2RGB)
            axes[i + 1].imshow(cluster_rgb)
            axes[i + 1].set_title(f'Cluster {i}')
            axes[i + 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
        return cluster_images
    
    def visualize_cluster_segmentation_horizontal(self, image_path, kmeans_model, save_path=None):
        """Create horizontal cluster segmentation visualization and save to storages/ with random filename"""
        
        # Get segmentation and original image
        result = self.segment_image_colors(image_path, kmeans_model)
        if result is None:
            return None

        segmented, original_img = result

        # Create color map for clusters
        cluster_colors = [
            [255, 0, 0],      # Red
            [0, 255, 0],      # Green  
            [0, 0, 255],      # Blue
            [255, 255, 0],    # Yellow
            [255, 0, 255],    # Magenta
            [0, 255, 255],    # Cyan
            [255, 128, 0],    # Orange
            [128, 0, 255],    # Purple
            [0, 128, 255],    # Light Blue
            [255, 255, 255]   # White
        ]

        # Create individual cluster images
        cluster_images = []
        height, width = segmented.shape

        for cluster_id in range(self.n_clusters):
            cluster_img = np.zeros((height, width, 3), dtype=np.uint8)
            color = cluster_colors[cluster_id % len(cluster_colors)]
            mask = segmented == cluster_id
            cluster_img[mask] = color
            cluster_images.append(cluster_img)

        # Create HORIZONTAL visualization
        fig, axes = plt.subplots(1, self.n_clusters + 1, figsize=(3 * (self.n_clusters + 1), 4))

        # Show original image
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(original_rgb)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Show each cluster
        for i, cluster_img in enumerate(cluster_images):
            cluster_rgb = cv2.cvtColor(cluster_img, cv2.COLOR_BGR2RGB)
            axes[i + 1].imshow(cluster_rgb)
            axes[i + 1].set_title(f'Cluster {i}')
            axes[i + 1].axis('off')

        plt.tight_layout()

        # Save to storages/ with random filename
        if save_path is None:
            os.makedirs('storages', exist_ok=True)
            random_name = f"{uuid.uuid4().hex}_clusters.png"
            save_path = os.path.join('storages', random_name)

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return cluster_images, save_path
    
    def create_combined_cluster_visualization(self, image_paths, kmeans_model, max_images=5):
        """Create a combined visualization showing multiple images and their clusters"""
        n_images = min(len(image_paths), max_images)
        
        fig, axes = plt.subplots(n_images, self.n_clusters + 1, 
                                figsize=(2 * (self.n_clusters + 1), 2 * n_images))
        
        if n_images == 1:
            axes = axes.reshape(1, -1)
        
        cluster_colors = [
            [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255],
            [0, 255, 255], [255, 128, 0], [128, 0, 255], [0, 128, 255], [255, 255, 255]
        ]
        
        for img_idx in range(n_images):
            image_path = image_paths[img_idx]
            
            # Get segmentation
            result = self.segment_image_colors(image_path, kmeans_model)
            if result is None:
                continue
                
            segmented, original_img = result
            
            # Show original image
            original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            axes[img_idx, 0].imshow(original_rgb)
            axes[img_idx, 0].set_title(f'Original {img_idx + 1}' if img_idx == 0 else '')
            axes[img_idx, 0].axis('off')
            
            # Show clusters
            height, width = segmented.shape
            for cluster_id in range(self.n_clusters):
                cluster_img = np.zeros((height, width, 3), dtype=np.uint8)
                color = cluster_colors[cluster_id % len(cluster_colors)]
                mask = segmented == cluster_id
                cluster_img[mask] = color
                
                cluster_rgb = cv2.cvtColor(cluster_img, cv2.COLOR_BGR2RGB)
                axes[img_idx, cluster_id + 1].imshow(cluster_rgb)
                
                if img_idx == 0:  # Only add title to first row
                    axes[img_idx, cluster_id + 1].set_title(f'Cluster {cluster_id}')
                axes[img_idx, cluster_id + 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def create_combined_cluster_visualization_horizontal(self, image_paths, kmeans_model, max_images=6):
        """Create a horizontal combined visualization showing multiple images and their clusters"""
        n_images = min(len(image_paths), max_images)
        
        # Create horizontal layout: rows = images, cols = original + clusters
        fig, axes = plt.subplots(n_images, self.n_clusters + 1, 
                                figsize=(2.5 * (self.n_clusters + 1), 2.5 * n_images))
        
        if n_images == 1:
            axes = axes.reshape(1, -1)
        
        cluster_colors = [
            [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255],
            [0, 255, 255], [255, 128, 0], [128, 0, 255], [0, 128, 255], [255, 255, 255]
        ]
        
        for img_idx in range(n_images):
            image_path = image_paths[img_idx]
            
            # Get segmentation
            result = self.segment_image_colors(image_path, kmeans_model)
            if result is None:
                continue
                
            segmented, original_img = result
            
            # Show original image
            original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            axes[img_idx, 0].imshow(original_rgb)
            axes[img_idx, 0].set_title(f'Original {img_idx + 1}' if img_idx == 0 else '')
            axes[img_idx, 0].axis('off')
            
            # Show clusters
            height, width = segmented.shape
            for cluster_id in range(self.n_clusters):
                cluster_img = np.zeros((height, width, 3), dtype=np.uint8)
                color = cluster_colors[cluster_id % len(cluster_colors)]
                mask = segmented == cluster_id
                cluster_img[mask] = color
                
                cluster_rgb = cv2.cvtColor(cluster_img, cv2.COLOR_BGR2RGB)
                axes[img_idx, cluster_id + 1].imshow(cluster_rgb)
                
                if img_idx == 0:  # Only add title to first row
                    axes[img_idx, cluster_id + 1].set_title(f'Cluster {cluster_id}')
                axes[img_idx, cluster_id + 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_cluster_statistics(self, results):
        """Analyze statistics for each cluster across categories"""
        cluster_stats = {}
        
        for category, result in results.items():
            cluster_stats[category] = {}
            kmeans_model = result['kmeans_model']
            lab_features = result['lab_features']
            
            # Get cluster assignments
            lab_features_scaled = self.scaler.transform(lab_features)
            cluster_labels = kmeans_model.predict(lab_features_scaled)
            
            # Calculate statistics for each cluster
            for cluster_id in range(self.n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_pixels = lab_features[cluster_mask]
                
                if len(cluster_pixels) > 0:
                    cluster_stats[category][f'Cluster_{cluster_id}'] = {
                        'pixel_count': len(cluster_pixels),
                        'percentage': len(cluster_pixels) / len(lab_features) * 100,
                        'mean_L': np.mean(cluster_pixels[:, 0]),
                        'mean_A': np.mean(cluster_pixels[:, 1]),
                        'mean_B': np.mean(cluster_pixels[:, 2]),
                        'std_L': np.std(cluster_pixels[:, 0]),
                        'std_A': np.std(cluster_pixels[:, 1]),
                        'std_B': np.std(cluster_pixels[:, 2])
                    }
        
        return cluster_stats
    
    def save_models(self, results, save_directory="saved_models"):
        """Save all trained K-means models in one combined file using joblib"""
        import os
        
        # Create save directory
        os.makedirs(save_directory, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save only combined model file
        combined_filename = f"all_kmeans_models_{timestamp}.joblib"
        combined_filepath = os.path.join(save_directory, combined_filename)
        
        combined_data = {
            'all_results': results,
            'scaler': self.scaler,
            'n_clusters': self.n_clusters,
            'timestamp': timestamp,
            'categories': list(results.keys())
        }
        
        joblib.dump(combined_data, combined_filepath)
        
        return combined_filepath
    
    def load_model(self, model_path):
        """Load the saved combined K-means models"""
        try:
            model_data = joblib.load(model_path)
            
            # Update scaler
            if 'scaler' in model_data:
                self.scaler = model_data['scaler']
            
            # Update n_clusters
            if 'n_clusters' in model_data:
                self.n_clusters = model_data['n_clusters']
                
            return model_data['all_results']
        except Exception as e:
            return None
    
    def predict_with_saved_model(self, image_path, loaded_results, category_name):
        """Use a saved model to predict/segment a new image"""
        if category_name not in loaded_results:
            return None
        
        kmeans_model = loaded_results[category_name]['kmeans_model']
        return self.segment_image_colors(image_path, kmeans_model)
    
    def predict_image_category(self, image_path, loaded_results):
        """Predict which category an image belongs to based on color similarity"""
        
        # Extract LAB features from the input image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        lab_img = self.rgb_to_lab(img)
        lab_pixels = lab_img.reshape(-1, 3)
        
        # Sample pixels for consistency with training
        sample_size = min(1000, len(lab_pixels))
        indices = np.random.choice(len(lab_pixels), sample_size, replace=False)
        sampled_pixels = lab_pixels[indices]
        
        # Scale the features
        lab_features_scaled = self.scaler.transform(sampled_pixels)
        
        category_scores = {}
        
        # Test against each category model
        for category_name, result in loaded_results.items():
            kmeans_model = result['kmeans_model']
            
            # Get cluster assignments
            cluster_labels = kmeans_model.predict(lab_features_scaled)
            
            # Calculate similarity score based on cluster distribution
            cluster_counts = np.bincount(cluster_labels, minlength=self.n_clusters)
            cluster_percentages = cluster_counts / len(cluster_labels) * 100
            
            # Calculate distance to cluster centers
            distances = kmeans_model.transform(lab_features_scaled)
            avg_distance = np.mean(np.min(distances, axis=1))
            
            # Score based on inverse of average distance (lower distance = higher score)
            similarity_score = 1 / (1 + avg_distance)
            
            category_scores[category_name] = {
                'similarity_score': similarity_score,
                'avg_distance': avg_distance,
                'cluster_distribution': cluster_percentages
            }
        
        # Find best matching category
        best_category = max(category_scores.keys(), 
        key=lambda x: category_scores[x]['similarity_score'])
        
        return {
            'predicted_category': best_category,
            'confidence_score': category_scores[best_category]['similarity_score'],
            'all_scores': category_scores,
            'image_path': image_path
        }
    
    def predict_multiple_images(self, image_paths, loaded_results):
        """Predict categories for multiple images"""
        
        predictions = []
        
        for i, image_path in enumerate(image_paths):
            prediction = self.predict_image_category(image_path, loaded_results)
            if prediction:
                predictions.append(prediction)
        
        return predictions
    
    def visualize_prediction_results(self, predictions):
        """Visualize prediction results with confidence scores"""
        
        if not predictions:
            return
        
        # Create summary dataframe
        pred_data = []
        for pred in predictions:
            pred_data.append({
                'Image': os.path.basename(pred['image_path']),
                'Predicted_Category': pred['predicted_category'],
                'Confidence': pred['confidence_score']
            })
        
        df = pd.DataFrame(pred_data)
        
        # Plot confidence distribution
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Confidence by category
        sns.boxplot(data=df, x='Predicted_Category', y='Confidence', ax=axes[0])
        axes[0].set_title('Prediction Confidence by Category')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Category distribution
        category_counts = df['Predicted_Category'].value_counts()
        axes[1].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
        axes[1].set_title('Predicted Category Distribution')
        
        plt.tight_layout()
        plt.show()
        
        return df
    
    def predict_and_visualize(self, image_path, loaded_results, show_clusters=True):
        """Predict category and show cluster segmentation"""
        
        # Get prediction
        prediction = self.predict_image_category(image_path, loaded_results)
        
        if not prediction:
            return None
        
        cluster_images = None
        save_path = None

        # Show cluster segmentation if requested
        if show_clusters:
            best_category = prediction['predicted_category']
            kmeans_model = loaded_results[best_category]['kmeans_model']
            cluster_images, save_path = self.visualize_cluster_segmentation_horizontal(image_path, kmeans_model)
        
        return prediction, cluster_images, save_path