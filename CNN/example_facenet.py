import numpy as np
import os
# Assuming your AiFramework is in the parent directory or accessible via PYTHONPATH
from src import FaceNet, TripletLoss, Adam # Assuming Adam optimizer is available

import os
import random
from PIL import Image # Assuming Pillow is installed for image processing
import matplotlib.pyplot as plt

# --- Configuration ---
IMG_WIDTH = 160
IMG_HEIGHT = 160

def preprocess_image(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT)):
    """Loads an image, resizes, converts to RGB, and normalizes."""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img, dtype=np.float32)
        # Normalize to [0, 1] or [-1, 1] depending on model training
        img_array = img_array / 255.0 
        # Transpose to C, H, W format if your model expects that
        # (e.g., img_array.transpose(2, 0, 1))
        # Current Conv2D expects (channels, height, width)
        img_array = img_array.transpose(2, 0, 1)
        return img_array
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def load_casia_webface_subset(dataset_path, num_train_identities=100, num_test_identities=50):
    """Loads a subset of CASIA-WebFace identities and their image paths."""
    print(f"Scanning dataset at: {dataset_path}")
    identities = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    random.shuffle(identities) # Shuffle to get random identities

    if len(identities) < num_train_identities + num_test_identities:
        print(f"Warning: Not enough identities in dataset. Found {len(identities)}, requested {num_train_identities+num_test_identities}.")
        # Adjust numbers if not enough identities
        if len(identities) < num_train_identities:
            num_train_identities = len(identities)
            num_test_identities = 0
        else:
            num_test_identities = len(identities) - num_train_identities
        print(f"Using {num_train_identities} for training and {num_test_identities} for testing.")

    train_identity_names = identities[:num_train_identities]
    test_identity_names = identities[num_train_identities : num_train_identities + num_test_identities]

    train_data = {}
    test_data = {}

    print(f"Processing {len(train_identity_names)} training identities...")
    for identity_name in train_identity_names:
        identity_path = os.path.join(dataset_path, identity_name)
        image_files = [os.path.join(identity_path, f) for f in os.listdir(identity_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            train_data[identity_name] = image_files
    
    print(f"Processing {len(test_identity_names)} testing identities...")
    for identity_name in test_identity_names:
        identity_path = os.path.join(dataset_path, identity_name)
        image_files = [os.path.join(identity_path, f) for f in os.listdir(identity_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            test_data[identity_name] = image_files
            
    print(f"Loaded {len(train_data)} training identities and {len(test_data)} testing identities.")
    return train_data, test_data

def load_and_preprocess_dataset(data_paths_dict, target_size=(IMG_WIDTH, IMG_HEIGHT)):
    """Loads all images from paths and preprocesses them."""
    processed_data = {}
    identities_count = len(data_paths_dict)
    current_identity_idx = 0
    for identity_name, img_paths in data_paths_dict.items():
        current_identity_idx += 1
        print(f"Preprocessing identity {identity_name} ({current_identity_idx}/{identities_count})... ({len(img_paths)} images)")
        processed_images = []
        for img_path in img_paths:
            img_array = preprocess_image(img_path, target_size)
            if img_array is not None:
                processed_images.append(img_array)
        if processed_images:
            processed_data[identity_name] = processed_images
    print("Finished preprocessing all images.")
    return processed_data

def get_triplet_batch(processed_data_dict, batch_size):
    """Generates a batch of triplets (anchor, positive, negative) from preprocessed data."""
    anchors = []
    positives = []
    negatives = []
    
    identities = list(processed_data_dict.keys())
    if len(identities) < 2:
        raise ValueError("Need at least 2 identities to form triplets.")

    for _ in range(batch_size):
        # Pick anchor identity
        anchor_id_idx = np.random.randint(0, len(identities))
        anchor_id = identities[anchor_id_idx]
        
        # Pick positive image from anchor identity
        # Ensure at least 2 images for anchor identity to pick different anchor and positive
        if len(processed_data_dict[anchor_id]) < 2:
            # Skip if not enough images for this identity to form a pair
            # In a real scenario, handle this more gracefully or ensure data quality
            print(f"Warning: Identity {anchor_id} has less than 2 images. Skipping triplet.")
            continue
        anchor_idx, positive_idx = np.random.choice(len(processed_data_dict[anchor_id]), 2, replace=False)
        anchor_img = processed_data_dict[anchor_id][anchor_idx]
        positive_img = processed_data_dict[anchor_id][positive_idx]
        
        # Pick negative identity (different from anchor_id)
        negative_id_idx = np.random.randint(0, len(identities))
        while negative_id_idx == anchor_id_idx:
            negative_id_idx = np.random.randint(0, len(identities))
        negative_id = identities[negative_id_idx]

        # Pick negative image from negative identity
        if not processed_data_dict[negative_id]: # Ensure negative identity has images
            print(f"Warning: Negative identity {negative_id} has no preprocessed images. Skipping triplet.")
            continue
        negative_img_idx = np.random.randint(0, len(processed_data_dict[negative_id]))
        negative_img = processed_data_dict[negative_id][negative_img_idx]
        # Ensure all images were processed successfully
        if anchor_img is not None and positive_img is not None and negative_img is not None:
            anchors.append(anchor_img)
            positives.append(positive_img)
            negatives.append(negative_img)
        # Error printing for missing preprocessed images is handled during loading or if an identity has no images.

    if not anchors: # If no valid triplets were formed
        return None, None, None
        
    return np.array(anchors), np.array(positives), np.array(negatives)


if __name__ == "__main__":
    # --- Configuration ---
    DATASET_PATH = "/Users/yjl/Desktop/AiFramework/CNN/dataset/CASIA-WebFace"
    NUM_TRAIN_IDENTITIES = 100
    NUM_TEST_IDENTITIES = 10
    INPUT_SHAPE_MODEL = (3, IMG_HEIGHT, IMG_WIDTH) # C, H, W
    EMBEDDING_SIZE = 128
    LEARNING_RATE = 0.0001 # Adam's default is often 0.001, trying a smaller one
    BATCH_SIZE = 16 # Adjust based on memory
    EPOCHS = 50     # Increase for real training
    MARGIN = 0.2

    # --- 1. Load Data ---
    print("Loading CASIA-WebFace data subset paths...")
    train_data_paths_dict, test_data_paths_dict = load_casia_webface_subset(DATASET_PATH, NUM_TRAIN_IDENTITIES, NUM_TEST_IDENTITIES)

    if not train_data_paths_dict or len(train_data_paths_dict) < 2:
        print("Not enough training data paths to proceed. Exiting.")
        exit()

    print("\nPreprocessing training data...")
    train_data_processed = load_and_preprocess_dataset(train_data_paths_dict, target_size=(IMG_HEIGHT, IMG_WIDTH))
    print("\nPreprocessing testing data...")
    test_data_processed = load_and_preprocess_dataset(test_data_paths_dict, target_size=(IMG_HEIGHT, IMG_WIDTH))

    if not train_data_processed or len(train_data_processed) < 2:
        print("Not enough training data to proceed. Exiting.")
        exit()

    # --- 2. Initialize Model, Loss, Optimizer ---
    print("Initializing FaceNet model...")
    facenet_model = FaceNet(input_shape=INPUT_SHAPE_MODEL, embedding_size=EMBEDDING_SIZE)
    triplet_loss_fn = TripletLoss(margin=MARGIN)
    # Assuming Adam optimizer is implemented and available in src.optimizers
    # If not, you might need to implement it or use an existing one like SGD.
    optimizer = Adam(learning_rate=LEARNING_RATE) 
    
    # Compile the model (linking loss and optimizer)
    # The current Model class in the framework might not have a compile method that directly uses the optimizer
    # in the way Keras does. The train_step in FaceNet directly uses learning_rate.
    # We'll assign the loss function here for use in train_step.
    # facenet_model.loss_fn = triplet_loss_fn # This direct assignment will be handled by compile
    facenet_model.compile(optimizer=optimizer, loss=triplet_loss_fn)
    # The optimizer might need to be passed to train_step or handled internally by layers if they update themselves.
    # For now, the FaceNet train_step takes learning_rate. If Adam is used, its state needs to be managed.
    # This part highlights a potential area for framework improvement: integrating optimizers more centrally.

    print(f"Model: FaceNet, compiled with TripletLoss and Adam optimizer.")
    print(f"Loss: TripletLoss (margin={MARGIN})")
    print(f"Optimizer: Adam (lr={LEARNING_RATE})")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")

    # --- 3. Training Loop ---
    print("\nStarting training...")
    loss_history = [] # To store loss values for plotting
    for epoch in range(EPOCHS):
        print(f"--- Epoch {epoch+1}/{EPOCHS} ---")
        # In a real scenario, you'd iterate over your dataset in batches
        # For this dummy example, we'll just generate one batch per epoch
        
        anchor_b, positive_b, negative_b = get_triplet_batch(train_data_processed, BATCH_SIZE)
        
        if anchor_b is None or positive_b is None or negative_b is None or len(anchor_b) == 0:
            print(f"Skipping epoch {epoch+1} due to insufficient data for a batch.")
            continue

        # The FaceNet train_step currently updates weights internally using learning_rate.
        # If using an optimizer like Adam, the update logic would be more complex and involve optimizer.step().
        # This is a simplification based on the current framework structure.
        loss = facenet_model.train_step(anchor_b, positive_b, negative_b, LEARNING_RATE)
        loss_history.append(loss)
        print(f"Epoch {epoch+1} - Loss: {loss:.4f}")

    print("\nTraining finished.")

    # Save model weights
    model_weights_path = "facenet_model_weights.npz"
    facenet_model.save_weights(model_weights_path)

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS + 1), loss_history, marker='o', linestyle='-')
    plt.title('FaceNet Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Triplet Loss')
    plt.grid(True)
    plt.savefig("result.png")

    # --- 4. Validation/Embedding Generation (Conceptual) ---
    print("\nGenerating embeddings for some dummy images...")
    # Get a few sample images from the test set to generate embeddings
    print("\nGenerating embeddings for some test images...")
    test_identities_list = list(test_data_processed.keys()) # Use processed data keys
    if len(test_identities_list) >= 2 and test_data_processed.get(test_identities_list[0]) and test_data_processed.get(test_identities_list[1]):
        # Image 1 (from first test identity)
        # We need original path for printing, but use processed image for embeddings
        img1_original_path = train_data_paths_dict.get(test_identities_list[0],[None])[0] or test_data_paths_dict.get(test_identities_list[0],[None])[0] # Get original path for display
        img1_processed = test_data_processed[test_identities_list[0]][0]
        # No need to call preprocess_image again, it's already done
        if img1_processed is not None: # Should always be not None if preloading worked
            img1_batch = np.expand_dims(img1_processed, axis=0)
            embedding1 = facenet_model.get_embeddings(img1_batch)
            print(f"Embedding for test image 1 ({img1_original_path}, shape {embedding1.shape}):\n{embedding1[0,:10]}... (first 10 dims)")

            # Image 2 (from second test identity - should be different person)
            img2_original_path = train_data_paths_dict.get(test_identities_list[1],[None])[0] or test_data_paths_dict.get(test_identities_list[1],[None])[0]
            img2_processed = test_data_processed[test_identities_list[1]][0]
            if img2_processed is not None:
                img2_batch = np.expand_dims(img2_processed, axis=0)
                embedding2 = facenet_model.get_embeddings(img2_batch)
                print(f"Embedding for test image 2 ({img2_original_path}, shape {embedding2.shape}):\n{embedding2[0,:10]}... (first 10 dims)")
                distance_diff_person = np.sum(np.square(embedding1 - embedding2))
                print(f"Euclidean distance (different persons): {distance_diff_person:.4f}")

            # Image 3 (another image from first test identity - should be same person as img1)
            if len(test_data_processed[test_identities_list[0]]) > 1:
                img3_original_path = None
                train_paths = train_data_paths_dict.get(test_identities_list[0])
                test_paths = test_data_paths_dict.get(test_identities_list[0])
                if train_paths and len(train_paths) > 1:
                    img3_original_path = train_paths[1]
                elif test_paths and len(test_paths) > 1:
                    img3_original_path = test_paths[1]
                
                # If no second image path found, we can't proceed with this comparison
                if img3_original_path is None:
                    print(f"Not enough original image paths for identity {test_identities_list[0]} to get a second image for comparison.")
                    # continue # or break, depending on desired behavior

                img3_processed = test_data_processed[test_identities_list[0]][1]
                if img3_processed is not None:
                    img3_batch = np.expand_dims(img3_processed, axis=0)
                    embedding3 = facenet_model.get_embeddings(img3_batch)
                    print(f"Embedding for test image 3 ({img3_original_path} - same person as img1, shape {embedding3.shape}):\n{embedding3[0,:10]}... (first 10 dims)")
                    distance_same_person = np.sum(np.square(embedding1 - embedding3))
                    print(f"Euclidean distance (same person): {distance_same_person:.4f}")
            else:
                print(f"Not enough images for identity {test_identities_list[0]} to test same-person distance.")
        # else: # This case should ideally not happen if preloading is successful
            # print(f"Could not use preprocessed test image for: {img1_original_path}")
    else:
        print("Not enough test identities/images with at least two images each to perform embedding comparison.")

    # --- 5. 可视化验证集embedding分布 ---
    print("\n正在提取验证集所有样本的embedding并进行可视化...")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    all_embeddings = []
    all_labels = []
    # Use preprocessed test data for t-SNE
    for identity, processed_imgs_list in test_data_processed.items():
        for img_array in processed_imgs_list:
            # img_array is already preprocessed
            if img_array is not None: # Should be fine as it's preprocessed
                emb = facenet_model.get_embeddings(np.expand_dims(img_array, axis=0))[0]
                all_embeddings.append(emb)
                all_labels.append(identity)
    if len(all_embeddings) > 2:
        all_embeddings = np.array(all_embeddings)
        tsne = TSNE(n_components=2, random_state=42)
        emb_2d = tsne.fit_transform(all_embeddings)
        plt.figure(figsize=(8,6))
        # 为每个身份分配唯一颜色
        import matplotlib.cm as cm
        unique_labels = list(set(all_labels))
        colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        label2color = {label: color for label, color in zip(unique_labels, colors)}
        for label in unique_labels:
            idxs = [i for i, l in enumerate(all_labels) if l == label]
            plt.scatter(emb_2d[idxs,0], emb_2d[idxs,1], color=label2color[label], label=str(label), alpha=0.6, s=20)
        plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('t-SNE Visualization of FaceNet Embeddings (Validation Set)')
        plt.xlabel('t-SNE dim 1')
        plt.ylabel('t-SNE dim 2')
        plt.tight_layout()
        plt.savefig("tsne.png")
    else:
        print("验证集样本数不足，无法可视化embedding分布。")

    print("\nFaceNet example script with CASIA-WebFace subset finished.")
    print("Note: This script uses real image paths but relies on Pillow for image processing.")
    print("Ensure Pillow is installed ('pip install Pillow').")
    print("For robust evaluation, a more comprehensive validation set and metrics (e.g., ROC curve, accuracy@FAR) are needed.")

    print("\nFaceNet example script finished.")
    print("Note: This is a simplified example with dummy data and basic training.")
    print("A full implementation would require proper data loading, extensive training, and robust validation.")