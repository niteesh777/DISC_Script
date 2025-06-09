import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from PIL import Image
from scipy.ndimage import zoom
from tqdm import tqdm
import importlib.util
import face_recognition  
import torch
from segment_anything import sam_model_registry, SamPredictor

def main():
    base_dir = './output'
    image1_path = './input/image1.jpg'
    image2_path = './input/image2.jpg'
    pydic_module_path = './pydic.py'
    checkpoint_path = './sam_vit_h_4b8939.pth'

    output_dir = os.path.join(base_dir, 'Nick_Output')
    cropped_dir = os.path.join(output_dir, 'Cropped')
    pydic_result_dir = os.path.join(cropped_dir, 'pydic/result')
    csv_dir = os.path.join(cropped_dir, 'csv')
    overlay_dir = os.path.join(cropped_dir, 'Output/OVERLAY')

    for path in [output_dir, cropped_dir, pydic_result_dir, csv_dir, overlay_dir]:
        os.makedirs(path, exist_ok=True)

    settings = {
        'OverLay': True,
        'Contour': True,
        'Vector': False,
        'FixedScaleValue': [0, 90],
        'SparseVector': False,
        'VectorSubsamplingFactor': 100
    }

    prepare_images(image1_path, image2_path, output_dir)
    auto_crop_images(output_dir, cropped_dir)
    run_pydic(cropped_dir, pydic_module_path)
    auto_create_mask_and_process_csv(cropped_dir, checkpoint_path)
    generate_heatmaps_and_vectors(cropped_dir, csv_dir, overlay_dir, settings)
    calculate_symmetry_scores(cropped_dir, csv_dir)

    print("\u2705 All processing complete.")


def prepare_images(image1_path, image2_path, output_dir):
    """Copy the two images to the output directory and rename them sequentially."""
    if not os.path.exists(image1_path) or not os.path.exists(image2_path):
        raise FileNotFoundError("One or both image paths are invalid.")


    images = [image1_path, image2_path]
    for idx, img_path in enumerate(images):
        img = cv2.imread(img_path)
        if img is None:
            raise Exception(f"Error loading image: {img_path}")
        filename = os.path.join(output_dir, f'{1000000 + idx:07d}.png')
        cv2.imwrite(filename, img)
    print("Images prepared for processing.")


def auto_crop_images(input_dir, output_dir):
    """Automatically detect faces and crop images around the face, including more of the head."""
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
    if len(image_files) < 2:
        raise Exception(f"Not enough images found in directory: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)
    print("Automatic cropping process initiated...")

    face_coords = []

    # First pass: Detect face locations in all images
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)
        if image is not None:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image)
            if face_locations:
                face_coords.append(face_locations[0])  # Assuming one face per image
            else:
                raise Exception(f"No face detected in image: {image_file}. Cannot proceed.")
        else:
            raise Exception(f"Error loading image: {image_path}")

    # Calculate common crop coordinates
    tops, rights, bottoms, lefts = zip(*face_coords)
    top = min(tops)
    right = max(rights)
    bottom = max(bottoms)
    left = min(lefts)

    # Apply padding and extra headroom
    padding = 20
    face_height = max(bottoms) - min(tops)
    head_extra = int(face_height * 0.5)

    # Adjust the coordinates
    top = max(0, top - padding - head_extra)
    bottom = bottom + padding
    left = max(0, left - padding)
    right = right + padding

    # Second pass: Crop all images using the common coordinates
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)
        cropped_image = image[top:bottom, left:right]
        cv2.imwrite(os.path.join(output_dir, f"cropped_{image_file}"), cropped_image)


def run_pydic(cropped_dir, pydic_module_path):
    """Run PyDIC analysis on the cropped images."""
    if not os.path.exists(pydic_module_path):
        raise FileNotFoundError(f"pydic module not found at: {pydic_module_path}")

    spec = importlib.util.spec_from_file_location("pydic", pydic_module_path)
    pydic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pydic)

    correl_wind_size = (60, 60)
    correl_grid_size = (1, 1)
    dic_file = os.path.join(cropped_dir, 'result.dic')

    pydic.init(os.path.join(cropped_dir, '*.png'), correl_wind_size, correl_grid_size, dic_file, area_of_intersest='all')
    pydic.FixedscaleValues = [15, 3, 0]
    pydic.read_dic_file(dic_file, FixedScale=False, interpolation='raw', save_image=False, scale_disp=2, scale_grid=1)

def auto_create_mask_and_process_csv(cropped_dir):
    """Automatically create a mask using SAM and apply it to the CSV files."""
    mask_file_path = os.path.join(cropped_dir, 'mask_x.npy')
    csv_dir_path = os.path.join(cropped_dir, 'pydic/result')
    modified_csv_dir_path = os.path.join(cropped_dir, 'csv')
    os.makedirs(modified_csv_dir_path, exist_ok=True)

    auto_create_mask(cropped_dir, mask_file_path)
    process_csv_files(csv_dir_path, cropped_dir, mask_file_path, modified_csv_dir_path)
    print("Automatic masking and CSV processing complete.")

def auto_create_mask(image_dir_path, mask_file_path):
    """Create a face mask using SAM with center and cheek points as prompts."""

    # Load the image
    image_files = sorted([f for f in os.listdir(image_dir_path) if f.endswith('.png')])
    if not image_files:
        raise Exception(f"No PNG images found in directory: {image_dir_path}")

    img_path = os.path.join(image_dir_path, image_files[0])
    img = cv2.imread(img_path)
    if img is None:
        raise Exception("Error loading image for mask creation.")

    # Convert image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get image dimensions
    height, width = img_rgb.shape[:2]

    # Define points
    center_point = [width // 2, height // 2]
    cheek_offset = width // 5  # Adjust this value as needed
    left_cheek_point = [center_point[0] - cheek_offset, center_point[1]]
    right_cheek_point = [center_point[0] + cheek_offset, center_point[1]]

    # Prepare input points and labels
    input_points = np.array([center_point, left_cheek_point, right_cheek_point])
    input_labels = np.array([1, 1, 1])  # Labels: 1 for foreground

    # Load the SAM model
    model_type = "vit_h"  # Choose from 'vit_h', 'vit_l', 'vit_b'
    checkpoint_path = "./sam_vit_h_4b8939.pth"  # Update with your actual path

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # Run the model
    predictor.set_image(img_rgb)
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False,
    )

    # Process the mask
    mask = masks[0]
    mask_bool_inverted = ~mask.astype(bool)
    np.save(mask_file_path, mask_bool_inverted)

    # Visualize the masked image (optional)
    mask_3channel = np.dstack([mask]*3).astype(np.uint8)*255
    masked_image = cv2.bitwise_and(img, mask_3channel)
    # cv2.imshow('Masked Image', masked_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def process_csv_files(csv_dir_path, image_dir_path, mask_file_path, modified_csv_dir_path):
    """Apply the mask to each CSV file."""
    mask = np.load(mask_file_path)
    mask = ~mask  

    csv_files = sorted([f for f in os.listdir(csv_dir_path) if f.endswith('.csv')])
    if not csv_files:
        raise Exception(f"No CSV files found in directory: {csv_dir_path}")

    for csv_file in tqdm(csv_files, desc="Processing CSV files"):
        csv_file_path = os.path.join(csv_dir_path, csv_file)
        df = pd.read_csv(csv_file_path)

        x_unique = np.sort(df['pos_x'].unique())
        y_unique = np.sort(df['pos_y'].unique())

        X, Y = np.meshgrid(x_unique, y_unique)

        resized_mask = zoom(mask.astype(float), (
            len(y_unique) / mask.shape[0],
            len(x_unique) / mask.shape[1]
        ), order=0).astype(bool)

        Z = df.pivot(index='pos_y', columns='pos_x', values='disp_xy_indi')

        Z_masked = Z.where(resized_mask)

        df_masked = Z_masked.stack().reset_index()
        df_masked.columns = ['pos_y', 'pos_x', 'disp_xy_indi']
        df_final = df.drop(columns=['disp_xy_indi']).merge(df_masked, on=['pos_x', 'pos_y'], how='inner')

        df_final.to_csv(os.path.join(modified_csv_dir_path, f'modified_{csv_file}'), index=False)
 
def generate_heatmaps_and_vectors(cropped_dir, csv_dir, overlay_dir, settings):
    """Generate heatmaps and vector fields from the CSV files and overlay them on images."""

    # Unpack settings
    OverLay = settings['OverLay']
    Contour = settings['Contour']
    Vector = settings['Vector']
    FixedScaleValue = settings['FixedScaleValue']
    SparseVector = settings.get('SparseVector', True)
    VectorSubsamplingFactor = settings.get('VectorSubsamplingFactor', 10)

    png_dir = cropped_dir
    csv_dir = os.path.join(cropped_dir, 'csv')
    output_dir_path = os.path.join(png_dir, 'Output/OVERLAY')
    os.makedirs(output_dir_path, exist_ok=True)

    # Define and create subdirectories for different types of outputs
    if Contour:
        heatmap_dir = os.path.join(output_dir_path, 'Heatmap')
        os.makedirs(heatmap_dir, exist_ok=True)
    if Vector:
        vector_dir = os.path.join(output_dir_path, 'Vector_folder')
        os.makedirs(vector_dir, exist_ok=True)

    # List all CSV and PNG files
    csv_files = sorted([f for f in os.listdir(csv_dir) if f.endswith('.csv')])
    png_files = sorted([f for f in os.listdir(png_dir) if f.endswith('.png')])

    # Since each CSV file corresponds to the displacement between two images,
    # we need to pair the CSV file with the second image in the sequence.
    if len(csv_files) != len(png_files) - 1:
        raise Exception("The number of CSV files should be one less than the number of PNG files.")

    for idx, csv_file in enumerate(tqdm(csv_files, total=len(csv_files), desc="Processing files")):
        csv_file_path = os.path.join(csv_dir, csv_file)
        # Use the second image in the pair for visualization
        image_file = png_files[idx + 1]
        image_path = os.path.join(png_dir, image_file)
        df = pd.read_csv(csv_file_path)
        image = Image.open(image_path)
        empty_image = np.zeros((*image.size[::-1], 4), dtype=np.uint8)

        if Contour:
            scale = [i / 100 for i in range(int(100 * FixedScaleValue[0]), int(100 * FixedScaleValue[1]) + 1,
                                             int(100 * (FixedScaleValue[1] - FixedScaleValue[0]) / 25))]
            norm = mcolors.Normalize(vmin=FixedScaleValue[0], vmax=FixedScaleValue[1])

            dpi = image.info['dpi'] if 'dpi' in image.info else (300, 300)
            dpi = (int(dpi[0]), int(dpi[1]))

            # Reshape the data into a 2D grid
            x_unique = np.sort(df['pos_x'].unique())
            y_unique = np.sort(df['pos_y'].unique())
            Z = df.pivot_table(index='pos_y', columns='pos_x', values='disp_xy_indi', fill_value=np.nan).reindex(
                index=y_unique, columns=x_unique, fill_value=np.nan).values

            fig, ax = plt.subplots(figsize=(image.width / dpi[0], image.height / dpi[1]), dpi=dpi[0])

            if OverLay:
                ax.imshow(image, extent=[x_unique.min(), x_unique.max(), y_unique.max(), y_unique.min()])
            else:
                ax.imshow(empty_image, extent=[x_unique.min(), x_unique.max(), y_unique.max(), y_unique.min()])

            # Mask the array where Z values are NaN
            Z_masked = np.ma.masked_where(np.isnan(Z), Z)

            contour = ax.contourf(x_unique, y_unique, Z_masked, levels=scale, cmap=plt.cm.jet, alpha=0.5, norm=norm,
                                  extent=[x_unique.min(), x_unique.max(), y_unique.max(), y_unique.min()])
            ax.axis('off')

            plt.savefig(os.path.join(heatmap_dir, os.path.splitext(image_file)[0] + '_heatmap.png'), dpi=dpi[0],
                        bbox_inches='tight', pad_inches=0)
            plt.close()

        if Vector:
            plt.figure(figsize=(image.width / 100, image.height / 100))
            if OverLay:
                plt.imshow(image, cmap='gray', extent=[0, image.width, image.height, 0])
            else:
                plt.imshow(empty_image, extent=[0, image.width, image.height, 0])

            # Apply subsampling if SparseVector is True
            if SparseVector and VectorSubsamplingFactor > 1:
                print('here',VectorSubsamplingFactor)
                df_subsampled = df.iloc[::VectorSubsamplingFactor, :]
            else:
                df_subsampled = df  # No subsampling, use all data

            plt.quiver(
                df_subsampled['pos_x'],
                df_subsampled['pos_y'],
                df_subsampled['disp_x'],
                df_subsampled['disp_y'],
                angles='xy',
                scale_units='xy',
                scale=1,
                color='red',
                width=0.0005
            )
            plt.axis('equal')
            plt.axis('off')
            vector_output_path = os.path.join(vector_dir, os.path.splitext(image_file)[0] + '_vector.png')
            plt.savefig(vector_output_path, transparent=False, format='png', dpi=1000)
            plt.close()
            

def calculate_symmetry_scores(cropped_dir, csv_dir):
    """Calculate facial symmetry scores based on displacement data within specific face regions,
    and display the regions and scores on the face images."""

    csv_dir = os.path.join(cropped_dir, 'csv')
    mask_file_path = os.path.join(cropped_dir, 'mask_x.npy')
    png_dir = cropped_dir
    output_dir_path = os.path.join(cropped_dir, 'Output/OVERLAY/Symmetry_Scores')
    os.makedirs(output_dir_path, exist_ok=True)

    csv_files = sorted([f for f in os.listdir(csv_dir) if f.endswith('.csv')])
    png_files = sorted([f for f in os.listdir(png_dir) if f.endswith('.png')])

    if not csv_files:
        raise Exception(f"No CSV files found in directory: {csv_dir}")

    symmetry_scores = []

    for idx, csv_file in enumerate(csv_files):
        csv_file_path = os.path.join(csv_dir, csv_file)
        image_file = png_files[idx + 1]  # Use the second image in the pair
        image_path = os.path.join(png_dir, image_file)
        image = Image.open(image_path)
        df = pd.read_csv(csv_file_path)

        # Compute the displacement magnitude
        df['disp_magnitude'] = np.sqrt(df['disp_x']**2 + df['disp_y']**2)

        # Get unique x and y positions
        x_unique = np.sort(df['pos_x'].unique())
        y_unique = np.sort(df['pos_y'].unique())

        # Pivot the displacement data into a 2D grid
        Z_disp = df.pivot_table(index='pos_y', columns='pos_x', values='disp_magnitude', fill_value=np.nan).values

        # Now, detect facial landmarks on the image
        img_cv2 = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        face_landmarks_list = face_recognition.face_landmarks(img_rgb)
        if not face_landmarks_list:
            raise Exception("No face landmarks detected.")

        face_landmarks = face_landmarks_list[0]

        # Calculate face height for dynamic forehead region sizing
        chin_bottom = max(face_landmarks['chin'], key=lambda point: point[1])[1]
        nose_bridge_top = min(face_landmarks['nose_bridge'], key=lambda point: point[1])[1]
        face_height = chin_bottom - nose_bridge_top

        # Now, define the regions

        # Left cheek polygon points
        left_cheek_points = face_landmarks['chin'][0:9]  # Left side of chin
        left_cheek_points += [face_landmarks['left_eye'][0]]

        # Right cheek polygon points
        right_cheek_points = face_landmarks['chin'][8:17]  # Right side of chin
        right_cheek_points += [face_landmarks['right_eye'][-1]]

        # Adjustments for the forehead regions

        # Calculate forehead height as one-third of face height
        forehead_height = int(face_height * (1/3))

        # Define delta_y to start the forehead region slightly above the eyebrows
        delta_y = 10  # Adjust as needed

        # Left forehead polygon points
        left_eyebrow_leftmost = face_landmarks['left_eyebrow'][0]
        left_eyebrow_rightmost = face_landmarks['left_eyebrow'][-1]
        left_eyebrow_min_y = min([pt[1] for pt in face_landmarks['left_eyebrow']])

        left_forehead_bottom_y = left_eyebrow_min_y - delta_y
        left_forehead_top_y = max(0, left_forehead_bottom_y - forehead_height)  # Ensure we don't go above the image

        left_forehead_points = [
            (left_eyebrow_leftmost[0], left_forehead_bottom_y),
            (left_eyebrow_rightmost[0], left_forehead_bottom_y),
            (left_eyebrow_rightmost[0], left_forehead_top_y),
            (left_eyebrow_leftmost[0], left_forehead_top_y)
        ]

        # Right forehead polygon points
        right_eyebrow_leftmost = face_landmarks['right_eyebrow'][0]
        right_eyebrow_rightmost = face_landmarks['right_eyebrow'][-1]
        right_eyebrow_min_y = min([pt[1] for pt in face_landmarks['right_eyebrow']])

        right_forehead_bottom_y = right_eyebrow_min_y - delta_y
        right_forehead_top_y = max(0, right_forehead_bottom_y - forehead_height)

        right_forehead_points = [
            (right_eyebrow_leftmost[0], right_forehead_bottom_y),
            (right_eyebrow_rightmost[0], right_forehead_bottom_y),
            (right_eyebrow_rightmost[0], right_forehead_top_y),
            (right_eyebrow_leftmost[0], right_forehead_top_y)
        ]

        # For the purposes of masking, we need to create masks for these regions

        # Create empty masks
        mask_shape = img_rgb.shape[:2]
        left_cheek_mask = np.zeros(mask_shape, dtype=np.uint8)
        right_cheek_mask = np.zeros(mask_shape, dtype=np.uint8)
        left_forehead_mask = np.zeros(mask_shape, dtype=np.uint8)
        right_forehead_mask = np.zeros(mask_shape, dtype=np.uint8)

        # Fill the masks with the regions
        cv2.fillPoly(left_cheek_mask, [np.array(left_cheek_points, dtype=np.int32)], 1)
        cv2.fillPoly(right_cheek_mask, [np.array(right_cheek_points, dtype=np.int32)], 1)
        cv2.fillPoly(left_forehead_mask, [np.array(left_forehead_points, dtype=np.int32)], 1)
        cv2.fillPoly(right_forehead_mask, [np.array(right_forehead_points, dtype=np.int32)], 1)

        # Now, resize these masks to match the displacement grid
        grid_width = Z_disp.shape[1]
        grid_height = Z_disp.shape[0]

        left_cheek_mask_resized = cv2.resize(left_cheek_mask, (grid_width, grid_height), interpolation=cv2.INTER_NEAREST)
        right_cheek_mask_resized = cv2.resize(right_cheek_mask, (grid_width, grid_height), interpolation=cv2.INTER_NEAREST)
        left_forehead_mask_resized = cv2.resize(left_forehead_mask, (grid_width, grid_height), interpolation=cv2.INTER_NEAREST)
        right_forehead_mask_resized = cv2.resize(right_forehead_mask, (grid_width, grid_height), interpolation=cv2.INTER_NEAREST)

        # Apply these masks to the displacement data
        Z_disp_masked_left_cheek = np.where(left_cheek_mask_resized, Z_disp, np.nan)
        Z_disp_masked_right_cheek = np.where(right_cheek_mask_resized, Z_disp, np.nan)
        Z_disp_masked_left_forehead = np.where(left_forehead_mask_resized, Z_disp, np.nan)
        Z_disp_masked_right_forehead = np.where(right_forehead_mask_resized, Z_disp, np.nan)

        # Calculate mean displacements in these regions
        left_cheek_mean_disp = np.nanmean(Z_disp_masked_left_cheek)
        right_cheek_mean_disp = np.nanmean(Z_disp_masked_right_cheek)
        left_forehead_mean_disp = np.nanmean(Z_disp_masked_left_forehead)
        right_forehead_mean_disp = np.nanmean(Z_disp_masked_right_forehead)

        # Calculate symmetry scores
        cheek_symmetry_score = right_cheek_mean_disp - left_cheek_mean_disp
        forehead_symmetry_score = right_forehead_mean_disp - left_forehead_mean_disp

        symmetry_scores.append({
            'csv_file': csv_file,
            'left_cheek_mean_disp': left_cheek_mean_disp,
            'right_cheek_mean_disp': right_cheek_mean_disp,
            'cheek_symmetry_score': cheek_symmetry_score,
            'left_forehead_mean_disp': left_forehead_mean_disp,
            'right_forehead_mean_disp': right_forehead_mean_disp,
            'forehead_symmetry_score': forehead_symmetry_score
        })

        print(f"Symmetry scores for {csv_file}:")
        print(f"  Left Cheek Mean Displacement: {left_cheek_mean_disp}")
        print(f"  Right Cheek Mean Displacement: {right_cheek_mean_disp}")
        print(f"  Cheek Symmetry Score (Right - Left): {cheek_symmetry_score}")
        print(f"  Left Forehead Mean Displacement: {left_forehead_mean_disp}")
        print(f"  Right Forehead Mean Displacement: {right_forehead_mean_disp}")
        print(f"  Forehead Symmetry Score (Right - Left): {forehead_symmetry_score}\n")

        # Visualization
        plt.figure(figsize=(image.width / 100, image.height / 100))
        plt.imshow(image)
        plt.axis('off')

        # Overlay the regions
        # Create an overlay image
        overlay = np.zeros((*mask_shape, 4), dtype=np.uint8)
        # Set colors for each region
        overlay[left_cheek_mask.astype(bool), :] = [255, 0, 0, 100]        # Red with alpha
        overlay[right_cheek_mask.astype(bool), :] = [0, 255, 0, 100]       # Green with alpha
        overlay[left_forehead_mask.astype(bool), :] = [0, 0, 255, 100]     # Blue with alpha
        overlay[right_forehead_mask.astype(bool), :] = [255, 255, 0, 100]  # Yellow with alpha

        plt.imshow(overlay)

        # Annotate the mean displacements
        # Left Cheek
        left_cheek_center = np.mean(np.array(left_cheek_points), axis=0)
        plt.text(left_cheek_center[0], left_cheek_center[1], f'{left_cheek_mean_disp:.2f}', color='white', fontsize=12,
                 ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5))
        # Right Cheek
        right_cheek_center = np.mean(np.array(right_cheek_points), axis=0)
        plt.text(right_cheek_center[0], right_cheek_center[1], f'{right_cheek_mean_disp:.2f}', color='white', fontsize=12,
                 ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5))
        # Left Forehead
        left_forehead_center_x = (left_forehead_points[0][0] + left_forehead_points[1][0]) / 2
        left_forehead_center_y = (left_forehead_points[2][1] + left_forehead_points[1][1]) / 2
        plt.text(left_forehead_center_x, left_forehead_center_y, f'{left_forehead_mean_disp:.2f}',
                 color='white', fontsize=12, ha='center', va='center',
                 bbox=dict(facecolor='black', alpha=0.5))
        # Right Forehead
        right_forehead_center_x = (right_forehead_points[0][0] + right_forehead_points[1][0]) / 2
        right_forehead_center_y = (right_forehead_points[2][1] + right_forehead_points[1][1]) / 2
        plt.text(right_forehead_center_x, right_forehead_center_y, f'{right_forehead_mean_disp:.2f}',
                 color='white', fontsize=12, ha='center', va='center',
                 bbox=dict(facecolor='black', alpha=0.5))

        # Save the visualization
        output_image_path = os.path.join(output_dir_path, f'{os.path.splitext(image_file)[0]}_symmetry.png')
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()

    # Optionally, save the symmetry scores to a CSV file
    scores_df = pd.DataFrame(symmetry_scores)
    output_dir_path = os.path.join(cropped_dir, 'Output/OVERLAY')
    scores_df.to_csv(os.path.join(output_dir_path, 'symmetry_scores.csv'), index=False)


if __name__ == "__main__":
    main()