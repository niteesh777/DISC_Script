# Disc Analysis Tool: Facial Symmetry Pipeline

This repository provides a complete Python pipeline for processing paired facial images to:

* Detect and crop faces
* Run Digital Image Correlation (DIC) via PyDIC
* Segment facial regions using Meta’s SAM (Segment Anything Model)
* Generate heatmaps and vector fields
* Calculate symmetry scores for facial features

---

## 📁 Project Structure

```
.
├── Disc_Code.py               # Main script: runs the entire facial analysis pipeline
├── pydic.py                   # PyDIC module: required for DIC processing
├── requirements.txt           # Python dependencies
├── sam_vit_h_4b8939.pth       # Pre-trained SAM model (must be downloaded manually)
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/niteesh777/DISC_Script.git
cd DISC_Script
```

### 2. Set up your environment

```bash
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Prepare your input images

* Place **two frontal face images** in any location on your machine.
* Ensure both images are of the **same person**, captured at slightly different expressions or conditions.

Update the following paths inside `Disc_Code.py`:

```python
image1_path = 'path/to/first_image.jpg'
image2_path = 'path/to/second_image.jpg'
base_dir = 'path/to/output_directory'
```

### 4. Download SAM model checkpoint manually

The `sam_vit_h_4b8939.pth` file is required but **too large for GitHub**. Download it from the official [Segment Anything repo](https://github.com/facebookresearch/segment-anything).

Place the `.pth` file in the root of this project:

```bash
mv /path/to/sam_vit_h_4b8939.pth ./sam_vit_h_4b8939.pth
```

---

## ⚡ Running the Pipeline

Simply execute:

```bash
python Disc_Code.py
```

This performs the following:

* Copies and renames images to a working directory
* Auto-crops around face region using `face_recognition`
* Runs PyDIC on cropped images
* Uses SAM to segment and mask cheeks/forehead
* Applies mask to DIC results
* Generates:

  * Displacement heatmaps
  * Displacement vectors (optional)
  * Symmetry score overlays and CSV outputs

---

## 🔍 Key Parameters & Directories

* `base_dir`: Directory where all intermediate and final outputs are saved
* `image1_path`, `image2_path`: Input images (must be frontal faces)
* `cropped_dir`: Auto-generated dir storing cropped images
* `csv_dir`: Stores processed DIC output CSVs
* `overlay_dir`: Stores visual outputs like heatmaps, symmetry overlays
* `pydic.py`: Required dependency that handles the DIC logic (should not be altered unless customizing core behavior)

---

## 🌐 Output Artifacts

Upon completion, the following will be generated:

* Cropped face images
* Modified CSVs with displacement values masked to relevant facial regions
* Heatmaps and vector fields
* Symmetry score visualizations (left vs right cheek/forehead)
* `symmetry_scores.csv` summarizing all displacement values per region

---

## 📄 License

MIT License
