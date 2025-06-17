# 📦 Dataset Preparation Guide for SatDiff

**SatDiff: A Stable Diffusion Framework for Inpainting Very High-Resolution Satellite Imagery**  
**Author**: Teerapong Panboonyuen  
**License**: MIT  
**Citation**: [IEEE Xplore – SatDiff Paper](https://ieeexplore.ieee.org/document/10929005)

---

## 🧭 Overview

This document outlines the steps required to prepare your dataset for use with the SatDiff framework. SatDiff expects paired satellite imagery and corresponding binary masks to learn inpainting of occluded or missing regions in very high-resolution satellite imagery.

## 📁 Directory Structure

Your dataset should follow this folder structure:

```

data/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
├── test/
│   ├── images/
│   └── masks/

````

- `images/`: RGB satellite images (`.png`, `.jpg`, `.tif`)
- `masks/`: Grayscale binary masks indicating missing regions (`0=valid`, `255=missing`)

> ✅ **Tip**: Ensure that each mask filename matches the corresponding image filename (e.g., `image001.png` ↔ `image001.png`).

---

## 🔧 Step-by-Step Instructions

### 1. **Collect High-Resolution Satellite Imagery**

Use publicly available sources such as:

- [DeepGlobe Dataset](http://deepglobe.org/)
- [SpaceNet](https://spacenet.ai/)
- [Massachusetts Roads Dataset](https://www.cs.toronto.edu/~vmnih/data/)
- Commercial VHR satellite data (if licensed)

Ensure the spatial resolution is high enough (e.g., ≤ 1m per pixel).

---

### 2. **Generate or Acquire Masks**

Masks should indicate where the inpainting is needed (e.g., cloud-covered areas, missing tiles, or artificial gaps).

You may:

- Use cloud detection algorithms (e.g., Fmask) to create cloud masks.
- Manually draw missing regions using photo editors.
- Simulate occlusions (e.g., random blocks, structural gaps).

### 3. **Resize (Optional)**

If needed, resize large images to match the `image_size` defined in `config.yaml` (e.g., 1024x1024).

```python
from PIL import Image
image = Image.open("your_image.jpg")
image = image.resize((1024, 1024))
image.save("resized_image.jpg")
````

---

### 4. **Preprocess and Organize**

Run the preprocessing script to format the dataset into the expected structure:

```bash
python dataset_preprocessing.py
```

You can modify paths inside the script as needed. This will copy, rename, and standardize your dataset layout into `data/train`, `data/val`, and `data/test`.

---

## 📌 Important Notes

* All masks should be **binary (0 or 255)**. You can convert grayscale masks using OpenCV or Pillow.
* Keep training and validation images in the same resolution and format.
* We recommend maintaining a balanced dataset with diverse landscapes and occlusion types.

---

## 🔖 Citation

If you use or build upon SatDiff or this dataset preparation workflow, please cite our work:

```bibtex
@article{panboonyuen2025satdiff,
  title={SatDiff: A Stable Diffusion Framework for Inpainting Very High-Resolution Satellite Imagery},
  author={Panboonyuen, Teerapong and et al.},
  journal={IEEE Access},
  year={2025},
  publisher={IEEE}
}
```

---

## 🙏 Acknowledgment

This documentation was prepared by **Teerapong Panboonyuen** as part of the SatDiff research project at AGL (Advancing Geoscience Laboratory), Chulalongkorn University.

For questions, issues, or contributions, please [open an issue on GitHub](https://github.com/kaopanboonyuen/SatDiff).