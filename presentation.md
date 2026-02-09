---
marp: true
theme: uncover
class: invert
paginate: true
backgroundColor: #001f3f
style: |
  section {
    font-family: 'Inter', sans-serif;
    color: #ffffff;
  }
  h1 {
    color: #00d1ff;
  }
  h2 {
    color: #00d1ff;
  }
  strong {
    color: #ffcc00;
  }
  blockquote {
    background: #003366;
    border-left: 10px solid #00d1ff;
    color: #ffffff;
  }
---

# 🚦 Traffic Sign Classification
### Deep Feature Manifolds & Hybrid SVM Optimization

**Đỗ Duy Toàn** & Team (USTH)
2026 Final Project

---

## 🎯 Project Objectives

- **Goal:** Robust recognition for **Autonomous Vehicles**.
- **Core:** Bridge **Deep Learning** and **SVM**.
- **Innovation:** Investigation of **3D Manifold** Classification.

---

## 📦 Dataset: GTSRB (10 Classes)

![width:700px](../reports/figures/dataset_samples.png)

- 10,000+ images.
- High noise, scale, and light variations.

---

## 🏗️ Pipeline Architecture

![width:900px](../reports/figures/meta_reference.png)

1. **Extraction:** ResNet50 & InceptionV3
2. **Reduction:** PCA & UMAP (3D)
3. **Classification:** RBF-SVM Hybrid

---

## 🧊 3D Manifolds (UMAP)

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
  <div>
    <img src="../reports/figures/umap_3d_resnet50_static.png" width="100%">
    <p style="font-size: 0.5em;">ResNet50 clusters</p>
  </div>
  <div>
    <img src="../reports/figures/umap_3d_inceptionv3_static.png" width="100%">
    <p style="font-size: 0.5em;">InceptionV3 clusters</p>
  </div>
</div>

---

## ⚡ Manifold Strategy (RFC 3D)

- **Problem:** Linear models fail in 3D (< 35%).
- **Insight:** **Random Forest** dominates this space.
- **Why?** Recursive split perfectly isolates non-linear clusters.

> **Result:** **82.28% Accuracy** using only **0.14%** of data.

---

## 📊 Performance Matrix

| Configuration | ResNet50 | InceptionV3 |
| :--- | :---: | :---: |
| SGD-SVM (Baseline) | 91.2% | 88.5% |
| **RBF-SVM (Hybrid)** | **95.9%** | **97.2%** |

![width:450px](../reports/performance/inceptionv3_rbf_confusion_matrix.png)

---

## 🏁 Conclusion

1. **Deep Manifolds** provide robust features.
2. **RBF Kernels** are superior for non-linear deep logic.
3. **3D UMAP** preserves high semantic density.

---

## 💻 Tech Stack & Repo

- **Languages:** Python, LaTeX, Markdown
- **AI:** PyTorch, Scikit-learn, UMAP
- **GitHub:** [toando0212/lab-ml2](https://github.com/toando0212/lab-ml2)

# 🚀 Thank You!
## Any Questions?
