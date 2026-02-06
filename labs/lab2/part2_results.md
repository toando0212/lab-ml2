# Part 2: Subspace Clustering Results

## 1. Dataset Selection
- **Dataset**: ISOLET (Isolated Letter Speech Recognition)
- **Features**: 617 continuous features
- **Classes**: 26 (Letters A-Z)

## 2. PCA Visualization
We utilized PCA to project the 617-dimensional data into 2D and 3D spaces.

- **2D Projection**: Retains approx 17% of total variance.
- **3D Projection**: Retains approx 22% of total variance.
- **Observation**: Classes are heavily overlapped in low dimensions. Visual separation is poor.

## 3. Clustering on PCA-Reduced Data
We applied K-means (k=26) on the reduced data.

| Method | NMI | ARI | Time (s) |
|--------|-----|-----|----------|
| **Original (617D)** | **0.7308** | **0.5225** | 1.50 |
| **PCA (2D)** | 0.4312 | 0.1605 | 0.21 |
| **PCA (3D)** | 0.4802 | 0.2100 | 0.13 |

**Analysis**:
- Dimensionality reduction causes a **massive drop** in clustering quality (NMI 0.73 -> 0.48).
- This confirms that critical information for distinguishing letters is lost when compressing to just 2-3 dimensions.
- However, computation is much faster (0.13s vs 1.50s).

## 4. Clustering on Random Subspace
We selected a random 10% subset of features (61 features) and re-ran K-means.

| Method | NMI | ARI | Time (s) |
|--------|-----|-----|----------|
| **Random Subspace (61D)** | **0.6811** | **0.4795** | 0.18 |

**Comparision**:
- **Random Subspace (0.68)** significantly outperforms **PCA 3D (0.48)**.
- It is surprisingly close to the **Full Dataset (0.73)**.

**Discussion**:
- This suggests the ISOLET dataset has high **feature redundancy**. Many features carry similar information.
- A random subset is able to capture the overall geometry of the data better than a linear projection that tries to maximize variance (PCA).
- PCA may be focusing on "variance" that is actually noise or irrelevant to class separation, whereas original features (even random ones) preserve the discriminative signal.
