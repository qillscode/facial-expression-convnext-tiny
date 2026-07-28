# Analisis Performa ConvNeXt-Tiny dalam Pengenalan Ekspresi Wajah pada Dataset FER+

Proyek ini berisi notebook penelitian untuk menganalisis performa **ConvNeXt-Tiny** dalam klasifikasi ekspresi wajah menggunakan dataset **FER+**.

## Ringkasan

Notebook `Performa_ConvNeXt_Tiny_Pada_FER+.ipynb` disusun mengikuti alur **CRISP-DM**:
1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment

## Dataset

> ⚠️ **Penting:** Dataset tidak disertakan dalam repository karena keterbatasan ukuran GitHub.
>
> Silakan unduh dataset FER+ melalui:
> - https://github.com/Microsoft/FERPlus
> - https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

## Model

Model utama yang digunakan adalah **ConvNeXt-Tiny** untuk tugas klasifikasi ekspresi wajah.

Implementasi model dibuat dengan pendekatan deep learning berbasis PyTorch dan library `timm`.

## Struktur Proyek

```
.
├── Final_Notebook.ipynb    # Notebook utama penelitian
├── Dataset/                  # Folder untuk dataset (tidak disertakan)
├── .gitignore               # Pengecualian file lokal dan environment
└── README.md                # File dokumentasi
```

## Kebutuhan

Notebook ini dijalankan pada environment Python dengan dependensi utama:

- `torch`
- `torchvision`
- `timm`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `opencv-python`
- `scikit-learn`
- `split-folders`

## Cara Menjalankan

1. Aktifkan environment Python
2. Unduh dataset FER+ dari link yang disediakan di atas
3. Pastikan dataset sudah berada di folder `Dataset/`
4. Buka `Performa_ConvNeXt_Tiny_Pada_FER+.ipynb`
5. Jalankan cell dari atas ke bawah

## Sitasi

### Dataset

Barsoum, E., Zhang, C., Canton Ferrer, C., dan Zhang, Z. (2016). *Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution*. ACM International Conference on Multimedia.

### Model

Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., dan Xie, S. (2022). *A ConvNet for the 2020s*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Hamdani, S. A. (2024). *Pengenalan Ekspresi Wajah dengan ResNeXt*. Penulisan Ilmiah. Universitas Sumatera Utara.
