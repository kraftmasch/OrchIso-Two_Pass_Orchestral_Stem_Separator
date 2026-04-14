# OrchIso: Learning-Based Orchestral Isolation Tool
## by: tembok ratapan solo
Learning-Based Tool for Isolating Orchestral Section (String, Brass, etc) of a Popular Song wioth Two-Stage Audio Source Separation

for KCVanguard 2026 purpose


## 1. Overview
OrchIso adalah tool berbasis machine learning untuk mengisolasi bagian orkestra dari sebuah lagu audio. Tool ini menggunakan pipeline dua tahap: pertama memisahkan stem menggunakan Demucs, kemudian memperbaiki hasil separasi menggunakan model Spectrogram U-Net yang dapat dilatih sendiri.

Tujuan utama OrchIso adalah mengekstrak stem orkestra (strings, brass, woodwind, dan instrumen klasik lainnya) dari lagu yang sudah dicampur, terutama lagu pop atau klasik yang mengandung elemen orkestra.

## 1.1 Fitur Utama
Pipeline dua tahap: Demucs (Stage 1) + Spectrogram U-Net (Stage 2). Model berbasis spectrogram domain untuk menghindari distorsi waveform dan Mask-based output agar tidak menghasilkan noise yang tidak ada di input

Model telah support dataset MP3, WAV, FLAC. Training dengan early stopping otomatis dan learning rate scheduler, overlap-add processing dengan Hann window untuk transisi chunk yang mulus dan juga integrasi Google Drive untuk penyimpanan model dan dataset.

## 1.2 Arsitektur Pipeline
Pipeline OrchIso terdiri dari dua tahap utama:
| Tahap | Komponen | Input | Output |
| :--- | :--- | :--- | :--- |
| Stage 1 | Demucs (mdx_extra) | File lagu MP3/WAV | Stem: vocals, bass, drums, other |
| Stage 2 | Spectrogram U-Net 2D | Stem 'other' dari Demucs	| Orkestra yang sudah dibersihkan |

## 2. Requirements
## 2.1 Platform
a.Google Colab (direkomendasikan) dengan GPU T4 atau lebih tinggi

b. Google Drive untuk penyimpanan dataset dan model

c. Browser modern (Chrome, Firefox, Edge)
## 2.2 Library Python
| Library | Versi |
| :--- | :--- |
| PyTorch |	>=2.0	| 
| torchaudio | >=2.0	|
| Demucs | >=4.0 |
| ffmpeg |>=4.4 |
| numpy |	>=1.21 |

Langkah penggunaan telah disediakan pada file documentation.
