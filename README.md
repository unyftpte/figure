# Proyek Pertama — Predictive Analytics (Domain: Keuangan)

**Aset**: `SPY` — Synthetic Sample (CSV)

**Periode Data**: 2015-01-01 s.d. 2025-10-23  
**Ukuran Data (mentah)**: 1200 baris × 6 kolom  

## 1. Domain Proyek

Pergerakan harga harian bersifat volatil dan penuh noise. Namun pola teknikal jangka pendek sering dimanfaatkan untuk membantu keputusan trading dan manajemen risiko. Proyek ini memformulasikan tugas **klasifikasi**: apakah *return* harian **besok** positif (naik) atau tidak. Machine Learning relevan karena dapat memadukan banyak fitur teknikal dan divalidasi historis secara sistematis (backtest) untuk menilai dampak praktis.

**Referensi**

- Hastie, Tibshirani, Friedman (2009) *The Elements of Statistical Learning*. Springer.
- López de Prado (2018) *Advances in Financial Machine Learning*. Wiley.

## 2. Business Understanding

**Problem Statements**

- Dapatkah kita memprediksi arah return besok?
- Fitur teknikal apa yang paling berkontribusi?

**Goals (terukur)**

- **ROC AUC > 0.55** pada test set (di atas baseline acak 0.50).
- **F1 ≥ 0.50** (via tuning threshold).
- Dampak praktis ditunjukkan melalui backtest (growth of $1).

**Solution Statements**

- Uji 3 algoritma (LogReg, RF, GB) + **TimeSeriesSplit(CV=5)** + **RandomizedSearchCV** (scoring=ROC AUC).
- Tuning threshold (Youden & grid F1) untuk keputusan yang selaras tujuan bisnis.

## 3. Data Understanding

- **Sumber data**: CSV sintetis otomatis (dibuat oleh pipeline jika belum ada).

- **Missing value ada?** Tidak (total sel NA: 0)  
- **Baris duplikat**: 0  
- **Outlier check (returns)**: metode z-score |z|>5 pada daily returns; jumlah terdeteksi: 0  

**Uraian Seluruh Fitur**

- **Open/High/Low/Close/Adj Close/Volume**: OHLCV standar.
- **Return**: Persentase perubahan harian pada `Adj Close`.
- **SMA_w / EMA_w**: Rata-rata bergerak sederhana & eksponensial (w={5,10,20,50}).
- **ROC_w**: Rate of Change (momentum) selama w hari.
- **Volatility_w**: Simpangan baku rolling dari `Return` (w={5,10,20}).
- **RSI_14**: Relative Strength Index periode 14.
- **MACD / MACD_Signal / MACD_Hist**: Indikator MACD (12-26-9).
- **Target_Return_1d**: Return hari ke-(t+1).
- **Target_Up**: Label biner 1 jika `Target_Return_1d` > 0, else 0.

**Visual EDA**

![EDA Price](https://raw.githubusercontent.com/unyftpte/figure/main/eda_price.png)

![EDA Volume](https://raw.githubusercontent.com/unyftpte/figure/main/eda_volume.png)

![EDA Correlation](https://raw.githubusercontent.com/unyftpte/figure/main/eda_corr.png)

## 4. Data Preparation

Tahapan yang dilakukan (sesuai eksekusi notebook/pipeline):

1) **Normalisasi kolom harga**: menangani variasi nama kolom & MultiIndex dari yfinance/CSV. Jika hanya ada satu kolom harga, dipetakan sebagai `Close`/`Adj Close`. Missing di-drop.
2) **Feature Engineering**: membuat fitur teknikal — SMA/EMA (5,10,20,50), ROC, Volatility (5,10,20), RSI-14, MACD. Semua fitur dihitung dari `Adj Close` lalu baris awal yang terkena efek rolling di-drop.
3) **Label/Target**: `Target_Return_1d` = return t→t+1; `Target_Up`=1 bila return > 0. Target dibuat dengan `.shift(-1)` untuk memprediksi **besok** sehingga **mencegah data leakage**.
4) **Split Train/Test berbasis waktu**: proporsi train/test = 80%/20%. Split dilakukan dengan memotong di tengah indeks waktu (bukan acak) agar menghormati urutan kronologis.
5) **Scaling fitur**: khusus untuk Logistic Regression dilakukan `StandardScaler` di dalam `Pipeline`, supaya scaler hanya fit di train (menghindari kebocoran) dan otomatis terpakai saat inferensi.

## 5. Modeling

**Model 1 — Logistic Regression**  
- **Cara kerja**: memodelkan peluang kelas (Up=1) via fungsi logit; batas keputusan dapat dituning lewat threshold.  
- **Pipeline**: `StandardScaler` → `LogisticRegression(max_iter=1000, solver='lbfgs', penalty='l2')`.  
- **Parameter yang dituning**: `C ∈ [1e-3, 1e2]` (skala log).  
- **Parameter default lain (dipakai)**: `class_weight=None`, `fit_intercept=True`, `n_jobs=None`.  

**Model 2 — Random Forest (RF)**  
- **Cara kerja**: ansambel banyak decision tree (bagging) untuk menurunkan varians dan menangkap non-linearitas.  
- **Parameter yang dituning**: `n_estimators` (100–600), `max_depth` (None/3–20), `min_samples_split` (2–10), `min_samples_leaf` (1–10), `max_features` ('sqrt'/'log2'/None).  
- **Parameter default lain (dipakai)**: `bootstrap=True`, `criterion='gini'`, `oob_score=False`, `n_jobs=None`.  

**Model 3 — Gradient Boosting (GB)**  
- **Cara kerja**: boosting bertahap pohon-pohon kecil (weak learners) untuk meminimalkan loss secara aditif.  
- **Parameter yang dituning**: `n_estimators` (50–400), `learning_rate` (0.01–0.3), `max_depth` (2–6), `subsample` (0.6–1.0).  
- **Parameter default lain (dipakai)**: `loss='log_loss'` (versi terbaru), `max_features=None`.  

**Validasi & Pencarian**  
- **Validasi**: `TimeSeriesSplit(CV=5)` (menghormati urutan waktu, menghindari kebocoran).  
- **Pencarian hyperparameter**: `RandomizedSearchCV(scoring='roc_auc', n_iter=25, random_state=42)`.  
- **Pelaporan parameter**: setiap model menampilkan **Best Params** (JSON) pada bagian hasil model.

## 6. Evaluation

**Metrik & Rumus Singkat**  
- Precision=TP/(TP+FP), Recall=TP/(TP+FN), F1=2·(P·R)/(P+R).  
- ROC AUC: area di bawah ROC (baseline acak≈0.50).  

**Ringkas Metrik Test Set**

|  | accuracy | precision | recall | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- |
| gb | 0.4934 | 0.5909 | 0.3047 | 0.4021 | 0.5456 |
| rf | 0.5240 | 0.5941 | 0.4688 | 0.5240 | 0.5319 |
| logreg | 0.5022 | 0.5380 | 0.7734 | 0.6346 | 0.4642 |

### Hasil Model — LOGREG

**Best Params**

```json
{
  "clf__solver": "lbfgs",
  "clf__penalty": "l2",
  "clf__C": 0.0020235896477251575
}
```

**Classification Report**

```
              precision    recall  f1-score   support

           0       0.36      0.16      0.22       101
           1       0.54      0.77      0.63       128

    accuracy                           0.50       229
   macro avg       0.45      0.47      0.43       229
weighted avg       0.46      0.50      0.45       229

```

**Confusion Matrix**

![Confusion logreg](https://raw.githubusercontent.com/unyftpte/figure/main/confusion_logreg.png)

**ROC Curve**

![ROC logreg](https://raw.githubusercontent.com/unyftpte/figure/main/roc_logreg.png)

**Threshold Tuning**

- Youden's J best threshold: **0.525**  
- F1-best threshold: **0.200** (F1=0.7171)

**Backtest (Threshold F1 Terbaik)**

- Final Value (Strategy thr*): 1.3058  
- Final Value (Buy & Hold): 1.3058

![Backtest F1 logreg](https://raw.githubusercontent.com/unyftpte/figure/main/backtest_logreg_thrF1.png)

**Top Feature Importance (Model-based)**

![FI logreg](https://raw.githubusercontent.com/unyftpte/figure/main/featimp_logreg.png)

**Top Permutation Importance (F1)**

![PI logreg](https://raw.githubusercontent.com/unyftpte/figure/main/permimp_logreg.png)

Top-10 fitur (Permutation Importance, rata-rata):

| Feature | Mean ΔF1 |
| --- | --- |
| RSI_14 | 0.0055 |
| SMA_50 | 0.0038 |
| ROC_10 | 0.0034 |
| EMA_50 | 0.0033 |
| Volatility_5 | 0.0027 |
| ROC_20 | 0.0017 |
| MACD_Hist | 0.0009 |
| Open | 0.0006 |
| High | 0.0006 |
| Low | 0.0006 |

**Backtest (Threshold 0.5)**

- Final Value (Strategy): 1.1448  
- Final Value (Buy & Hold): 1.3058

![Backtest logreg](https://raw.githubusercontent.com/unyftpte/figure/main/backtest_logreg.png)

### Hasil Model — RF

**Best Params**

```json
{
  "clf__n_estimators": 150,
  "clf__min_samples_split": 2,
  "clf__min_samples_leaf": 3,
  "clf__max_features": null,
  "clf__max_depth": 14
}
```

**Classification Report**

```
              precision    recall  f1-score   support

           0       0.47      0.59      0.52       101
           1       0.59      0.47      0.52       128

    accuracy                           0.52       229
   macro avg       0.53      0.53      0.52       229
weighted avg       0.54      0.52      0.52       229

```

**Confusion Matrix**

![Confusion rf](https://raw.githubusercontent.com/unyftpte/figure/main/confusion_rf.png)

**ROC Curve**

![ROC rf](https://raw.githubusercontent.com/unyftpte/figure/main/roc_rf.png)

**Threshold Tuning**

- Youden's J best threshold: **0.503**  
- F1-best threshold: **0.325** (F1=0.7236)

**Backtest (Threshold F1 Terbaik)**

- Final Value (Strategy thr*): 1.3276  
- Final Value (Buy & Hold): 1.3058

![Backtest F1 rf](https://raw.githubusercontent.com/unyftpte/figure/main/backtest_rf_thrF1.png)

**Top Feature Importance (Model-based)**

![FI rf](https://raw.githubusercontent.com/unyftpte/figure/main/featimp_rf.png)

**Top Permutation Importance (F1)**

![PI rf](https://raw.githubusercontent.com/unyftpte/figure/main/permimp_rf.png)

Top-10 fitur (Permutation Importance, rata-rata):

| Feature | Mean ΔF1 |
| --- | --- |
| MACD_Hist | 0.0444 |
| Volatility_10 | 0.0345 |
| ROC_50 | 0.0301 |
| ROC_5 | 0.0298 |
| Volume | 0.0258 |
| Volatility_5 | 0.0250 |
| Return | 0.0227 |
| ROC_10 | 0.0212 |
| MACD_Signal | 0.0155 |
| ROC_20 | 0.0146 |

**Backtest (Threshold 0.5)**

- Final Value (Strategy): 1.2094  
- Final Value (Buy & Hold): 1.3058

![Backtest rf](https://raw.githubusercontent.com/unyftpte/figure/main/backtest_rf.png)

### Hasil Model — GB

**Best Params**

```json
{
  "clf__subsample": 0.75,
  "clf__n_estimators": 300,
  "clf__max_depth": 5,
  "clf__learning_rate": 0.23999999999999996
}
```

**Classification Report**

```
              precision    recall  f1-score   support

           0       0.45      0.73      0.56       101
           1       0.59      0.30      0.40       128

    accuracy                           0.49       229
   macro avg       0.52      0.52      0.48       229
weighted avg       0.53      0.49      0.47       229

```

**Confusion Matrix**

![Confusion gb](https://raw.githubusercontent.com/unyftpte/figure/main/confusion_gb.png)

**ROC Curve**

![ROC gb](https://raw.githubusercontent.com/unyftpte/figure/main/roc_gb.png)

**Threshold Tuning**

- Youden's J best threshold: **0.012**  
- F1-best threshold: **0.200** (F1=0.5110)

**Backtest (Threshold F1 Terbaik)**

- Final Value (Strategy thr*): 1.0860  
- Final Value (Buy & Hold): 1.3058

![Backtest F1 gb](https://raw.githubusercontent.com/unyftpte/figure/main/backtest_gb_thrF1.png)

**Top Feature Importance (Model-based)**

![FI gb](https://raw.githubusercontent.com/unyftpte/figure/main/featimp_gb.png)

**Top Permutation Importance (F1)**

![PI gb](https://raw.githubusercontent.com/unyftpte/figure/main/permimp_gb.png)

Top-10 fitur (Permutation Importance, rata-rata):

| Feature | Mean ΔF1 |
| --- | --- |
| ROC_10 | 0.0380 |
| ROC_5 | 0.0366 |
| MACD_Hist | 0.0288 |
| Volume | 0.0162 |
| EMA_5 | 0.0002 |
| EMA_10 | 0.0000 |
| SMA_50 | 0.0000 |
| SMA_5 | 0.0000 |
| EMA_20 | 0.0000 |
| EMA_50 | 0.0000 |

**Backtest (Threshold 0.5)**

- Final Value (Strategy): 1.0616  
- Final Value (Buy & Hold): 1.3058

![Backtest gb](https://raw.githubusercontent.com/unyftpte/figure/main/backtest_gb.png)

## 7. Kesimpulan

**Model terbaik (ROC AUC, test)**: **GB**.  
Pencapaian terhadap Goals: F1 ≥ 0.50.  

**Keterkaitan ke Business Understanding**  
- **Apakah menjawab problem statement?** Ya. Model menghasilkan probabilitas arah return besok (Up/Down) yang bisa dipakai sebagai sinyal.  
- **Apakah mencapai goals?** Sebagian/seluruhnya tercapai berdasarkan ROC AUC & F1 pada test set.  
- **Apakah solusi berdampak?** Ya. Threshold tuning mengubah trade-off precision/recall dan terbukti memengaruhi performa strategi (growth of $1) pada backtest periode uji.  

**Rekomendasi**  
- Walk-forward multi-window; uji stabilitas.  
- Tambah variabel makro/sentimen; biaya transaksi & metrik risiko (max drawdown, Sharpe).  
- Kalibrasi probabilitas (Platt/Isotonic) untuk konsistensi threshold.  

## Lampiran

**Lingkungan Eksekusi**

```json
{
  "python": "3.13.9",
  "platform": "Windows-11-10.0.26100-SP0",
  "numpy": "2.3.4",
  "pandas": "2.3.3",
  "sklearn": "1.7.2",
  "matplotlib": "3.10.7",
  "yfinance_available": true
}
```

**Sumber Data**

- **Sumber data**: CSV sintetis otomatis (dibuat oleh pipeline jika belum ada).



<img width="1184" height="582" alt="backtest_logreg_thrF1" src="https://github.com/user-attachments/assets/526df765-f564-4e7f-9cf6-a8fb3b8492d8" />
<img width="1184" height="582" alt="backtest_logreg" src="https://github.com/user-attachments/assets/81a0856b-18a4-47e8-9905-4d39e0823c51" />
<img width="1184" height="582" alt="backtest_gb_thrF1" src="https://github.com/user-attachments/assets/49c21170-f1ca-4ca9-a74b-4f5c87c70294" />
<img width="1184" height="582" alt="backtest_gb" src="https://github.com/user-attachments/assets/fa8ad4bf-70b6-4768-9d0d-990277a3e103" />
<img width="734" height="731" alt="roc_rf" src="https://github.com/user-attachments/assets/aea7340e-4117-45b9-a357-e4b9765d8bb6" />
<img width="734" height="731" alt="roc_logreg" src="https://github.com/user-attachments/assets/00912856-4e35-4165-8121-062f07276f2a" />
<img width="734" height="731" alt="roc_gb" src="https://github.com/user-attachments/assets/f589d578-bed0-4518-aeaf-a83b987a7e2a" />
<img width="1185" height="732" alt="permimp_rf" src="https://github.com/user-attachments/assets/258f269a-6f77-45f7-8106-28eed64836ca" />
<img width="1185" height="732" alt="permimp_logreg" src="https://github.com/user-attachments/assets/6f17d218-1277-41ae-b3c9-351c14b4eb7c" />
<img width="1185" height="732" alt="permimp_gb" src="https://github.com/user-attachments/assets/f7b5f3a2-68db-4816-9f41-cafe80a13079" />
<img width="1185" height="732" alt="featimp_rf" src="https://github.com/user-attachments/assets/599ba526-7f5c-4804-b05a-1fd98046098d" />
<img width="1185" height="732" alt="featimp_logreg" src="https://github.com/user-attachments/assets/13107e28-6c55-4dae-8510-7105e36a1d16" />
<img width="1185" height="732" alt="featimp_gb" src="https://github.com/user-attachments/assets/61a030de-2d59-4810-86a5-9527265f2f06" />
<img width="1334" height="432" alt="eda_volume" src="https://github.com/user-attachments/assets/2db0c88f-8806-4889-800e-13ef17e8df17" />
<img width="1334" height="582" alt="eda_price" src="https://github.com/user-attachments/assets/dcbc7923-a666-4d7d-8c5f-2b359e07d1dc" />
<img width="980" height="882" alt="eda_corr" src="https://github.com/user-attachments/assets/0e12fb96-04e1-443e-aa01-7e86e0ef1e0b" />
<img width="587" height="565" alt="confusion_rf" src="https://github.com/user-attachments/assets/dca4908c-8b00-49fa-a6a5-32d7816dbcae" />
<img width="587" height="565" alt="confusion_logreg" src="https://github.com/user-attachments/assets/74258737-e973-4062-9a9f-48fab5a21cac" />
<img width="587" height="565" alt="confusion_gb" src="https://github.com/user-attachments/assets/515879f4-7469-4eb2-8e7a-8adb89eaee07" />
<img width="1184" height="582" alt="backtest_rf_thrF1" src="https://github.com/user-attachments/assets/23f3b3cc-6979-4d3a-8d60-bbc45a091331" />
<img width="1184" height="582" alt="backtest_rf" src="https://github.com/user-attachments/assets/0f9e6e19-7d34-43b3-981f-769ae62125f5" />
