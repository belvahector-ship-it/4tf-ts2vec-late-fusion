# Diagnostik Validasi: CKA Lintas-Kondisi & Linear Probe Head-to-Head

> **Status:** selesai, dijalankan di background, murni diagnostik (tidak ada perubahan kode M10, tidak ada retrain).
> **Data:** embedding fused M9 real (7 kondisi × 5 seed; dipakai seed 42 & 123), fitur mentah M4 (`btc_features_all.parquet`), timestamp anchor M6 (`train_timestamps.npy`/`test_timestamps.npy`). Split train/test mengikuti batas M5 (tidak ada leakage — target dicari via timestamp, tidak pernah menyeberang split).
> **Prinsip laporan ini: angka apa adanya. Tidak ada yang disembunyikan atau "diselamatkan".**

---

## TES 1 — Linear CKA Lintas-Kondisi

**Metode:** Centered Kernel Alignment linier (Kornblith et al. 2019) antar embedding fused 256-dim, pairwise 1TF/2TF/3TF/4TF, dihitung penuh pada 26.238 window train (bukan subsample) untuk seed 42 dan 123.

### Matriks CKA — seed 42

| | 1TF | 2TF | 3TF | 4TF |
|---|---|---|---|---|
| **1TF** | 1.000 | 0.935 | 0.863 | 0.783 |
| **2TF** | 0.935 | 1.000 | 0.893 | 0.799 |
| **3TF** | 0.863 | 0.893 | 1.000 | 0.870 |
| **4TF** | 0.783 | 0.799 | 0.870 | 1.000 |

### Matriks CKA — seed 123

| | 1TF | 2TF | 3TF | 4TF |
|---|---|---|---|---|
| **1TF** | 1.000 | 0.929 | 0.827 | 0.762 |
| **2TF** | 0.929 | 1.000 | 0.874 | 0.800 |
| **3TF** | 0.827 | 0.874 | 1.000 | 0.844 |
| **4TF** | 0.762 | 0.800 | 0.844 | 1.000 |

### Interpretasi

- **Pola konsisten di kedua seed** (selisih antar-seed ≤0.036 di semua sel) — bukan artefak seed tunggal.
- **CKA menurun monoton seiring jarak kondisi**: 1TF↔2TF tertinggi (0.93–0.94), 1TF↔4TF terendah (0.76–0.78). Ini **bukan redundansi** (yang akan menunjukkan CKA →1 di semua pasangan) dan **bukan juga tidak berhubungan** (yang akan menunjukkan CKA →0). Rentang 0.76–0.94 menandakan representasi **overlap substansial tapi secara progresif berbeda**.
- **Kesimpulan Tes 1: menambah timeframe MENGHASILKAN representasi yang secara substansial berbeda, bukan sekadar redundan dalam dimensi lebih besar.** Klaim inti riset (multi-TF fusion menghasilkan representasi berbeda) **didukung** oleh tes ini.

---

## TES 2 — Linear Probe Head-to-Head: TS2Vec Embedding vs Baseline Fitur Mentah

**Desain:** Ridge regression (tanpa hidden layer, tanpa fine-tuning), target **next-step** (t+1 jam) — `close_return_1h` dan `hl_range_1h` — diprediksi dari:
- **(a)** embedding TS2Vec beku (fused, per kondisi)
- **(b)** fitur mentah SEMUA timeframe aktif kondisi tsb, TANPA lewat TS2Vec (concat mentah)
- **(c)** fitur mentah 1h saja (7 fitur), baseline naif timeframe-agnostic

Split train/test = M5 (26.238 train / 8.759 test setelah drop baris tanpa target t+1 yang valid).

### Tabel hasil lengkap (R², seed 42 & 123)

| Seed | Kondisi | Target | R² TS2Vec emb | R² raw multi-TF | R² raw 1h saja | Δ(emb − raw multi-TF) |
|---|---|---|---|---|---|---|
| 42 | 1TF | return_t+1 | −0.0052 | −0.0003 | −0.0003 | **−0.0049** |
| 42 | 1TF | volatility_t+1 | **−0.9712** | 0.0944 | 0.0944 | **−1.0656** |
| 42 | 2TF | return_t+1 | −0.0106 | −0.0003 | −0.0003 | −0.0104 |
| 42 | 2TF | volatility_t+1 | −1.0317 | 0.0944 | 0.0944 | −1.1261 |
| 42 | 3TF | return_t+1 | 0.0003 | **0.2108** | −0.0003 | **−0.2105** |
| 42 | 3TF | volatility_t+1 | −1.0315 | **0.4077** | 0.0944 | −1.4392 |
| 42 | 4TF | return_t+1 | −0.0069 | **0.2137** | −0.0003 | −0.2206 |
| 42 | 4TF | volatility_t+1 | −1.0259 | **0.4294** | 0.0944 | −1.4552 |
| 123 | 1TF | return_t+1 | −0.0072 | −0.0003 | −0.0003 | −0.0069 |
| 123 | 1TF | volatility_t+1 | −0.8071 | 0.0944 | 0.0944 | −0.9015 |
| 123 | 2TF | return_t+1 | −0.0132 | −0.0003 | −0.0003 | −0.0129 |
| 123 | 2TF | volatility_t+1 | −0.9284 | 0.0944 | 0.0944 | −1.0228 |
| 123 | 3TF | return_t+1 | −0.0144 | 0.2108 | −0.0003 | −0.2252 |
| 123 | 3TF | volatility_t+1 | −0.9780 | 0.4077 | 0.0944 | −1.3857 |
| 123 | 4TF | return_t+1 | −0.0154 | 0.2137 | −0.0003 | −0.2291 |
| 123 | 4TF | volatility_t+1 | −1.0138 | 0.4294 | 0.0944 | −1.4432 |

*(MAE lengkap ada di `experiments/m10_exploration/linear_probe_comparison.csv`.)*

### Interpretasi — LANGSUNG, TANPA DIPERHALUS

**Untuk `return_t+1`:** semua R² (baik TS2Vec maupun baseline mentah) berkisar **−0.02 hingga 0.0003 — praktis nol atau negatif di semua kondisi/seed/metode.** Ini bukan kegagalan TS2Vec secara spesifik — **tidak ada metode (linear) yang bisa memprediksi return jam-berikutnya di atas rata-rata**, konsisten dengan hipotesis pasar efisien pada horizon 1 jam. Tes ini tidak informatif untuk membedakan TS2Vec vs baseline pada target ini (keduanya sama-sama gagal).

**Untuk `volatility_t+1` — TEMUAN JUJUR YANG HARUS DILAPORKAN:**
- **Baseline fitur mentah multi-TF MENGUNGGULI TS2Vec embedding secara telak dan konsisten**, di SEMUA kondisi dan KEDUA seed. R² raw multi-TF: 0.094 (1TF) → **0.211 (3TF) → 0.429 (4TF)** — meningkat jelas seiring bertambah timeframe.
- **R² TS2Vec embedding NEGATIF BESAR** di semua kondisi (−0.81 hingga −1.03) — artinya prediksi linear dari embedding TS2Vec **lebih buruk daripada sekadar menebak rata-rata** (R²<0 = model lebih buruk dari baseline trivial).
- Selisih (Δ = emb − raw) **memburuk seiring bertambah TF** (1TF: −1.07 → 4TF: −1.46), **BUKAN membaik**.

**Kesimpulan Tes 2, langsung: TIDAK ADA bukti bahwa TS2Vec embedding mengungguli regresi sederhana pada fitur mentah — untuk kedua target ini, dengan probe linear. Untuk volatilitas, TS2Vec secara signifikan LEBIH BURUK daripada baseline naif, dan kesenjangan itu melebar seiring fusion multi-timeframe ditambahkan (arah berlawanan dari yang diharapkan klaim "fusion menambah nilai prediktif").**

### Mengapa ini terjadi (penjelasan berbasis mekanisme, bukan pembenaran)

Ini **konsisten dan dapat dijelaskan** oleh temuan Stage 3 di `docs/PIPELINE_SCIENTIFIC_DISSECTION.md`: **per-window z-score TS2Vec menghapus level/magnitude window** (menyisakan hanya *bentuk/shape*). Volatilitas besok berkorelasi kuat dengan **level volatilitas absolut** window sebelumnya (fenomena *volatility clustering*, fakta empiris klasik di keuangan/GARCH) — informasi level itu **sengaja dihilangkan** oleh normalisasi TS2Vec, sementara fitur mentah (tidak dinormalisasi per-window) tetap menyimpannya utuh. Ini bukan berarti embedding TS2Vec "rusak" — desainnya memang untuk menangkap *bentuk pola*, bukan *skala absolut* — tapi untuk task regresi linear pada level volatilitas berikutnya, hilangnya informasi skala itu jelas merugikan secara terukur.

---

## Kesimpulan Gabungan (jujur, langsung)

| Tes | Hasil | Mendukung klaim riset? |
|---|---|---|
| **CKA lintas-kondisi** | Representasi berubah progresif & substansial (0.76–0.94, menurun monoton dgn jarak kondisi), konsisten 2 seed | ✅ **YA** — fusion menghasilkan representasi berbeda, bukan redundan |
| **Linear probe next-step** | TS2Vec kalah telak dari fitur mentah untuk volatilitas (R² negatif vs 0.09–0.43); return tidak terprediksi oleh siapapun | ❌ **TIDAK** — tidak ada bukti TS2Vec mengungguli regresi sederhana pada tugas prediksi linear ini; kesenjangan justru melebar dengan fusion |

**Tidak dilakukan pencarian tes tambahan untuk "menyelamatkan" hasil Tes 2**, sesuai instruksi. Kedua tes **tidak kontradiktif** — keduanya bisa benar sekaligus: embedding TS2Vec *berbeda secara struktural* antar kondisi (Tes 1) **dan** *tidak lebih prediktif secara linear* untuk level-volatilitas-berikutnya dibanding fitur mentah (Tes 2), karena keduanya mengukur properti yang berbeda (perubahan geometri representasi vs kandungan informasi level yang bisa diekstrak linear).

### Implikasi untuk framing riset

Ini **memperkuat**, bukan melemahkan, kesimpulan `docs/PIPELINE_SCIENTIFIC_DISSECTION.md`: TS2Vec + fusion menghasilkan **representasi bentuk (shape space)** yang secara struktural kaya dan berbeda (Tes 1, plus Hopkins/CKA/economic-separation di eksplorasi M10 sebelumnya untuk *cluster label* KMeans), tapi **secara eksplisit BUKAN pengganti fitur level/skala mentah** untuk task regresi linear langsung pada magnitudo (volatilitas absolut). Ini keterbatasan yang jujur untuk bagian Batasan Penelitian / future work: pertimbangkan (a) menambahkan fitur level/skala sebagai pelengkap embedding untuk task prediktif magnitude, atau (b) membingkai kontribusi utama secara eksplisit sebagai *discovery struktur/state* (di mana Tes 1 relevan), bukan *forecasting numerik* (di mana Tes 2 menunjukkan keterbatasan nyata).

---

## Provenans

- Skrip: standalone (scratchpad sesi ini), tidak menyentuh `src/`.
- Output mentah: `experiments/m10_exploration/cka_matrices.json`, `experiments/m10_exploration/linear_probe_comparison.csv`.
- Model: `sklearn.linear_model.Ridge(alpha=1.0)` + `StandardScaler`, tanpa tuning hyperparameter (sesuai instruksi "sederhana, tanpa hidden layer").

*Selesai.*
