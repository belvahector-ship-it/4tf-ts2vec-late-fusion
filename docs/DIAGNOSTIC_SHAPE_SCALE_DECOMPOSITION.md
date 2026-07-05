# Diagnostik Hipotesis "Shape-Scale Decomposition"

> **Status:** selesai, dijalankan di background, murni diagnostik (tidak ada perubahan kode M10, tidak ada retrain).
> **Hipotesis diuji:** TS2Vec tidak menambah informasi secara umum, tapi secara selektif mempertahankan **bentuk (shape)** sambil membuang **skala/magnitude** — konsekuensi langsung normalisasi per-window (ADR-016).
> **Data:** embedding fused M9 real (seed 42 & 123), window mentah pra-normalisasi (dihitung ulang read-only dari `WindowGenerator.extract_windows`, tidak mengubah `src/`), fitur M4 (`btc_features_all.parquet`).
> **Prinsip laporan: angka apa adanya, termasuk yang bertentangan dengan ekspektasi hipotesis.**

---

## TES A — Prediksi σ_w (skala window) dari embedding vs fitur mentah

**Metode:** Ridge regression (train M5 → test M5), target σ_w = rata-rata std per-channel window (penyebut z-score ADR-016, dihitung dari window RAW sebelum normalisasi). Tiga prediktor: (a) embedding TS2Vec fused, (b) window mentah flatten (fair, harus menyimpulkan skala dari nilai-nilai linear — non-trivial untuk model linear), (c) fitur statistik-mentah-teknik (mean/std per-channel langsung — *sanity check* batas atas, secara literal memuat jawabannya).

| Seed | Kondisi | R² TS2Vec emb | R² raw window flatten | R² sanity (upper bound) |
|---|---|---|---|---|
| 42 | 1TF | 0.301 | 0.519 | 1.000 |
| 42 | 2TF | 0.296 | 0.519 | 1.000 |
| 42 | 3TF | 0.340 | 0.533 | 1.000 |
| 42 | 4TF | **0.476** | **0.444** | 1.000 |
| 123 | 1TF | 0.281 | 0.519 | 1.000 |
| 123 | 2TF | 0.305 | 0.519 | 1.000 |
| 123 | 3TF | 0.354 | 0.533 | 1.000 |
| 123 | 4TF | **0.468** | **0.444** | 1.000 |

**Sanity check R²=1.000 tepat** di semua baris → pipeline pengukuran benar (fitur yang secara harfiah memuat std memang memprediksi std sempurna).

**Interpretasi — SESUAI EKSPEKTASI SEBAGIAN, BUKAN PENUH:**
- Ekspektasi hipotesis murni: R²(embedding) ≈ 0 (skala terhapus total). **Hasil aktual: R²(embedding) = 0.28–0.48 — SUBSTANSIAL, bukan nol.** Skala **berkurang**, bukan **hilang total**, dari embedding.
- R²(raw) memang lebih tinggi dari R²(embedding) di 1TF/2TF/3TF (selisih 0.18–0.22), **konsisten arah hipotesis**.
- **ANOMALI di 4TF (dua seed, konsisten):** R²(embedding)=0.48 justru **melebihi** R²(raw)=0.44. Ini **berlawanan arah** dengan hipotesis. Kemungkinan penjelasan (bukan pembenaran, murni dugaan teknis): prediktor raw 4TF berdimensi 4×336=1.344 (4 timeframe × 48×7) jauh lebih tinggi dari embedding 256-dim pada N=26.238 — model Ridge linear pada dimensi setinggi itu berisiko overfit/kurang stabil pada test set, sehingga R²-nya turun relatif terhadap kondisi berdimensi lebih rendah (1TF raw=336-dim). Ini **bukan bukti bahwa embedding menyimpan lebih banyak skala** — hipotesis alternatif yang sama validnya adalah keterbatasan model linear pada dimensi tinggi.

**Kesimpulan TES A: MENDUKUNG SEBAGIAN.** Arah keseluruhan (embedding < raw untuk memprediksi skala) benar di 3/4 kondisi, tapi skala TIDAK hilang total (R² jauh dari nol) dan ada satu anomali terbalik (4TF) yang tidak bisa diabaikan.

---

## TES B — Korelasi σ_w saat ini vs volatility_t+1 (volatility clustering)

| Kondisi | Split | n | Pearson r | p (Pearson) | Spearman ρ | p (Spearman) |
|---|---|---|---|---|---|---|
| 1TF | train | 26.238 | 0.221 | ~0 (2.6e-288) | 0.215 | ~0 |
| 1TF | test | 8.759 | 0.177 | 2.0e-62 | 0.222 | 6.4e-98 |
| 2TF | train/test | idem 1TF (σ_w rata-rata 15m+1h ≈ dominan 1h) | | | | |
| 3TF | train | 26.238 | 0.250 | ~0 | 0.245 | ~0 |
| 3TF | test | 8.759 | 0.205 | 2.4e-83 | 0.244 | 1.3e-118 |
| 4TF | train | 26.238 | **0.285** | ~0 | 0.271 | ~0 |
| 4TF | test | 8.759 | 0.201 | 1.7e-80 | 0.240 | 5.4e-115 |

**Interpretasi: MENDUKUNG.** Korelasi positif, **secara statistik sangat signifikan** (p mendekati 0) di semua kondisi dan kedua split (train **dan** test — bukan hanya in-sample), mengonfirmasi *volatility clustering* nyata dalam data. Kekuatan korelasi **moderat** (r≈0.18–0.29) — jujur, bukan "sangat kuat" seperti kadang diasumsikan di literatur GARCH sederhana, tapi cukup untuk menjadi prasyarat valid bagi argumen Tes A: **jika** σ_w tidak masuk ke embedding, embedding kehilangan sinyal prediktif riil (bukan sinyal palsu).

---

## TES C — Retrieval: bentuk vs skala di ruang TS2Vec vs ruang mentah

**Metode:** 60 query acak (test split, seed sampling=42) per kondisi {1TF, 4TF}; top-5 NN dicari di (a) ruang embedding TS2Vec (cosine, karena embedding berada di shell — lihat dissection Stage 4) dan (b) ruang window mentah flatten (Euclidean, skala utuh). Untuk tiap pasangan: jarak bentuk = DTW multivariat pada window **z-scored** (input asli TS2Vec), selisih skala = |σ_w(query) − σ_w(neighbor)|.

### Ringkasan (rata-rata atas 300 pasangan per sel)

| Kondisi | Metode | DTW bentuk (mean±std) | Selisih skala σ_w (mean±std) |
|---|---|---|---|
| 1TF | raw_window | 123.58 ± 19.46 | 0.0138 ± 0.0105 |
| 1TF | **ts2vec** | **20.43 ± 20.92** | **0.0035 ± 0.0054** |
| 4TF | raw_window | 247.45 ± 96.67 | 0.0093 ± 0.0087 |
| 4TF | **ts2vec** | **36.88 ± 21.41** | **0.0029 ± 0.0032** |

**Bagian bentuk (DTW): SESUAI EKSPEKTASI.** Neighbor TS2Vec konsisten jauh lebih dekat secara bentuk (DTW 20–37) dibanding neighbor raw (124–247) — TS2Vec memang memprioritaskan kemiripan bentuk.

**Bagian skala: BERTOLAK BELAKANG DENGAN EKSPEKTASI EKSPLISIT poin 4 instruksi.** Ekspektasi: neighbor TS2Vec seharusnya menunjukkan selisih skala TINGGI/bervariasi (skala diabaikan). **Hasil aktual: selisih skala neighbor TS2Vec justru LEBIH RENDAH** (0.0029–0.0035) dibanding neighbor raw (0.0093–0.0138) — arah terbalik.

### 🔍 Investigasi lanjutan (dilakukan untuk memahami anomali ini, bukan untuk "menyelamatkan" hipotesis)

Diperiksa selisih indeks waktu `|neighbor_idx − query_idx|` untuk seluruh 600 pasangan (bukan hanya 3 contoh ilustrasi):

| Kondisi | Metode | Median gap indeks | Proporsi gap ≤ 5 | Proporsi gap ≤ 1 |
|---|---|---|---|---|
| 1TF | raw_window | 2.112 | 2.3% | 2.3% |
| 1TF | **ts2vec** | **2** | **95.3%** | 36.7% |
| 4TF | raw_window | 1.872 | 14.3% | 14.3% |
| 4TF | **ts2vec** | **2** | **100%** | 38.7% |

**Penyebab ditemukan: neighbor TS2Vec HAMPIR SELALU (95–100%) adalah window yang secara temporal sangat berdekatan** (median gap indeks 1–2 jam), sedangkan neighbor raw tersebar acak sepanjang seluruh test set (median gap ~1.900–2.100, mendekati sampling seragam). Ini **konsisten dan menjelaskan** temuan geometri sebelumnya (`docs/M10_CLUSTERING_EXPLORATION.md` §3.4): window stride-1 overlap 47/48 baris menciptakan **trajektori kontinu (filamen)** — window yang bersebelahan waktu otomatis mirip di ruang embedding TS2Vec, **dalam SEMUA aspek sekaligus** (bentuk **dan** skala, karena rezim volatilitas berubah perlahan — persis fenomena yang dikonfirmasi Tes B).

**Implikasi jujur: Tes C, sebagaimana dirancang, TIDAK dapat mengisolasi "bentuk vs skala" secara independen dari waktu.** Retrieval TS2Vec pada dasarnya menemukan **tetangga temporal** (window t±1, t±2), bukan kecocokan bentuk lintas periode waktu yang berjauhan. Selisih skala yang rendah pada neighbor TS2Vec bukan bukti "skala turut dipertahankan" — itu efek samping dari kedekatan waktu yang mendominasi retrieval, sebuah **confound struktural** yang berasal dari desain windowing (stride=1), bukan dari perilaku selektif shape-vs-scale encoder itu sendiri.

**Kesimpulan TES C: bagian bentuk (DTW rendah) mendukung hipotesis, tapi bagian skala TIDAK bisa disimpulkan — hasil bertentangan dengan ekspektasi, dan penyebabnya adalah confound temporal-adjacency, bukan bukti valid bahwa skala "dipertahankan".** Sesuai instruksi, saya tidak merancang tes tambahan untuk memaksa kesimpulan yang diinginkan — confound ini dilaporkan apa adanya sebagai keterbatasan desain tes.

### Contoh ilustratif (untuk paper, cond=4TF, neighbor rank-1)

| query_idx | DTW raw | DTW ts2vec | Δskala raw | Δskala ts2vec | neighbor_idx raw | neighbor_idx ts2vec |
|---|---|---|---|---|---|---|
| 6156 | 313.45 | 20.77 | 0.00349 | 0.00783 | 6804 | **6155** (query−1) |
| 7462 | 273.87 | 26.00 | 0.00189 | 0.00456 | 7870 | **7463** (query+1) |
| 7760 | 277.87 | 14.11 | 0.00016 | 0.00247 | 272 | **7761** (query+1) |

Pola di 3 contoh ini persis mengonfirmasi temuan kuantitatif di atas: neighbor TS2Vec = window bersebelahan (±1), neighbor raw = window jauh secara acak.

---

## Kesimpulan Gabungan — LANGSUNG, TIDAK DIPERHALUS

| Tes | Hasil | Mendukung hipotesis "shape-scale decomposition"? |
|---|---|---|
| **A — prediksi σ_w** | Embedding R²=0.28–0.48 (bukan ~0); raw R²=0.44–0.53; arah benar di 3/4 kondisi, **terbalik di 4TF** | **SEBAGIAN** — skala berkurang, tidak hilang total; ada anomali tak terjelaskan bersih |
| **B — σ_w vs vol masa depan** | Korelasi positif signifikan (r 0.18–0.29, p→0) di semua kondisi & split | **MENDUKUNG** — mengonfirmasi prasyarat (volatility clustering nyata) |
| **C — retrieval bentuk vs skala** | DTW rendah untuk TS2Vec (✅ sesuai), TAPI selisih skala JUGA rendah (❌ berlawanan arah) — disebabkan confound temporal-adjacency (95–100% neighbor = window ±1–2 jam) | **TIDAK KONKLUSIF** untuk komponen skala; confound structural ditemukan |

**Jawaban langsung: hipotesis "Shape-Scale Decomposition" mendapat dukungan CAMPURAN/PARSIAL, bukan konfirmasi bersih.**

- **Yang didukung kuat:** TS2Vec memprioritaskan kemiripan **bentuk** (DTW rendah, Tes C) dan skala window (σ_w) memang membawa sinyal prediktif riil terhadap volatilitas masa depan (Tes B) — jadi **jika** informasi ini hilang dari embedding, itu benar-benar kerugian, bukan sinyal derau.
- **Yang TIDAK didukung bersih:** klaim bahwa skala "dibuang" dari embedding. R² Tes A untuk embedding jauh dari nol (0.28–0.48), dan satu kondisi (4TF) bahkan menunjukkan arah terbalik. Tes C tidak bisa menguji komponen skala secara independen karena confound structural (trajectori kontinu akibat window overlap mendominasi retrieval, sebagaimana sudah ditemukan di eksplorasi M10 sebelumnya).

**Interpretasi yang lebih akurat (menggantikan "shape-scale decomposition murni"):** normalisasi per-window **mengurangi** kontribusi skala terhadap embedding secara terukur (Tes A, 3/4 kondisi), tapi tidak menghapusnya sepenuhnya secara linear-recoverable — kemungkinan karena sisa korelasi antar-channel (mis. `hl_range`/`body_ratio` yang berkorelasi dengan level volatilitas relatif meski dinormalisasi per window) tetap membawa jejak skala tak-langsung ke dalam bentuk. Klaim yang lebih defensif untuk paper: **"normalisasi per-window secara signifikan MENGURANGI — bukan sepenuhnya menghilangkan — kandungan informasi skala pada embedding, dengan bukti kuantitatif campuran; window overlap (stride-1) menimbulkan confound temporal yang membatasi kesimpulan tegas dari uji retrieval."**

---

## Provenans

- Skrip: standalone (scratchpad sesi ini), memanggil `WindowGenerator.extract_windows`/`normalize_window` milik `src/data/window_generation.py` secara read-only (tidak ada modifikasi kode).
- Output mentah: `experiments/m10_exploration/{tesA_sigma_prediction.csv, tesB_sigma_vs_futurevol.csv, tesC_retrieval_comparison.csv, tesC_summary.csv, tesC_illustrative_examples.csv}`.

*Selesai.*
