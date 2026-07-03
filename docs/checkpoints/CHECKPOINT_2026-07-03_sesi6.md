# CHECKPOINT — Sesi 6 — 2026-07-03

## Status Global
Milestone saat ini: **2 — Data Pipeline Complete** (in progress, 5/6 modul selesai)
Modul terakhir DONE: **M5 — Temporal Split** (kode lengkap, verifikasi nyata penuh)
Modul sedang IN PROGRESS: tidak ada — menunggu instruksi lanjut ke M6

## Yang Sudah Selesai Sesi Ini

### Kode (`src/data/`)
- `temporal_split.py` — modul lengkap M5:
  - `TemporalSplitter.split()` — filter boolean murni berdasarkan timestamp absolut (`TRAIN_END = 2022-12-31 23:00:00 UTC`, `TEST_START = 2023-01-01 00:00:00 UTC`), tidak bergantung pada bug timestamp M4 karena boundary berbasis kalender absolut
  - `check_split_boundary()` — V-DATA-005: train max harus persis `TRAIN_END`, test min harus persis `TEST_START`
  - `check_no_overlap()` — V-DATA-005: nol duplikat timestamp lintas split
  - `check_split_sizes()` — verifikasi ~26,269 train / ~8,760 test sesuai DS-02 v1.1 (dengan timestamp start yang sudah dikoreksi)
  - `verify_lc3()` — **implementasi V-LEAK-002/LC-3 yang correctly-scoped**: tidak mencoba "menghilangkan" efek rolling-window boundary (itu akan bertentangan dengan DS-02 v1.1 yang secara eksplisit mendokumentasikan dan menerima efek ini), melainkan memverifikasi split sendiri tidak menambah leakage baru + melaporkan interaksi rolling-window sebagai catatan informasional yang sudah pre-registered
  - `run_temporal_split()` — orkestrator penuh, `TemporalSplitError` informatif

### CLI (`scripts/`)
- `run_m5_split.py` — baca `data/processed/btc_features_all.parquet`, tulis `train_features.parquet` + `test_features.parquet`

### Test (`tests/test_temporal_split.py`)
- ~25 test case: boundary exact-match, overlap detection (positif+negatif), LC-3 documentation check, dan **dataset ukuran asli 35,045 baris**

## Keputusan Desain Penting Sesi Ini

**Scoping V-LEAK-002 dengan benar:** DS-02 v1.1 LC-3 secara eksplisit menyatakan bahwa efek rolling-window `volume_zscore` di boundary split **bukan leakage** — itu diakui, didokumentasikan, dan "dinetralkan" nanti oleh per-window normalization di M6 (Stage 5), bukan oleh M5. Modul `verify_lc3()` dirancang untuk **tidak mencoba memperbaiki** sesuatu yang menurut protokol yang disetujui memang tidak perlu diperbaiki di tahap ini — ini konsisten dengan prinsip "no research decisions change during implementation" dari IMP-01. Modul ini memverifikasi hal yang benar-benar jadi tanggung jawab M5 (split tidak menambah leakage baru: tidak ada baris train yang bocor ke sisi test atau sebaliknya) dan melaporkan interaksi rolling-window sebagai catatan informasional yang mengarahkan pembaca ke tempat verifikasi sebenarnya (V-LEAK-004 di M6).

## Verifikasi yang BENAR-BENAR Dijalankan (bukan klaim)

M5 murni `pandas`+`numpy` — verifikasi nyata penuh dijalankan:

✅ **Bagian 1 (10 assertion):** Split logic tepat di boundary — baris `23:00` (TRAIN_END) masuk train, baris `00:00` (TEST_START) masuk test, dikonfirmasi via log nyata. `check_split_boundary`/`check_no_overlap` PASS pada data benar, dan **terbukti mendeteksi bug yang disuntikkan** (duplicate timestamp injection)
✅ **Bagian 2 (11 assertion):** `verify_lc3` PASS pada split benar dengan catatan informasional lengkap; **terbukti mendeteksi row yang bocor lewat boundary** (train berisi timestamp setelah TRAIN_END). **Dataset ukuran asli 35,045 baris** (mulai `2020-01-01 19:00:00 UTC`, sesuai desain M4 yang sudah dikoreksi) menghasilkan **train=26,285 baris, test=8,760 baris** — dikonfirmasi log nyata, cocok dengan klaim DS-02 v1.1 "~26,269"/"~8,760" dalam toleransi wajar
✅ **Regression check M0-M5 lengkap:** semua modul (kecuali `device.py`) tetap ter-import tanpa error

**Hasil total: 21/21 assertion PASS**, semuanya dijalankan nyata dengan output log yang bisa diaudit.

## Yang BELUM Bisa Diverifikasi Nyata di Sandbox Ini

⚠️ `pytest` tetap tidak ter-install (tanpa network).
⚠️ `scripts/run_m5_split.py` belum dieksekusi end-to-end (butuh output M4→M3→M1 yang butuh network).

## Keputusan/Deviasi
Tidak ada deviasi dari DS-01/DS-02/DS-03. Keputusan scoping `verify_lc3()` (di atas) adalah interpretasi implementasi yang setia pada protokol yang sudah disetujui, bukan penyimpangan.

## Item Terbuka
1. Commit hash TS2Vec — status tetap "Pending".
2. Jalankan `pytest tests/ -v` di environment ber-network.
3. Fork fallback TS2Vec — buat setelah Milestone 1 selesai.
4. Pertimbangkan update manual `.docx` asli DS-04 jika dipakai di luar repo (masih ada dari checkpoint sesi 5).

## TODO List Terbaru

- [x] M0 — Project Bootstrap **[DONE]**
- [x] M1 — Data Acquisition **[DONE]**
- [x] M2 — Data Validation **[DONE]**
- [x] M3 — Temporal Alignment **[DONE]**
- [x] **[AUDIT]** DS-01–DS-04 + IMP-01 logic audit **[DONE]**
- [x] M4 — Feature Engineering **[DONE]**
- [x] M5 — Temporal Split **[DONE — verifikasi nyata penuh, 21/21 PASS]**
- [ ] M6 — Window Generation ← **lanjut di sini** (modul terakhir sebelum V-LEAK gate)
- [ ] M7 — TS2Vec Wrapper (masih menunggu commit hash TS2Vec, status Pending)
- [ ] **[GATE] V-LEAK-004** — per-window z-score normalization, harus lulus sebelum M8
- [ ] M8 — Branch Training
- [ ] M9 — Fusion
- [ ] M10.5 — External Baselines (HMM + KM-PCA)
- [ ] M10 — HDBSCAN Clustering
- [ ] M11 — Evaluation
- [ ] M12 — Visualization (paralel M14)
- [ ] M14 — Statistical Analysis (paralel M12)
- [ ] M13 — Experiment Runner
- [ ] M15 — Paper Artifact Generator

## Instruksi untuk Sesi Berikutnya
**Mulai dari:** M6 — Window Generation (DS-02 v1.1 Stage 5). Sliding window W=48, stride=1, per-window z-score normalization (mean/std dari window itu sendiri, +1e-8 epsilon). Ini adalah **modul terakhir sebelum V-LEAK gate wajib** (V-LEAK-001 s.d. V-LEAK-004 harus semua lulus sebelum M8 training dimulai — V-LEAK-001 sudah selesai di M3, V-LEAK-002 sudah di-scope dengan benar di M5, V-LEAK-003 dan V-LEAK-004 diimplementasikan di M6 ini).
**File yang perlu dilihat pertama:** `docs/DS-02_v1.1.md` bagian "Stage 5: Window Generation and Normalization" (perhatikan LC-4: 47 test window pertama mengandung timestamp training — ini WAJAR dan didokumentasikan, bukan bug), `docs/DS-04_v1.1.md` bagian V-LEAK-003 dan V-LEAK-004.
**Catatan:** M6 tetap pure pandas/numpy (bisa diverifikasi nyata penuh). Setelah M6 selesai, Milestone 2 (Data Pipeline Complete) tercapai — saatnya jalankan gate V-LEAK lengkap sebelum lanjut ke M7/M8.
