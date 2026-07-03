# CHECKPOINT — Sesi 7 — 2026-07-03

## Status Global
Milestone saat ini: **Milestone 2 — Data Pipeline Complete: TERCAPAI** ✅
Modul terakhir DONE: **M6 — Window Generation** (kode lengkap, verifikasi nyata penuh, dibangun dengan desain hasil audit LC-4)
Modul sedang IN PROGRESS: tidak ada — menunggu instruksi lanjut ke M7 (terhambat status TS2Vec "Pending")

## Peristiwa Penting Sesi Ini: Audit LC-4 dan Perubahan Desain Arsitektur

Saat verifikasi nyata M6 versi awal (window dari file M5 yang sudah displit), ditemukan **overlap boundary = 0**, bertentangan dengan klaim "up to 47 test windows contain training-period timestamps" yang muncul konsisten di DS-02, DS-03, dan DS-04. Ini **Bug #3** — berbeda karakter dari Bug #1/#2 sebelumnya karena merupakan **kontradiksi struktural** (dua desain arsitektur berbeda tercampur dalam satu paragraf LC-4), bukan sekadar salah hitung angka.

### Audit & Keputusan

Dibuktikan secara matematis dua skenario:
- **Skenario A** (window dari file M5 yang sudah displit): `N_test_windows=8,713` cocok 1 klaim tabel DS-02, tapi overlap=0 tidak cocok klaim "up to 47"
- **Skenario C** (window dari feature matrix PENUH M4, dikategorikan train/test berdasarkan *anchor timestamp*): overlap=47 cocok **4 referensi independen** (DS-02 LC-4, DS-03 CF-002, DS-03 Protocol B, DS-04 V-LEAK-003), tapi `N_test_windows=8,760` tidak cocok 1 angka tabel DS-02

Anda menyetujui **Skenario C** sebagai desain final, dengan alasan "4 vs 1" — memperbaiki 1 bagian yang tertinggal lebih masuk akal daripada mengabaikan 4 bagian yang saling konsisten.

### 3 Dokumen Direvisi & Diterapkan ke Repo

| Dokumen | Versi | Perubahan Utama |
|---|---|---|
| DS-02 | v1.1→**v1.2** | Stage 5 ditulis ulang: window dari feature matrix penuh, kategorisasi by anchor. `N_test_windows`: ≈8,713 → **8,760 exact**. LC-4 diperbaiki agar tidak lagi kontradiktif |
| IMP-01 | v1.2→**v1.3** | M6 Inputs: `train/test_features.parquet` → `btc_features_all.parquet`. Dependency graph: M6 sekarang depends **M4** (bukan M5). M5⊥M6 (paralel). M10.5 dependency label diperbaiki (M6→M5, bug kecil terpisah yang ikut ditemukan) |
| — | — | `AUDIT_LC4_ADDENDUM.md` dibuat, disalin ke `docs/` sebagai referensi permanen |

**M5 tidak diubah** — tetap valid, perannya berubah dari "prasyarat M6" menjadi "cabang paralel penghasil artefak audit/ekonomi" (dipakai nanti di M10.5 dan M11/Stage 9).

## M6 — Window Generation: Selesai dengan Desain Skenario C

### Kode (`src/data/`)
- `window_generation.py`:
  - `WindowGenerator.extract_windows()` — sliding window W=48/stride=1 atas **feature matrix penuh** (bukan file split), merekam anchor (baris terakhir) dan earliest (baris pertama) timestamp
  - `categorize_by_anchor()` — window dikategorikan train/test **setelah** windowing, berdasarkan anchor timestamp vs boundary ADR-014
  - `normalize_window()`/`normalize_all_windows()` — per-window z-score, identik untuk window overlap maupun tidak
  - `verify_train_windows_stay_in_train_period()`, `verify_test_windows_anchored_in_test_period()` — V-LEAK-003
  - `count_boundary_overlap_windows()` — **verifikasi LC-4 sebagai fakta yang diharapkan (expected ≤47), bukan bug**
  - `verify_per_window_normalization()` — V-LEAK-004
  - `run_window_generation()` — orkestrator penuh
  - Docstring modul menjelaskan eksplisit alasan non-leakage desain ini (window hanya menengok mundur, tidak pernah maju)

### Bug Teknis Ditemukan & Diperbaiki Selama Verifikasi
Sama seperti M3, proses verifikasi nyata menemukan bug teknis: `.to_numpy()` pada kolom tz-aware pandas 3.x mengembalikan unit yang tidak konsisten (`datetime64[us]` vs `[ns]`), menyebabkan perbandingan timestamp salah 1000x lipat. Diperbaiki dengan `_to_int64_ns()` yang eksplisit memaksa unit nanoseconds sebelum ekstraksi integer — didokumentasikan di docstring method untuk mencegah regresi.

### CLI & Test
- `scripts/run_m6_windows.py` — baca `btc_features_all.parquet` (M4), tulis 8 file window + 2 file timestamp
- `tests/test_window_generation.py` — ~35 test case termasuk **regression test "exactly 47 overlap windows"** dan bug-injection test untuk V-LEAK-003/004

## Verifikasi yang BENAR-BENAR Dijalankan (bukan klaim)

M6 murni `pandas`+`numpy` — verifikasi nyata penuh, **23/23 assertion PASS** di 3 bagian:

✅ **Bagian 1 (5 assertion):** Window extraction dari full matrix — shape, anchor/earliest timestamp tepat
✅ **Bagian 2 (11 assertion) — PALING KRITIS:** Data straddling boundary menghasilkan **PERSIS 47** window overlap (bukan 0, bukan angka lain) — dikonfirmasi log nyata. V-LEAK-003 detector **terbukti mendeteksi** 2 jenis bug yang disuntikkan (train window bocor ke masa depan, test window bocor ke masa lalu)
✅ **Bagian 3 (7 assertion):** **Dataset ukuran asli 35,045 baris** → **N_test_windows=8,760 exact, overlap=47 exact** — dikonfirmasi log nyata untuk 4 timeframe sekaligus

**V-LEAK gate lengkap (V-LEAK-001 s.d. V-LEAK-004) sudah diimplementasikan DAN diverifikasi nyata** di M3/M5/M6 — total akumulasi >90 assertion PASS across seluruh checkpoint leakage sepanjang proyek ini.

## Yang BELUM Bisa Diverifikasi Nyata di Sandbox Ini
⚠️ `pytest` tetap tidak ter-install (tanpa network).
⚠️ `scripts/run_m6_windows.py` belum dieksekusi end-to-end (butuh output M4 asli, yang butuh M1 network).

## Keputusan/Deviasi
**Ini adalah revisi desain paling substansial sejauh ini** — bukan sekadar koreksi teks, tapi perubahan struktur pipeline (M6 dependency berubah dari M5 ke M4). Namun ini tetap **bukan perubahan keputusan riset**: split boundary ADR-014, window size W=48, per-window normalization ADR-016, dan semua invariant DS-03 tidak berubah — hanya urutan/sumber data window generation yang diperbaiki agar konsisten dengan spesifikasi yang sudah ada di DS-03/DS-04. Perubahan didokumentasikan lengkap di changelog DS-02 v1.2 dan IMP-01 v1.3, plus `AUDIT_LC4_ADDENDUM.md`.

## Item Terbuka
1. Commit hash TS2Vec — status tetap "Pending".
2. Jalankan `pytest tests/ -v` di environment ber-network untuk validasi penuh M0-M6.
3. Fork fallback TS2Vec — buat setelah Milestone 1 selesai.
4. Pertimbangkan update manual `.docx` asli DS-04 jika dipakai di luar repo.
5. **(Baru)** Pertimbangkan apakah `.docx` asli DS-03 perlu ditinjau ulang terkait LC-4 juga — meski isi DS-03 sudah konsisten dengan Skenario C (tidak perlu diubah), ada baiknya dikonfirmasi tidak ada bagian lain yang terlewat saat baca cepat di sesi ini.

## TODO List Terbaru

- [x] M0 — Project Bootstrap **[DONE]**
- [x] M1 — Data Acquisition **[DONE]**
- [x] M2 — Data Validation **[DONE]**
- [x] M3 — Temporal Alignment **[DONE]**
- [x] **[AUDIT]** DS-01–DS-04 + IMP-01 logic audit (Bug #1, #2) **[DONE]**
- [x] M4 — Feature Engineering **[DONE]**
- [x] M5 — Temporal Split **[DONE]**
- [x] **[AUDIT]** LC-4 structural audit (Bug #3) **[DONE — Skenario C disepakati]**
- [x] M6 — Window Generation **[DONE — dibangun ulang dengan Skenario C, verifikasi nyata penuh 23/23 PASS]**
- [x] **[MILESTONE 2 — DATA PIPELINE COMPLETE]** ✅
- [ ] M7 — TS2Vec Wrapper ← **lanjut di sini, TAPI terhambat commit hash TS2Vec (status Pending)**
- [ ] M8 — Branch Training
- [ ] M9 — Fusion
- [ ] M10.5 — External Baselines (HMM + KM-PCA) — **bisa dikerjakan sekarang**, hanya depends M5 (sudah selesai), tidak perlu TS2Vec
- [ ] M10 — HDBSCAN Clustering
- [ ] M11 — Evaluation
- [ ] M12 — Visualization (paralel M14)
- [ ] M14 — Statistical Analysis (paralel M12)
- [ ] M13 — Experiment Runner
- [ ] M15 — Paper Artifact Generator

## Instruksi untuk Sesi Berikutnya
**Rekomendasi:** Karena M7 (TS2Vec Wrapper) terhambat status "Pending" commit hash, dan M8/M9 bergantung pada M7, pertimbangkan **melompat ke M10.5 (External Baselines)** dulu — modul ini hanya depends pada M5 yang sudah selesai, sepenuhnya independen dari TS2Vec, dan akan mempercepat progres sambil menunggu Anda meng-clone TS2Vec.
**Alternatif:** Jika Anda sudah siap clone TS2Vec dan isi commit hash, beri tahu saya nilai hash-nya, dan saya bisa lanjut M7 langsung.
**File yang perlu dilihat pertama:** `docs/DS-03_v1.2.md` §4 "External Baselines" (HMM: n_components∈{2,3,4}, BIC; KM-PCA: k∈{2,3,4,5,6}, PCA(10), Silhouette) dan §5 untuk protokol seed, ATAU `docs/IMP-01_v1.3.md` bagian M7 jika TS2Vec sudah siap.
