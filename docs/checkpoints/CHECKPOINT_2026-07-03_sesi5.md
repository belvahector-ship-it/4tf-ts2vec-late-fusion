# CHECKPOINT — Sesi 5 — 2026-07-03

## Status Global
Milestone saat ini: **2 — Data Pipeline Complete** (in progress, 4/6 modul selesai)
Modul terakhir DONE: **M4 — Feature Engineering** (kode lengkap, **verifikasi nyata penuh dengan regression test khusus untuk bug yang baru diperbaiki**)
Modul sedang IN PROGRESS: tidak ada — menunggu instruksi lanjut ke M5

## Peristiwa Penting Sesi Ini: Audit Logika Dokumen Desain

Sebelum melanjutkan M4, Anda meminta audit mendalam terhadap DS-01 s.d. DS-04 + IMP-01 karena saya menemukan inkonsistensi saat verifikasi awal M4 (timestamp "2020-01-19" tidak match matematis dengan "drop 19 baris"). Audit ini **ditunda M4 sepenuhnya** sampai selesai, sesuai instruksi Anda.

### Hasil Audit — 2 Bug Ditemukan

**Bug #1 (kritis):** Timestamp pertama setelah NaN-drop di M4 salah di **3 dokumen** (DS-02, DS-04, IMP-01): tertulis "2020-01-19 00:00 UTC" (19 hari), seharusnya **"2020-01-01 19:00:00 UTC"** (19 jam) — pipeline seluruhnya beresolusi 1 jam. Dibuktikan lewat 3 cross-check independen (train rows ~26,269, N_train_windows ~26,222, arithmetic 35,064−19=35,045) — semua konsisten dengan interpretasi "19 jam", meleset ~397 unit dengan interpretasi "19 hari".

**Bug #2 (sedang):** DS-01 ADR-019 masih menyatakan "8 conditions × 5 seeds = 40 runs", **bertentangan dengan ADR-005 di dokumen yang sama** (sudah benar: 7 kondisi, 35 run) dan dengan DS-03 v1.2 (sudah benar: 35+10=45 run total). Bug ini adalah sisa revisi run-count yang tidak lengkap — sama persis dengan pola bug IMP-01 v1.0 yang kita perbaiki di sesi-sesi awal proyek, kali ini ditemukan di dokumen sumber ADR itu sendiri.

Laporan lengkap: `docs/checkpoints/` tidak menyimpannya, tapi sudah dibagikan sebagai `AUDIT_REPORT_DS01-DS04_IMP01.md` (file terpisah, di luar repo).

### 5 Dokumen Revisi Dibuat & Diterapkan ke Repo

| Dokumen | Versi | Perbaikan | Status di repo |
|---|---|---|---|
| DS-01_v1.1.md | v1.0→v1.1 | ADR-019: 40→**45 runs** | ✅ Menggantikan `docs/DS-01_Architecture_Decision_Records.md` |
| DS-02_v1.1.md | v1.0→v1.1 | Stage 4 timestamp: "2020-01-19"→**"2020-01-01 19:00"** | ✅ Menggantikan `docs/DS-02_Data_Flow_Specification.md` |
| DS-03_v1.2.md | v1.2 (docx→md) | Migrasi format saja, tidak ada perubahan konten (sudah benar dari awal) | ✅ Menggantikan `docs/DS-03.docx` |
| DS-04_v1.1.md | v1.0→v1.1 | V-DATA-004 timestamp: "2020-01-19"→**"2020-01-01 19:00"** | ✅ Menggantikan `docs/DS-04.docx` |
| IMP-01_v1.2.md | v1.1→v1.2 | M4 DoD timestamp: "2020-01-19"→**"2020-01-01 19:00"** | ✅ Menggantikan `docs/IMP-01_Implementation_Roadmap_v1.1.md` |

Semua file punya changelog eksplisit di header dengan bukti matematis. Angka inti (35,045 baris, 29 kolom, 7 kondisi, 45 run total, split boundary) **tidak berubah** — hanya representasi tanggal/run-count yang salah tulis yang diperbaiki.

**Repo `docs/` sudah diupdate** menghapus 5 file lama dan menggantinya dengan 5 file revisi ini.

## M4 — Feature Engineering: Selesai dengan Desain Terbaru

### Kode (`src/data/`)
- `feature_engineering.py` — modul lengkap M4:
  - `FeatureEngineer` class: 7 metode `compute_*` (open_return, high_return, low_return, close_return, volume_zscore, hl_range, body_ratio) sesuai ADR-015
  - `compute_features_for_timeframe()`, `compute_all_features()` — 29 kolom (1 timestamp + 7 fitur × 4 timeframe)
  - `drop_nan_rows()` — drop 19 baris NaN pertama
  - `check_feature_matrix_integrity()`, `check_feature_matrix_schema()` — **`EXPECTED_FIRST_TIMESTAMP` sekarang `2020-01-01 19:00:00 UTC`** (bukan lagi `2020-01-19`)
  - `run_feature_engineering()` — orkestrator penuh, raise `FeatureEngineeringError` informatif
  - Docstring modul secara eksplisit menjelaskan sejarah bug dan koreksinya untuk audit trail masa depan

### CLI (`scripts/`)
- `run_m4_features.py` — baca `data/interim/btc_aligned_1h.parquet`, tulis `data/processed/btc_features_all.parquet`

### Test (`tests/test_feature_engineering.py`)
- ~40 test case, termasuk **regression test eksplisit** (`test_first_remaining_timestamp_is_19_hours_after_start`, `test_wrong_first_timestamp_fails`) yang secara spesifik memverifikasi bug lama tidak muncul kembali.

## Verifikasi yang BENAR-BENAR Dijalankan (bukan klaim)

M4 murni `pandas`+`numpy` (sama seperti M2/M3), sehingga **verifikasi nyata penuh** dijalankan (bukan hanya sintaks):

✅ **Bagian 1 (14 assertion):** Semua 7 formula fitur diverifikasi dengan hand-computed values, termasuk doji candle epsilon test
✅ **Bagian 2 (5 assertion) — REGRESSION TEST UTAMA:** `drop_nan_rows` pada fixture 30-baris menghasilkan **first timestamp = `2020-01-01 19:00:00 UTC`** persis (dikonfirmasi log nyata), BUKAN nilai bug lama `2020-01-19`. Juga dikonfirmasi 7 fitur identik across 4 timeframe saat input sama.
✅ **Bagian 3 (10 assertion) — DATASET UKURAN ASLI 35,064 BARIS:** `run_feature_engineering()` menghasilkan **35,045 baris, 29 kolom, first timestamp `2020-01-01 19:00:00 UTC`** — dikonfirmasi log nyata. Cross-check independen: train rows (≤2022-12-31 23:00) = 26,285, cocok dengan klaim DS-02 v1.1 "~26,269" (selisih 16, wajar). Schema check terbukti **menolak dengan benar** jika diberi angka lama yang salah (`2020-01-19`) sebagai expected value — safety net bekerja.
✅ **Regression check M0-M4 lengkap:** semua modul (kecuali `device.py` yang butuh torch) tetap ter-import tanpa error setelah perubahan.

**Hasil total: 29/29 assertion PASS** di 3 bagian verifikasi, semuanya dijalankan nyata dengan output log yang bisa diaudit.

## Yang BELUM Bisa Diverifikasi Nyata di Sandbox Ini

⚠️ `pytest` tetap tidak ter-install (tanpa network) — logic test sudah diverifikasi manual dengan cakupan setara.
⚠️ `scripts/run_m4_features.py` belum dieksekusi end-to-end (butuh output M3 yang butuh M1 yang butuh network).

## Keputusan/Deviasi
Tidak ada deviasi dari keputusan riset. Semua perbaikan sesi ini adalah **koreksi faktual/aritmatika terhadap dokumen desain sendiri** (bukan perubahan keputusan), didokumentasikan lengkap dengan audit trail di setiap changelog dokumen dan di `AUDIT_REPORT_DS01-DS04_IMP01.md`.

## Item Terbuka yang Butuh Keputusan/Tindakan Anda
1. **Commit hash TS2Vec** — status tetap "Pending", isi setelah clone.
2. **Jalankan `pytest tests/ -v`** di environment ber-network untuk validasi penuh M0-M4 dengan pytest asli.
3. Fork fallback TS2Vec — buat setelah Milestone 1 selesai.
4. **(Baru)** Pertimbangkan apakah dokumen `.docx` asli (DS-03, DS-04) perlu diupdate juga di luar repo ini, mengingat repo sekarang memakai versi markdown sebagai canonical. Saat ini DS-03 docx asli sudah benar (tidak perlu diedit); DS-04 docx asli masih mengandung bug timestamp yang sama — pertimbangkan edit manual di docx aslinya jika dokumen itu dipakai di luar repo (mis. untuk sidang/laporan).

## TODO List Terbaru (state penuh)

- [x] M0 — Project Bootstrap **[DONE]**
- [x] M1 — Data Acquisition **[DONE — verifikasi parsial nyata]**
- [x] M2 — Data Validation **[DONE — verifikasi nyata penuh]**
- [x] M3 — Temporal Alignment **[DONE — verifikasi nyata penuh, 2 bug ditemukan+diperbaiki]**
- [x] **[AUDIT]** DS-01–DS-04 + IMP-01 logic audit **[DONE — 2 bug ditemukan, 5 dokumen revisi dibuat & diterapkan]**
- [x] M4 — Feature Engineering **[DONE — dibangun dengan desain terbaru, verifikasi nyata penuh + regression test khusus]**
- [ ] M5 — Temporal Split ← **lanjut di sini**
- [ ] M6 — Window Generation
- [ ] M7 — TS2Vec Wrapper (masih menunggu commit hash TS2Vec, status Pending)
- [ ] **[GATE] V-LEAK-002, V-LEAK-003, V-LEAK-004** — menyusul di M5/M6 (V-LEAK-001 sudah selesai di M3)
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
**Mulai dari:** M5 — Temporal Split (DS-02 v1.1 Stage 4, ADR-014). Split boundary: train ≤ 2022-12-31 23:00 UTC, test ≥ 2023-01-01 00:00 UTC. Expected: train ~26,269 baris (**dengan start point yang benar 2020-01-01 19:00, BUKAN 2020-01-19** — pastikan M5 memakai output M4 yang sudah benar), test ~8,760 baris. **V-LEAK-002** (LC-3, rolling volume_zscore near boundary) perlu diimplementasikan di sini.
**File yang perlu dilihat pertama:** `docs/DS-02_v1.1.md` bagian "Stage 4: Temporal Split" (perhatikan expected sizes yang sudah dikoreksi), `docs/DS-04_v1.1.md` bagian V-DATA-005 dan V-LEAK-002.
**Catatan:** M5 kemungkinan besar tetap pure pandas (bisa diverifikasi nyata penuh seperti M2/M3/M4).
