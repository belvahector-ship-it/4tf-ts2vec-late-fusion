# AUDIT LC-4 — Addendum to AUDIT_REPORT_DS01-DS04_IMP01.md

**Tanggal:** 2026-07-03 (sesi M6)
**Pemicu:** Saat verifikasi nyata M6 (Window Generation), implementasi window generation murni dari `test_features.parquet` (hasil M5) menghasilkan **overlap boundary = 0**, bertentangan dengan klaim "up to 47 test windows contain training-period timestamps" di DS-02/DS-03/DS-04.

---

## Bug #3 (KRITIS, berbeda karakter dari Bug #1/#2) — LC-4 mendeskripsikan dua desain berbeda dalam satu paragraf

### Masalah

DS-02 v1.1 Stage 5 "Leakage Checkpoint 4" (LC-4) berisi klaim yang **saling bertentangan dalam paragraf yang sama**:

> "Windows crossing the train/test boundary do not exist: train windows are generated from `train_features.parquet` **only**, and test windows from `test_features.parquet` **only**. The first window of the test set starts at 2023-01-01 00:00 UTC and its earliest timestamp is 2023-01-01 00:00 − 47h = **2022-12-30 01:00 UTC**. This means the first 47 test windows contain **training-period data**."

Kalimat pertama: window test dihasilkan **HANYA** dari `test_features.parquet` (yang menurut M5 hanya berisi baris `timestamp >= 2023-01-01 00:00`).
Kalimat berikutnya: window test pertama **earliest timestamp-nya `2022-12-30 01:00`** — yaitu **periode training**, yang secara definisi TIDAK ADA di `test_features.parquet`.

**Kedua kalimat ini tidak bisa sama-sama benar.** Jika window test benar-benar hanya dibangun dari `test_features.parquet`, window test pertama TIDAK MUNGKIN memiliki baris dengan timestamp training — datanya memang tidak ada di file itu.

### Bukti Matematis — Dua Skenario, Keduanya Cocok Sebagian

| | **Skenario A**: window murni dari `test_features.parquet` | **Skenario C**: window dari FULL feature matrix (35,045 baris), difilter berdasarkan anchor (baris terakhir) ≥ TEST_START |
|---|---|---|
| **N_test_windows** | `8,760 − 48 + 1 = 8,713` | `8,760` (window count sama dengan jumlah baris test, karena window "menengok mundur" ke training untuk 47 window pertama) |
| **Cocok dengan tabel "Output Tensor Shapes" DS-02 (klaim: N_test_windows ≈ 8,713)?** | ✅ **Cocok persis** | ❌ Tidak cocok (selisih 47) |
| **Overlap (window dengan earliest timestamp < TEST_START)** | **0** | **47** |
| **Cocok dengan klaim LC-4 "up to 47 test windows contain training-period data"?** | ❌ Tidak cocok | ✅ **Cocok persis** |
| **Cocok dengan DS-03 v1.2 §3 CF-002 / §5 Protocol B ("first 47 test windows contain training-period timestamps")?** | ❌ Tidak cocok | ✅ **Cocok persis** |
| **Cocok dengan DS-04 v1.1 V-LEAK-003 ("expected: up to 47", "matches the expected value of at most 47")?** | ❌ Tidak cocok | ✅ **Cocok persis** |

### Kesimpulan Bukti

- **Skenario A** (window murni dari file test yang sudah displit) matematis konsisten dengan **1 klaim** di DS-02 (tabel N_test_windows≈8,713), tapi bertentangan dengan **klaim overlap "up to 47"** yang muncul di **3 dokumen berbeda** (DS-02 LC-4, DS-03 CF-002/Protocol B, DS-04 V-LEAK-003).
- **Skenario C** (window dari feature matrix penuh sebelum split, difilter berdasarkan anchor) matematis konsisten dengan **klaim overlap "up to 47" di 3 dokumen**, tapi bertentangan dengan **1 klaim tabel N_test_windows di DS-02**.

**Bobot bukti mengarah ke Skenario C sebagai desain yang benar-benar dimaksud**, karena:
1. Klaim "up to 47 test windows contain training-period timestamps" bukan muncul sekali (yang bisa jadi typo lokal), melainkan **diulang konsisten di 3 dokumen independen** (DS-02, DS-03, DS-04) dengan detail teknis yang saling mendukung (formula `2023-01-01 00:00 − 47h`, `CF-002` di variable registry, `Protocol B` sebagai mitigasi resmi, `V-LEAK-003` sebagai kriteria pass eksplisit).
2. DS-03 v1.2 bahkan punya **Protocol B** — prosedur robustness-check formal yang secara eksplisit dirancang untuk menangani skenario "47 window pertama mengandung data training" — protokol ini **tidak akan masuk akal untuk didefinisikan** jika overlap-nya memang seharusnya 0.
3. DS-04 v1.1 V-LEAK-003 **secara eksplisit meminta implementasi menghitung dan mendokumentasikan jumlah window overlap, dengan kriteria pass "matches the expected value of at most 47"** — bukan "harus nol". Kriteria pass yang eksplisit mengharapkan angka mendekati 47 adalah sinyal kuat bahwa overlap memang bagian dari desain yang disengaja, bukan sesuatu yang harus dihindari.
4. Nilai tabel "N_test_windows ≈ 8,713" di DS-02 kemungkinan adalah **sisa dari draft awal** (sebelum keputusan window-generation-dari-full-matrix ditetapkan), yang tidak ikut ter-update saat paragraf LC-4 dan dokumen turunannya (DS-03, DS-04) direvisi untuk mendokumentasikan overlap 47 secara eksplisit.

### Dampak

Implementasi M6 awal saya (sudah dihapus, tertunda) menggunakan Skenario A — **ini akan lolos DS-02's tabel row-count check, tapi GAGAL DS-04 v1.1 V-LEAK-003** (yang mengharapkan overlap count terdokumentasi "up to 47", bukan 0), dan tidak konsisten dengan Protocol B yang didefinisikan di DS-03.

### Rekomendasi

Desain M6 yang benar (Skenario C):
1. Window generation membaca dari **feature matrix penuh** (`btc_features_all.parquet`, 35,045 baris, output M4) — **bukan** dari `train_features.parquet`/`test_features.parquet` yang sudah displit di M5.
2. Window dikategorikan sebagai **"train" jika anchor (baris terakhir window) berada di periode train**, dan **"test" jika anchor berada di periode test** — split terjadi **setelah** windowing, berdasarkan anchor timestamp, bukan sebelum windowing berdasarkan baris feature matrix.
3. Ini secara alami menghasilkan **47 window test pertama** yang punya earliest-timestamp masih di periode training (baris-baris itu memang ada di feature matrix penuh, hanya saja secara kalender masuk periode train) — window itu tetap dikategorikan "test" karena anchornya (baris terakhir, dipakai untuk join balik ke evaluasi) ada di periode test.
4. `N_test_windows = 8,760` (bukan 8,713) — **tabel DS-02 perlu dikoreksi** sebagai bagian dari perbaikan dokumen (Bug #3 fix), konsisten dengan bukti matematis di atas.
5. M5 (`temporal_split.py`, sudah selesai) **tetap benar dan tidak perlu diubah** — ia menghasilkan `train_features.parquet`/`test_features.parquet` yang dipakai HANYA sebagai artefak audit/analisis ekonomi (Stage 8/9 economic validity, join balik OHLCV), bukan sebagai sumber langsung window generation.

Keputusan ini butuh persetujuan eksplisit Anda sebelum M6 dibangun ulang, karena ini mengubah cara M6 membaca input (dari M5's split files menjadi M4's full feature matrix) — bukan sekadar koreksi angka seperti Bug #1/#2.
