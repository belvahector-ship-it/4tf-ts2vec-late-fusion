# CHECKPOINT — Sesi 8 — 2026-07-03

## Status Global
Milestone saat ini: **Milestone 2 — Data Pipeline Complete (tercapai sesi 7)**
Sesi ini: **TS2Vec commit hash di-pin, persiapan migrasi penuh ke Claude Code**
Modul sedang IN PROGRESS: tidak ada — repo siap dilanjutkan di environment baru

## Peristiwa Penting Sesi Ini

### 1. Analisis Source Code TS2Vec
User mengupload `ts2vec-main.zip` (download langsung dari GitHub, tanpa metadata `.git`). Saya baca lengkap: `ts2vec.py`, `models/encoder.py`, `models/dilated_conv.py`, `models/losses.py`, `utils.py`, `README.md`, `requirements.txt`. Temuan kunci untuk M7:
- `TS2Vec.encode()` default mengembalikan representasi per-timestamp `[N,T,output_dims]` — **wajib `encoding_window='full_series'`** untuk dapat 1 vector per window sesuai DS-03
- `save()`/`load()` beroperasi pada `self.net` (AveragedModel/EMA wrapper), bukan `self._net`
- Dependency inti TS2Vec (`ts2vec.py`+`models/`) hanya butuh `torch`+`numpy` — requirement resmi (`scipy`/`pandas`/`sklearn`/dst) cuma dipakai benchmark harness mereka yang tidak kita pakai
- TS2Vec resmi minta `torch==1.8.1`, proyek kita pakai `torch==2.3.1` — API yang dipakai (`Conv1d`, `AdamW`, `AveragedModel`, `DataLoader`) stabil lintas versi, tapi **ini harus diverifikasi nyata di M7**, bukan diasumsikan

### 2. Commit Hash TS2Vec Dikonfirmasi & Di-pin
User memberi hash `b0088e14a99706c05451316dc6db8d3da9351163`. Saya cross-check via web_fetch ke `github.com/zhihanyue/ts2vec/blob/main/ts2vec.py`: HEAD `main` saat ini adalah commit **berbeda** (`ac76ac278b564e81010cf07d14d4109e9d202ead`), tapi **isi file `ts2vec.py` identik persis** dengan yang ada di zip user. Karena user yang punya konteks asal hash tersebut, dan tidak ada perbedaan kode fungsional yang terdeteksi, hash dari user dipakai sebagai pin resmi. **Status berubah dari "Pending" → "Pinned"** di 4 tempat: `configs/base.yaml`, `requirements.txt`, `environment.yml`, `README.md` — semua dengan catatan transparan soal perbedaan dengan HEAD saat ini.

### 3. Vendoring Source TS2Vec sebagai Referensi (bukan runtime dependency)
Disalin ke `third_party_reference/ts2vec/` (hanya `ts2vec.py`, `models/`, `utils.py`, `README.md`, `LICENSE` — file benchmark harness `datasets/`, `scripts/`, `tasks/`, `train.py`, `datautils.py` DIBUANG karena tidak relevan). Tujuan: (a) referensi baca untuk menulis `ts2vec_wrapper.py` di M7, (b) fallback dokumentasi kalau `pip install` dari commit terpin gagal di masa depan. **Bukan untuk di-import** — runtime tetap pakai `pip install git+...`.

### 4. Migrasi ke Claude Code — Alasan dan Persiapan
User akan lanjutkan proyek di Claude Code (bukan lagi chat Claude.ai) karena sandbox chat ini **tanpa network, tanpa torch/pytest/ccxt** — sudah berkali-kali jadi hambatan (M1 tidak pernah bisa dijalankan nyata, test suite M0-M6 hanya diverifikasi via simulasi manual Python, bukan `pytest` sungguhan). Disiapkan 3 artefak migrasi baru di root repo:
- **`MIGRATION_TO_CLAUDE_CODE.md`** — panduan wajib dibaca duluan oleh AI yang lanjutkan sesi: ringkasan status M0-M6, tabel status verifikasi per modul, API TS2Vec yang sudah dikonfirmasi, protokol kerja yang harus dilanjutkan (TODO list, tanya-dulu-sebelum-asumsi untuk inkonsistensi dokumen, update checkpoint tiap sesi)
- **`setup_and_verify.sh`** — skrip bash otomatis: install `requirements.txt` → verifikasi semua import kunci (`torch`, `pandas`, `ts2vec`, `ccxt`, `hdbscan`, dst) → jalankan `pytest tests/ -v` sebagai baseline, dengan pesan eksplisit "jangan buru-buru fix test, cek dulu apakah bug asli atau beda environment" (mengacu ke 2 kasus bug pandas 3.x yang sudah terjadi)
- **`PUSH_TO_GITHUB.md`** — langkah git init/add/commit/push step-by-step, plus alternatif GitHub Desktop untuk yang tidak familiar command line, plus prompt pembuka siap-pakai untuk Claude Code

## Verifikasi Sesi Ini
✅ `configs/base.yaml` tetap valid setelah update commit hash (dicek ulang via `load_config`)
✅ Isi `ts2vec.py` di zip user vs GitHub `main` saat ini: **dikonfirmasi identik** (dibaca penuh keduanya, dibandingkan)
✅ Sintaks `setup_and_verify.sh` valid (`bash -n`)

## Keputusan/Deviasi
- Memakai commit hash dari user (bukan HEAD GitHub saat ini) — **bukan deviasi**, ini justru inti dari "pinning": commit spesifik yang user verifikasi, bukan HEAD yang bisa berubah kapan saja.
- Vendoring source TS2Vec ke `third_party_reference/` — deviasi kecil dari ADR-001 murni ("pip install dari commit terpin saja"), ditambahkan sebagai lapisan keamanan ekstra karena reproducibility commit hash yang tidak match HEAD berisiko sedikit lebih tinggi dari biasanya. Didokumentasikan jelas bukan untuk runtime use.

---

## Sesi 9 (Claude Code) — 2026-07-03 — Migrasi environment & baseline verification

### Kondisi environment yang ditemukan (berbeda dari asumsi migrasi)
Mesin Claude Code ini **hanya punya Python 3.14.3** (via `py`/pythoncore-3.14), tanpa conda. Ini **tidak kompatibel** dengan stack ter-pin proyek (`environment.yml` pin `python=3.11`):
- `torch==2.3.1`, `numpy==1.26.4`, `pandas==2.2.2` **tidak punya wheel untuk Python 3.14** (torch untuk 3.14 baru mulai dari 2.9.0). Interpreter 3.14 global bahkan sudah membawa `numpy 2.4.4` + `pandas 3.0.2` — pandas 3.x adalah persis landmine `datetime64` yang sudah 2x menggigit proyek ini.
- Baris `git+https://github.com/zhihanyue/ts2vec.git@<commit>#egg=ts2vec` di `requirements.txt`/`environment.yml` **tidak bisa di-pip-install sama sekali** (independen dari versi Python): repo di commit terpin tidak punya `setup.py`/`pyproject.toml` → error "does not appear to be a Python project". Inilah alasan struktural kenapa source di-vendor ke `third_party_reference/`.

### Yang dilakukan (dengan persetujuan user)
1. **Python 3.11 terisolasi**: install `uv` (via pip user), lalu `uv python install 3.11` + `uv venv --python 3.11 .venv` → `.venv` berisi **CPython 3.11.15** standalone. Tidak menyentuh Python sistem.
2. **Install pinned deps ke `.venv`** (torch 2.3.1+cpu, numpy 1.26.4, pandas 2.2.2, scipy 1.13.1, sklearn 1.5.0, pyyaml, ccxt 4.3.41, pyarrow 16.1.0, statsmodels 0.14.2, matplotlib 3.9.0, umap-learn 0.5.6, pytest 8.2.2, pytest-cov). **Semua versi ter-pin persis** → kontrak reproducibility M0 terjaga.
3. **`hdbscan==0.8.38.post1` (M10) dan `hmmlearn==0.3.2` (M10.5) DITUNDA**: keduanya compile-from-source dan butuh MSVC C++ Build Tools yang belum ada. Tidak diperlukan untuk baseline M0-M6 maupun M7. → **item terbuka**, harus diselesaikan sebelum M10/M10.5.
4. **ts2vec via vendored copy**: dibuat `.venv/Lib/site-packages/ts2vec_vendored.pth` berisi path absolut `third_party_reference/ts2vec` → `from ts2vec import TS2Vec` importable dari copy ter-vendor. `requirements.txt` git+ line tidak dipakai untuk install.

### Keputusan/Deviasi (Sesi 9)
- **[DEVIASI, disetujui user]** ts2vec sekarang **di-import dari `third_party_reference/ts2vec/`** (via `.pth`), bukan dari paket pip. Ini **merevisi** instruksi "JANGAN import dari sini" di `MIGRATION_TO_CLAUDE_CODE.md` (baris ~94) dan `third_party_reference/ts2vec/VENDORED_README.md`. Alasan: paket di commit terpin tidak pip-installable (tidak ada setup.py/pyproject), jadi "install via pip dari commit terpin" (ADR-001) tidak mungkin dijalankan apa adanya. Vendored copy sudah dikonfirmasi identik dengan source terpin (lihat Sesi 8). **TODO dokumentasi**: perbarui teks di MIGRATION doc + VENDORED_README + ADR-001 agar konsisten dengan keputusan ini; M7 `ts2vec_wrapper.py` tetap `from ts2vec import TS2Vec` (kini di-resolve ke vendored via `.pth`).
- **[DEVIASI environment, bukan research]** environment eksekusi = venv Python 3.11.15 buatan `uv`, bukan conda `market-state-discovery`. Versi paket ter-pin identik dengan `environment.yml` untuk paket yang terpasang; hanya mekanisme provisioning yang beda.

- **[KEPUTUSAN M7 — optimizer TS2Vec, FINAL, disetujui user]** Kita PILIH opsi (b): proceed dengan default TS2Vec (`weight_decay=0.01`, weight-level resume), TIDAK melakukan monkey-patch/subclass ke library TS2Vec. Formalisasi: **ADR-021** ([docs/ADR-021_ts2vec_optimizer_defaults.md](ADR-021_ts2vec_optimizer_defaults.md)) — dibuat sebagai addendum standalone (pola sama seperti `AUDIT_LC4_ADDENDUM.md`) karena DS-01 masih di-pin v1.1 oleh `MIGRATION_TO_CLAUDE_CODE.md`; ADR-021 harus dimasukkan ke ADR Index DS-01 pada version bump DS-01 berikutnya. Alasan lengkap (verbatim dari keputusan user):

  1. weight_decay adalah parameter INTERNAL milik TS2Vec (pihak ketiga, kode vendored/pinned di `third_party_reference/ts2vec/`), bukan bagian dari kontribusi/novelty riset ini. Kontribusi utama riset ada di Cross-Timeframe Attention dan Late Fusion architecture di atas TS2Vec, bukan di tuning internal TS2Vec itu sendiri.

  2. Filosofi proyek ini adalah NETRAL terhadap implementasi upstream TS2Vec — kita memakai TS2Vec apa adanya sesuai commit yang di-pin (`b0088e14a99706c05451316dc6db8d3da9351163`), tanpa mengedit atau membungkus ulang optimizernya. Ini konsisten dengan ADR-001 (pip install dari commit terpin, vendoring hanya untuk referensi baca) — memonkey-patch optimizer akan melanggar semangat ADR-001 dan menambah kompleksitas + risiko baru ke kode pihak ketiga yang sudah terverifikasi jalan (269 test passed).

  3. `base.yaml training.weight_decay: 0.0001` TIDAK dihapus dari config — dibiarkan tetap ada sebagai dokumentasi nilai yang DIINGINKAN secara desain, tapi wrapper mencatat secara eksplisit (sudah diimplementasikan) bahwa nilai ini tidak benar-benar digunakan oleh `TS2Vec.fit()`, digantikan oleh default library 0.01.

  4. Ini akan didokumentasikan secara jujur sebagai keterbatasan (limitation) di bagian metodologi paper nanti: bahwa optimizer internal TS2Vec menggunakan default library (wd=0.01), tidak di-override, untuk menjaga kompatibilitas penuh dengan implementasi upstream yang sudah divalidasi.

  5. Konsekuensi terkait: `optimizer_state_dict` tetap `None` di checkpoint (root cause sama — optimizer tidak exposed oleh TS2Vec), weight-level resume (bukan true optimizer-state resume) diterima sebagai batasan yang sama, dengan alasan yang sama (netral terhadap upstream).

  Status: **final** — tidak ditanyakan ulang kecuali ada temuan baru yang mengharuskan revisi. M8 dibangun di atas asumsi ini.

### Baseline pytest (pertama kali dijalankan dengan pytest sungguhan)
`python -m pytest tests/ -v` di `.venv` → **250 passed, 1 failed** (6.74s).
- ✅ Semua V-LEAK gate lulus: `test_alignment.py`, `test_temporal_split.py` (kecuali 1 di bawah), `test_window_generation.py` — termasuk `test_full_size_dataset_matches_ds02_v1_2_exact_numbers` (Skenario C / N_test_windows=8,760) dan test bug-injection overlap-47.
- ❌ **1 gagal: `tests/test_temporal_split.py::TestRunTemporalSplit::test_no_shuffling_train_is_sorted`**.
  - **Diagnosis: bug desain TEST, BUKAN bug `src/` dan BUKAN perbedaan versi environment.** Test membuat fixture 1000 baris hourly mulai `2020-01-01 19:00` (semua < 2023), lalu memanggil `run_temporal_split(..., expected_train_rows=None, expected_test_rows=None)`. Tapi `run_temporal_split` **selalu** menjalankan `check_split_boundary` (hard-assert train max == 2022-12-31 23:00 & test non-empty) — check ini tidak bisa dimatikan lewat `expected_*_rows`. Karena data sintetis tak pernah masuk 2023, test set kosong → boundary check gagal.
  - `src/data/temporal_split.py` **berperilaku benar** (menegakkan boundary ADR-014); data produksi 35,045 baris (2020–2023) memenuhinya. pandas 2.2.2 ter-pin; tidak ada keterlibatan datetime64 unit.
  - Test ini **tak pernah benar-benar dijalankan** sebelumnya (M5 "diverifikasi" via simulasi manual, bukan pytest) — persis skenario yang diwanti-wanti MIGRATION doc.
  - **Fix diterapkan (disetujui user):** fixture diganti ke data menyeberang boundary (`2022-12-01 00:00`, 1000 periods → span ke 2023), meniru sibling test `test_does_not_raise_when_size_check_disabled`. Intent test tak berubah (cek `train["timestamp"].is_monotonic_increasing`). Hanya file test yang diubah; `src/` tidak disentuh.
  - **Hasil setelah fix: `251 passed` (0 gagal).** Baseline M0-M6 kini hijau penuh via pytest sungguhan untuk pertama kalinya.

### Status untuk sesi berikutnya
- Baseline hijau → **M7 (TS2Vec Wrapper) siap dimulai** sesuai `docs/IMP-01_v1.3.md`. `from ts2vec import TS2Vec` sudah importable (vendored via `.pth`); torch 2.3.1+cpu terpasang — kompatibilitas API TS2Vec dengan torch 2.3.1 masih **wajib diverifikasi nyata saat M7** (train/encode sungguhan), belum dilakukan.
- **Item terbuka**: (a) install `hdbscan`+`hmmlearn` (butuh MSVC Build Tools) sebelum M10/M10.5; (b) sinkronkan teks MIGRATION doc / VENDORED_README / ADR-001 dengan keputusan import-vendored.

### M7 — TS2Vec Wrapper (SELESAI sesi ini, di Claude Code)

**Gate wajib TERPENUHI:** kompatibilitas TS2Vec (vendored, terpin) dengan `torch 2.3.1+cpu` **diverifikasi nyata** (bukan diasumsikan) via smoke test: `fit` jalan, `encode(encoding_window='full_series')` mengembalikan `[N,64]`, encode deterministik, dan `save`/`load` round-trip mereproduksi output. Tidak ada inkompatibilitas API.

**Kode:** [`src/models/ts2vec_wrapper.py`](../src/models/ts2vec_wrapper.py) — `TS2VecBranch` (`__init__`/`train`/`encode`/`save_checkpoint`/`load_checkpoint`) + `TrainingHistory`. Import `from ts2vec import TS2Vec` (resolve ke vendored via `.pth`); tidak ada source TS2Vec di `src/`. Checkpoint mengikuti bundel ADR-010 (semua field ada). `encode` pakai `full_series` max-pool → `[N,64]` (V-MODEL-001). Early stopping (patience) diimplementasikan di wrapper via `after_epoch_callback` yang meng-raise sinyal internal — tanpa mengubah source TS2Vec dan tanpa me-reset optimizer.

**Test:** [`tests/test_ts2vec_wrapper.py`](../tests/test_ts2vec_wrapper.py) — 18 test. Full suite: **269 passed** (251 baseline + 18 M7). Menutup V-MODEL-001 (shape [N,64]), V-INV-004 (tiap branch punya model/optimizer independen, tak ada shared loss/weights), kelengkapan bundel ADR-010, load GPU→CPU (`map_location`), dan determinisme encode.

**Deviasi/friksi M7 yang didokumentasikan (BUKAN diam-diam di-fix dengan mengedit upstream):**
1. **`weight_decay` config tidak diterapkan.** TS2Vec membangun `AdamW` di dalam `fit()` sebagai variabel lokal dengan `weight_decay` default library (**0.01**); `base.yaml training.weight_decay` (**1e-4**) tidak dibaca upstream dan optimizer tak ter-ekspos. Wrapper mencatat `config_weight_decay` vs `effective_weight_decay` di metadata checkpoint + WARNING di log. **Perlu keputusan user** (relevan untuk M8): terima default 0.01, atau honor 1e-4 via monkey-patch/subclass + ADR baru (konsekuensi ADR-001).
2. **Tidak ada `optimizer_state_dict` untuk resume optimizer sejati** (alasan sama: optimizer internal ke `fit()`). Field ADR-010 `optimizer_state_dict` ditulis `None`; resume level-bobot (re-`fit` dari weights tersimpan) tetap mungkin. Relevan untuk M8 (`load_or_train` resume).
3. **`projection_matrix`/`condition` di checkpoint branch = `None`** by default — keduanya konsep kondisi/fusi (M9), sedangkan branch bersifat condition-agnostic (ADR-002). Diisi caller via `extra_metadata` saat M9.

**Status M7:** kode + test selesai & hijau. Yang belum: training GPU sungguhan (M8), dan keputusan user atas friksi #1/#2 di atas.

### M8 — Branch Training (KODE+TEST SELESAI sesi ini; eksekusi 20-run menunggu data)

**Keputusan optimizer TS2Vec:** difinalkan sebagai **opsi (b)** via **ADR-021** (lihat Keputusan/Deviasi Sesi 9). M8 dibangun di atas asumsi ini.

**Ekstensi M7 wrapper (untuk ADR-010 best vs latest):** `TS2VecBranch` kini menyimpan snapshot bobot epoch terbaik (CPU copy di `_on_epoch_end` saat loss terbaik baru) dan `save_checkpoint(..., which="best"|"latest")`. Ini kode kita sendiri, bukan edit TS2Vec — konsisten ADR-021. +5 test.

**Kode:** [`src/models/branch_training.py`](../src/models/branch_training.py) — `BranchTrainer` (`train_single`/`load_or_train`/`train_all_branches` + helper path/loader/validitas checkpoint) dan `TrainingOrchestrator` (`run_all` dengan resume run-level + lanjut-melewati-kegagalan). Plus CLI [`scripts/run_m8_training.py`](../scripts/run_m8_training.py).
- **4 timeframe × 5 seed = 20 run** (bukan per kondisi — ADR-002). Checkpoint per (tf,seed) di `checkpoints/branch_{tf}/seed_{seed}/{best,latest}_model.pt`; log per-run `logs/training_branch_{tf}_seed_{seed}.log` (detail per epoch).
- **Idempoten/resume:** `load_or_train` melewati run yang sudah punya `best_model.pt` valid. **Catatan (ADR-021):** TS2Vec tak mengekspos optimizer → tidak ada true optimizer-state resume; resume di sini = skip run yang sudah selesai (granularitas run), `latest_model.pt` untuk weight-level continue.
- **Reuse antar-kondisi:** ada tepat satu checkpoint per (tf,seed); setiap kondisi yang memakai tf itu me-load file yang sama (structural reuse).

**Test:** [`tests/test_branch_training.py`](../tests/test_branch_training.py) — 13 test (windows sintetis, 1 epoch, tmp dir). Full suite: **287 passed**. Menutup: 4-run-per-seed, best+latest+log tertulis, idempotency `load_or_train`, resume+continue-on-failure (orchestrator), V-EXP-002 (seed beda → embedding beda), V-MODEL-005 (branch beda → representasi beda), V-INV-004.

**BELUM dieksekusi (prasyarat data):** 20 run training SUNGGUHAN belum jalan karena file window M6 (`data/processed/train_windows_{tf}.npy`) **tidak ada di environment ini** — butuh menjalankan pipeline M1→M6 dulu (M1 = download Binance 4 tahun, butuh network+waktu), lalu `python scripts/run_m8_training.py`. Ini langkah eksekusi berat, terpisah dari implementasi M8 yang sudah selesai & teruji. **Ini bukan blocker kode; ini prasyarat data untuk menghasilkan 20 checkpoint nyata.**

### Eksekusi pipeline dengan DATA REAL (Sesi 9 lanjutan) — M1 dijalankan pertama kali dengan network sungguhan

Konteks: user memilih menjalankan pipeline M1→M6 dengan data Binance nyata (pertama kali; sebelumnya M1 hanya diverifikasi via mock di sandbox tanpa network). Simbol dikonfirmasi tetap **BTC/USDT** sesuai `base.yaml` (sebutan "DASH/USDT + BTC/USD" di satu pesan user adalah salah tulis, dikonfirmasi bukan perubahan desain).

Estimasi M1 (dilaporkan sebelum download, sesuai protokol): tanpa API key (ccxt `binance()` public `fetch_ohlcv`), ~188 request paging @1000 candle, ~1–3 menit, ~5–15 MB. Probe konektivitas (1 request) sukses — Binance reachable (tidak geo-block), format response cocok dgn kode.

#### 🐞 BUG NYATA M1 ditemukan (root cause + fix + regression test)

- **Kapan ditemukan:** saat M1 dijalankan PERTAMA KALI dengan data real (2026-07-03, sesi 9). Tidak pernah terdeteksi selama seluruh verifikasi mock di sandbox lama.
- **Root cause:** `BinanceDownloader.download()` menghitung `end_ms` tapi **hanya memakainya sebagai guard loop** `while cursor < end_ms`. Setiap iterasi meng-`extend` satu halaman penuh (≤1000 candle) dan hanya *cursor* yang dibatasi; candle di halaman terakhir yang melewati study-end **tidak pernah di-trim**. Karena Binance menyajikan data sampai sekarang, tiap timeframe overshoot: 15m→2024-01-10, 1h→2024-02-10, 4h→2024-02-09, **1d→2025-06-22**. Row count jadi kelipatan 1000 (141000/36000/9000/2000).
- **Dampak:** raw files berisi baris di luar periode studi (masuk 2024–2025). 1d GAGAL V-DATA-001 (2000 vs 1461, +37%). Lebih parah: M3 alignment mengharapkan tepat 35,064 baris (2020–2023); baris ekstra akan merusak matriks aligned, boundary split 2023-12-31, dan **semua ekspektasi V-LEAK** downstream.
- **Kenapa mock melewatkannya:** test M1 memakai mock exchange yang mengembalikan candle hanya dalam range (terlalu "bersih") — trim-end tidak pernah teruji. **Ini contoh nyata kenapa protokol "verifikasi data real sebelum lanjut" penting.**
- **Fix (disetujui user, opsi 1):** satu blok trim di `download()` — filter `df` ke `[start_ts, end_ts]` inklusif (memakai `since`/`end_ms` yang sudah dihitung). Ini fix engineering agar realita cocok dengan semantik `[start, end]` yang sudah didokumentasikan — BUKAN perubahan desain.
- **Regression test ditambahkan** (`tests/test_acquisition.py`): `test_download_trims_candles_past_end_date` (mock 40 candle harian overshoot → di-trim ke 31) + `test_download_trims_candles_before_start_date`. M1 test suite: **31 passed**.
- **Verifikasi angka setelah trim (dari file real):** 15m=140,103 (exp 140,256), 1h=35,032 (exp 35,064), 4h=8,765 (exp 8,766), 1d=1,461 (exp 1,461, PERSIS). Kekurangan kecil di 15m/1h/4h = **gap data Binance asli** (maintenance/outage 4 tahun), 0 duplikat, **dalam toleransi 5% V-DATA-001** — bukan bug, tekstur data real yang tak ada di synthetic.
- **Durasi M1 aktual:** download 4 timeframe ~1 menit (18:51→18:52), first request cold ~13s. Untuk referensi sesi berikutnya.

#### 5 TEMUAN DATA REAL (M1–M4) saat verifikasi pipeline M1→M6 (semua di-fix dengan persetujuan user, semua ada regression test)

Menjalankan pipeline pertama kali dengan data Binance nyata mengungkap 5 bug/konflik yang TIDAK PERNAH terdeteksi selama verifikasi mock di sandbox lama (mock data terlalu "bersih"). Ini bukti konkret kenapa protokol "verifikasi data real sebelum lanjut" krusial.

| # | Modul | Temuan | Root cause | Fix (disetujui user) | Dampak angka |
|---|---|---|---|---|---|
| 1 | M1 | `download()` overshoot ke 2024–2025 | Tidak trim ke end date (hanya cursor yang dibatasi) | Trim `df` ke `[since, end_ms]` inklusif | 1d 2000→1461; semua TF berhenti tepat 2023-12-31 |
| 2 | M2 | `no_excessive_single_gap` hard-fail (15m/1h) | Outage Binance multi-jam (2020-02-19 6h, dst) — kondisi pasar legit | **ADR-022**: downgrade ke WARNING; aggregate 5% tetap hard gate | M2 PASS (warning tercatat) |
| 3 | M2 | `check_date_coverage` gagal 4h/1d | Hardcode akhir 23:00 (asumsi hourly) | Timeframe-aware: `(end+1d)-durasi` (4h→20:00, 1d→00:00) | 4h/1d coverage PASS |
| 4 | M3 | master 35,032 (bukan 35,064) | Anchor pada 1h candle aktual, bukan grid penuh | **ADR-023**: reindex ke grid 1-jam penuh; harga LOCF (no look-ahead), volume=0 saat outage | master 35,064 EXACT, grid gap-free |
| 5 | M4 | drop 7,324 baris (bukan 19) | `volume_zscore` tanpa epsilon → std=0 pada 1d konstan-per-hari → NaN | Tambah epsilon `+1e-8` (pola sama `body_ratio`) | 35,045 EXACT dipulihkan |

Catatan: temuan #1 (M1) sudah diuraikan detail di atas; #4 dan #5 punya ADR/section sendiri. Dokumen desain diamandemen: DS-04 V-DATA-001 (temuan #2, #3).

**Catatan terpisah — ADR-021 (bukan temuan data real):** friksi optimizer TS2Vec (weight_decay tidak diterapkan; tak ada optimizer_state_dict) adalah keputusan desain M7, BUKAN bug data-real — sudah dicatat lengkap di bagian "Keputusan/Deviasi (Sesi 9)" dan `docs/ADR-021_ts2vec_optimizer_defaults.md`. Dicantumkan di sini hanya sebagai silang-rujuk.

**ADR baru sesi ini** (ketiganya standalone addendum, WAJIB dilipat ke ADR Index DS-01 pada version bump berikutnya): **ADR-021** (optimizer TS2Vec — M7), **ADR-022** (single-gap WARNING — temuan #2), **ADR-023** (M3 reindex grid penuh — temuan #4).

#### Verifikasi NYATA M1→M6 dengan data real (BUKAN asumsi) — angka LC-4 TERBUKTI tetap valid

| Modul | Hasil data real | Angka LC-4/audit | Status |
|---|---|---|---|
| M1 | 15m=140,103; 1h=35,032; 4h=8,765; 1d=1,461 (raw, dgn gap outage) | dalam toleransi 5% V-DATA-001 | ✅ PASS |
| M2 | 4 TF PASS (15m/1h ada WARNING outage, tercatat) | aggregate gap ratio ≪ 5% | ✅ PASS |
| M3 | master **35,064 × 21**, grid gap-free, **V-LEAK-001 8/8 PASS** | 35,064 | ✅ EXACT |
| M4 | **35,045 × 29**, first ts **2020-01-01 19:00**, 19 baris drop | 35,045 / 19 drop | ✅ EXACT |
| M5 | train=**26,285**, test=**8,760**; boundary+overlap+LC-3 PASS | test 8,760 (exact); train ~26,269 (±tol) | ✅ (test exact) |
| M6 | N_train_windows=**26,238**, **N_test_windows=8,760 EXACT**, **overlap=47**, V-LEAK-003/004 PASS | N_test=8,760 exact; overlap ≤47 | ✅ EXACT |

**Semua angka EXACT yang diaudit LC-4 (test=8,760, N_test_windows=8,760, overlap=47, M4=35,045, first-ts 2020-01-01 19:00) TERBUKTI tetap sama dengan data real** — reindex grid penuh (ADR-023) + epsilon volume_zscore memulihkan persis angka yang diaudit dengan mock. Angka train (26,285 / 26,238) sedikit di atas estimasi "~26,269/~26,222" tapi dalam toleransi ±50 dan konsisten (35,045 = 26,285 + 8,760).

**Full test suite: 301 passed** (287 sebelumnya + 14 regression test baru untuk 5 temuan). 8 file window `.npy` real tertulis di `data/processed/`.

#### Durasi aktual (referensi sesi berikutnya)
- M1 download (4 TF, BTC/USDT, public endpoint, no API key): **~1 menit** (first request cold ~13s).
- M2→M6 pada data real: **detik** (M3 ~beberapa detik, M4/M5/M6 masing-masing <5s). Pipeline data ringan; yang berat adalah M8 training (belum jalan).

#### Update status verifikasi M0–M6 (revisi dari MIGRATION doc)
Tabel status di `MIGRATION_TO_CLAUDE_CODE.md` menyatakan M1 "terverifikasi sebagian (mock)" dan M2–M6 "terverifikasi penuh" TAPI hanya via simulasi manual/synthetic. **Kini direvisi: M1–M6 TERVERIFIKASI PENUH DENGAN DATA REAL** (Binance BTC/USDT 2020–2023), lewat pytest sungguhan (301 passed) + eksekusi pipeline nyata end-to-end, dengan 5 bug data-real ditemukan & diperbaiki + gate V-LEAK-001..004 lulus pada data real.

#### M8 SMOKE TEST (1 run nyata) — SUKSES

Dijalankan `scripts/run_m8_training.py --seeds 42 --timeframes 1h` pada window real (26,238 window 1h), CPU:
- **50 epoch penuh** (tidak early-stop), loss turun sehat & monoton-ish: 0.996 (e0) → 0.689 (e1) → 0.494 (e6) → 0.386 (e30) → **0.3697 (best @ e49)**. Tak ada NaN/divergensi/hang.
- **~4.1 menit/epoch** konsisten di CPU → **1 run ≈ 3h25m** (23:07→02:32).
- Checkpoint tertulis: `checkpoints/branch_1h/seed_42/best_model.pt` (1.13 MB) + `latest_model.pt`, plus log per-run `logs/training_branch_1h_seed_42.log`.
- **Validasi muat-ulang:** `load_checkpoint` sukses, metadata ADR-010 benar (`checkpoint_kind=best`, `ts2vec_commit=b0088e14...`, `config_weight_decay=0.0001` vs `effective=0.01` sesuai ADR-021), `encode(test[:100]) → [100,64]` semua finite. **Pipeline end-to-end real TERBUKTI: M1 real → M2-M6 real → M8 train → checkpoint → encode.**

#### M8 — SELESAI PENUH 20/20 (data real, GPU Kaggle) ✅

Training 20 run (4 timeframe × 5 seed) selesai di Kaggle (2× T4, task-queue notebook), checkpoint di-push ke branch `kaggle-checkpoints`, lalu **ditarik ke repo lokal** (`git fetch origin kaggle-checkpoints && git checkout FETCH_HEAD -- checkpoints`; checkpoints/ di-gitignore → tidak di-commit ke `main`).

**Validasi NYATA (bukan cek nama file — tiap bundle di-load):**
- **20/20 `best_model.pt` valid**, 0 missing, 0 invalid; 20 `latest_model.pt` juga ada.
- **`ts2vec_commit` = pin `b0088e14…` pada SEMUA 20.**
- 19 run full 50 epoch; **1 run** (`4h/seed456`) early-stop di epoch 45 (best@34, patience) — completed sah, bukan terpotong.
- `best_loss` konsisten per timeframe: 15m/1h ≈ 0.37, 4h ≈ 0.11, 1d ≈ 0.08 (best epoch 34–49).
- Full suite tetap **352 passed** setelah checkpoint masuk repo lokal.

**M8 kini SELESAI PENUH** (bukan lagi smoke-test) → M9 siap dijalankan dengan 20 checkpoint REAL.

### M9 — Fusion (KODE+TEST SELESAI sesi ini)

**Kode:** [`src/models/fusion.py`](../src/models/fusion.py) — `FusionModule` (concat fine-to-coarse 15m→1h→4h→1d per ADR-013, lalu proyeksi acak terpin ADR-003 → 256-dim) + `EmbeddingPipeline` (`encode_all_branches`, `fuse_condition`). **0 parameter trainable.** Proyeksi `P` deterministik dari `projection_seed` (base.yaml=42), baris di-normalisasi L2, `requires_grad=False`. Concat order dipaksa via tuple berurutan (bukan set/dict). 4TF (concat sudah 256) TETAP diproyeksi (bukan identitas).

Mapping kondisi→branch (DS-03 Table 3.10): 1TF=[1h]→64; 2TF=[15m,1h]→128; 3TF=[15m,1h,4h]→192; 4TF=[semua]→256; BL-15m/BL-4h/BL-1d=single→64. Semua diproyeksi ke 256.

**Test:** [`tests/test_fusion.py`](../tests/test_fusion.py) — 39 test. Full suite: **340 passed**. Menutup V-MODEL-002/V-INV-002 ([N,256] utk 7 kondisi), V-MODEL-003/V-INV-003 (0 param trainable; P tak berubah setelah gradient step), V-MODEL-004 (deterministik per seed), ADR-013 concat order, dan 4TF-still-projected.

**Catatan eksekusi — SUDAH DIJALANKAN DENGAN DATA REAL (sesi ini):** dengan 20 checkpoint M8 real, `scripts/run_m9_fusion.py` (baru) meng-encode window M6 real → embedding branch → fused untuk 7 kondisi. Output di `experiments/m9_real/embeddings/` (gitignored): **40 file branch** (`[N,64]`, 4 tf × 5 seed × 2 split) + **70 file fused** (`[N,256]`, 7 kondisi × 5 seed × 2 split). Semua 70 fused tervalidasi `[N,256]` float32 finite; train N=26,238, test N=8,760. Test data-real baru [`tests/test_fusion_real.py`](../tests/test_fusion_real.py) (11 test): branch-encode real `[N,64]`, fusi real `[N,256]` utk 7 kondisi, determinisme, kondisi berbeda → hasil berbeda. (Logika fusi inti tetap juga teruji dgn stub di `test_fusion.py`.)

### M10.5 — External Baselines (HMM + KM-PCA) — **SELESAI (kode + test + eksekusi real 10-run)**

**Keputusan KM-PCA: FINAL (Opsi 1, disetujui user via dokumen `EXTERNAL_BASELINE_PCA_UPDATE.md` + `PROMPT_CLAUDE_CODE_M10.5_PCA_DECISION.md`).** Kontradiksi `PCA(10)` vs 7 fitur 1h diselesaikan dengan **clamp `n_components = min(10, n_features) = 7`** → PCA full-rank (whitening/dekorelasi ortogonal, BUKAN reduksi dimensi), demi keadilan info vs kondisi 1TF/BL-1h. Difinalkan di **ADR-024** (kini di ADR Index [docs/DS-01_v1.2.md](DS-01_v1.2.md); addendum asli diarsip di `docs/archive/`). base.yaml `km_pca.pca_components: 10` kini diberi komentar eksplisit "ceiling, bukan nilai final".

**Kode:** [`src/models/external_baselines.py`](../src/models/external_baselines.py) — `HMMBaseline` (GaussianHMM, `covariance_type='diag'`, seleksi BIC atas `n_components∈{2,3,4}`), `KMeansPCABaseline` (PCA clamp→7, seleksi Silhouette atas `k∈{2,3,4,5,6}`, simpan explained-variance-ratio), `ExternalBaselineRunner` (10 run: 2 metode × 5 seed, tulis labels/model/selection-json). CLI: [`scripts/run_m10p5_baselines.py`](../scripts/run_m10p5_baselines.py). Sepenuhnya independen dari TS2Vec/M6/M7/M8/M9 (tak import satupun).

**Test:** [`tests/test_external_baselines.py`](../tests/test_external_baselines.py) — 12 test pakai **DATA REAL M5** (slice `_1h`, bukan synthetic). Full suite: **352 passed**. Menutup V-EXP-004 (10 run 5-seed), ADR-024 (test eksplisit: PCA efektif=7 bukan 10, tidak ada ValueError; + test yang membuktikan `PCA(10)` pada 7-dim memang error tanpa clamp), seleksi BIC/Silhouette, determinisme per-seed, label int M11-consumable.

**EKSEKUSI REAL (10-run penuh, CPU, ~1 menit):** `scripts/run_m10p5_baselines.py` dijalankan pada M5 penuh (train=26,285, test=8,760). Hasil konsisten lintas 5 seed: **HMM pilih n_components=4** (BIC terendah), **KM-PCA pilih k=2** (silhouette 0.625), **PCA=7** (explained variance: PC1≈95%, PC2≈4.8%, sisanya ~0 → full-rank). Output: `experiments/m10p5_external_baselines/external_baselines/{hmm,kmpca}/` (labels_train/test + model.pkl + selection.json per seed; 40 file). Siap dikonsumsi M11. *(experiments/ di-gitignore — artefak regenerable, tak di-commit.)*

**Catatan CATATAN pilihan engineering (didokumentasikan, bukan diam-diam):** (a) HMM `covariance_type='diag'` (default hmmlearn, robust/standar utk regime detection; spec tak membatasi). (b) Tidak ada StandardScaler sebelum PCA — mengikuti spec literal "PCA + KMeans" tanpa menambah parameter tak terdokumentasi (fitur sudah turunan OHLCV & sebagian ternormalisasi). Bisa direvisit bila perlu.

**Item dokumentasi tersisa (sesuai prompt, BELUM dikerjakan — perlu pass terpisah):** lipat ADR-021/022/023/**024** ke ADR Index DS-01 pada satu version bump (DS-01 v1.2). Ini churn lintas-dokumen (menyentuh version-pin di MIGRATION doc) → dikerjakan sebagai pass dokumentasi khusus, bukan disisipkan di sini. Lihat Item Terbuka #6.

## Item Terbuka
1. **[SELESAI]** ~~Commit hash TS2Vec~~ — status "Pinned".
2. **[SELESAI — terlewati]** ~~Jalankan `setup_and_verify.sh`~~ — tujuannya (buktikan M0–M6 lulus di environment penuh) sudah tercapai jauh melampaui itu: venv 3.11 dibuat, 301 test lulus, pipeline M1–M6 diverifikasi dengan DATA REAL, bahkan M8 smoke-test 1 run sukses. Skrip `.sh` aslinya tak bisa jalan apa adanya (lihat Sesi 9) dan sudah tidak relevan.
3. **[SELESAI — via prebuilt wheel, TANPA MSVC]** ~~`hdbscan` + `hmmlearn` belum terinstall~~. Diselesaikan sesi ini lewat jalur ringan (bukan MSVC Build Tools):
   - Dicek: MSVC C++ Build Tools **TIDAK ADA** di mesin ini (vswhere/cl.exe/registry semua nihil).
   - `hmmlearn==0.3.2` → **wheel prebuilt cp311/win tersedia**, terinstall bersih.
   - `hdbscan==0.8.38.post1` (pin lama) → **tak ada wheel** (harus compile). Wheel prebuilt tersedia untuk **`hdbscan==0.8.40`** yang **kompatibel & mempertahankan `scikit-learn==1.5.0`** (versi 0.8.44 ditolak karena memaksa sklearn→1.9.0). sklearn built-in `HDBSCAN` tak punya `approximate_predict`, jadi paket standalone tetap wajib.
   - **DEVIASI pin (minor, DISETUJUI USER — FINAL):** `hdbscan` **0.8.38.post1 → 0.8.40** (sebelumnya belum pernah benar-benar terinstall, jadi tak ada hasil yang berubah). Same maintainer, same API `approximate_predict`. sklearn tetap **1.5.0** pinned. Menghindari install MSVC ~GB. **Keputusan final (disetujui user): pakai wheel prebuilt 0.8.40, TIDAK diulang dengan MSVC.** `requirements.txt` + `environment.yml` diperbarui + catatan (commit `0633764`).
   - **Verifikasi:** `import hdbscan` (0.8.40) + `import hmmlearn` (0.3.2) sukses; `HDBSCAN.fit` + `approximate_predict` smoke OK; `sklearn` tetap 1.5.0; `numpy` tetap 1.26.4. **Full suite tetap 340 passed (0 regresi).**
   - **Metode instalasi (referensi environment setup):** `uv pip install --python .\.venv\Scripts\python.exe --only-binary=:all: "hdbscan==0.8.40" "hmmlearn==0.3.2" "scikit-learn==1.5.0"`. M10 & M10.5 kini TIDAK lagi terblokir install.
4. `.docx` asli DS-04 (dan cek DS-03 asli) — masih berpotensi mengandung versi lama sebelum audit LC-4; relevan hanya kalau dipakai di luar repo untuk sidang/laporan. Sumber kebenaran tetap file `.md` di `docs/`.
5. *(prioritas rendah)* Fork fallback TS2Vec — vendored copy sudah ada sebagai referensi dan kini di-import langsung (ADR-021), jadi fork eksternal kemungkinan tak perlu lagi; simpan sebagai catatan reproducibility saja.
6. **[SELESAI sesi 10]** ~~Lipat ADR-021/022/023/024 ke ADR Index DS-01~~ — **`docs/DS-01_v1.2.md` dibuat** berisi ADR-001–024 dalam satu dokumen (ADR-001–020 identik kata-per-kata dgn v1.1; changelog v1.1→v1.2 ditambahkan; 4 baris index + 4 section penuh ditambahkan). File addendum asli dipindah ke `docs/archive/`. Referensi version-pin "DS-01 v1.1" → "v1.2" diupdate di `MIGRATION_TO_CLAUDE_CODE.md`, `IMP-01_v1.3.md` (Depends-on header), `README.md`. Referensi historis (changelog/audit di DS-03/IMP-01) sengaja TIDAK diubah. DS-01 kembali single source of truth utk semua ADR. *(Amendemen DS-02 Stage 2/3 & DS-03 §4 footnote masih bisa dilakukan nanti, tapi bukan bagian tugas konsolidasi ADR ini.)*

## TODO List Terbaru

- [x] M0-M6 — **[SEMUA DONE, lihat checkpoint sesi 1-7 untuk detail]**
- [x] **[MILESTONE 2 — DATA PIPELINE COMPLETE]** ✅
- [x] TS2Vec commit hash — **[PINNED sesi ini]**
- [x] Persiapan migrasi Claude Code — **[SELESAI sesi ini]**
- [ ] **[MIGRASI]** User push repo ke GitHub, buka Claude Code, jalankan `setup_and_verify.sh`
- [x] M7 — TS2Vec Wrapper — **[SELESAI sesi 9: kode+18 test hijau; gate torch-2.3.1 terverifikasi; 2 friksi menunggu keputusan user, lihat bagian M7]**
- [x] M8 — Branch Training — **[SELESAI PENUH 20/20 di GPU Kaggle; semua checkpoint di-load & tervalidasi (ts2vec_commit=pin); ditarik ke lokal]**
- [x] M9 — Fusion — **[KODE+39 test hijau sesi 9 (340 total); eksekusi fused-embeddings menunggu 20 checkpoint M8]**
- [x] M10.5 — External Baselines (HMM + KM-PCA) — **SELESAI sesi 10: kode+12 test hijau (352 total); KM-PCA clamp→7 (ADR-024); eksekusi real 10-run sukses (HMM n=4, KM-PCA k=2). Lihat bagian "M10.5" di atas.**
- [ ] M10 — HDBSCAN Clustering
- [ ] M11 — Evaluation
- [ ] M12 — Visualization (paralel M14)
- [ ] M14 — Statistical Analysis (paralel M12)
- [ ] M13 — Experiment Runner
- [ ] M15 — Paper Artifact Generator

## Instruksi untuk Sesi Berikutnya (di Claude Code)
**Baca `MIGRATION_TO_CLAUDE_CODE.md` di root repo terlebih dahulu** — file itu sendiri berisi instruksi detail lebih lengkap dari ringkasan singkat ini, termasuk urutan langkah wajib, tabel status verifikasi, dan detail API TS2Vec. Jangan mulai coding sebelum membaca file itu dan menjalankan `setup_and_verify.sh`.
