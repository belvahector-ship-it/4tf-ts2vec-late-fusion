# CHECKPOINT — Sesi 1 — 2026-07-03

## Status Global
Milestone saat ini: **1 — Repository Ready** (in progress, mendekati selesai)
Modul terakhir DONE: **M0 — Project Bootstrap** (kode lengkap, test ditulis, verifikasi parsial dijalankan nyata)
Modul sedang IN PROGRESS: tidak ada — menunggu instruksi lanjut ke M1

## Yang Sudah Selesai Sesi Ini

### Struktur Repo
- Seluruh struktur folder dibuat sesuai IMP-01 v1.1: `src/{utils,data,models,experiments,evaluation,visualization}`, `scripts/`, `tests/`, `configs/`, `docs/checkpoints/`, `data/{raw,interim,processed}`, `checkpoints/`, `experiments/`, `outputs/final/{figures,tables}`, `logs/`.

### Kode (`src/utils/`)
- `paths.py` — path constants terpusat, termasuk `TS2VEC_CONDITIONS` (7), `EXTERNAL_BASELINES` (2), `TIMEFRAMES` (4), `RANDOM_SEEDS` (5), helper `get_experiment_dir()`, `get_branch_checkpoint_dir()`.
- `config.py` — YAML loader + schema validator (`REQUIRED_BASE_FIELDS` mengikuti DS-03 Tabel 3.11 penuh), `load_condition_config()` untuk merge base+override, `ConfigValidationError` dengan pesan informatif per-field.
- `seed.py` — `set_all_seeds()` (Python/NumPy/PyTorch/CUDA identik), `get_torch_rng_state()` untuk verifikasi.
- `device.py` — `get_device()`, `get_device_info()`.
- `logging_utils.py` — `get_logger()` standar (console + file handler).

### Konfigurasi (`configs/`) — 10 file
- `base.yaml` — SEMUA controlled variables DS-03 Tabel 3.11 (encoder, fusion, window, training, clustering, dataset, seeds, statistics, external_baselines, ts2vec pin).
- 7 config kondisi TS2Vec: `experiment_1tf.yaml`, `experiment_2tf.yaml`, `experiment_3tf.yaml`, `experiment_4tf.yaml`, `experiment_bl_15m.yaml`, `experiment_bl_4h.yaml`, `experiment_bl_1d.yaml` — masing-masing hanya override `condition.name` + `condition.active_timeframes` (INV-001 dipatuhi, diverifikasi test).
- 2 config baseline eksternal: `baseline_hmm.yaml`, `baseline_kmpca.yaml`.

### Infrastruktur
- `requirements.txt` + `environment.yml` — konsisten, TS2Vec pinned via placeholder commit hash (`COMMIT_HASH` — **belum diisi commit asli**, lihat item terbuka).
- `.gitignore` — mengecualikan data/, checkpoints/, outputs/, experiments/, logs/, __pycache__/, .env; `.gitkeep` dipasang di semua folder yang harus tetap ada.
- `README.md` — skeleton lengkap: project summary, install (pip+conda), dataset info, reproduction steps, expected outputs, struktur repo.
- `docs/` — 6 dokumen desain di-copy: PROPOSAL, DS-01, DS-02, DS-03.docx, DS-04.docx, IMP-01 v1.1 (hasil koreksi sesi sebelumnya).

### Test (`tests/`) — 4 file, ~35 test case
- `test_config.py` — validasi skema, error informatif, real config files (base + 7 kondisi + 2 eksternal), verifikasi INV-001 (kondisi hanya beda di `condition` block), verifikasi 7 nama unik.
- `test_seed.py` — reproducibility Python/NumPy/PyTorch per seed, 5 seed DS-03 diparameterisasi.
- `test_device.py` — get_device() tidak pernah raise, konsisten dengan `torch.cuda.is_available()`.
- `test_paths.py` — struktur direktori, konstanta 7 kondisi/2 eksternal/4 timeframe/5 seed, **assert matematis 45 total run & 20 branch checkpoint**.

## Verifikasi yang BENAR-BENAR Dijalankan (bukan klaim)

✅ **Sintaks semua file `.py` valid** (`python3 -m py_compile` — PASS semua)
✅ **Semua 10 file YAML valid & ter-parse** (PASS semua)
✅ **Logic `config.py` dijalankan manual nyata** (bukan lewat pytest): base.yaml load & validasi OK, 7 kondisi merge & INV-001 terverifikasi OK, error informatif saat field hilang OK, 2 config eksternal OK, matematika run (35+10=45, 20 checkpoint) OK.
✅ **Logic `seed.py` bagian NumPy & Python `random`** diverifikasi manual nyata (reproducibility per seed, 5 seed DS-03 semua reproducible).

## Yang BELUM Selesai / Blocking

⚠️ **`pytest` dan `torch` TIDAK terinstall di sandbox ini, dan sandbox tidak punya akses network** (`bash_tool` network disabled) untuk `pip install`. Akibatnya:
- Test suite (`tests/*.py`) sudah ditulis lengkap tapi **belum pernah dijalankan via `pytest` di sini** — hanya logic intinya diverifikasi manual lewat `python3 -c "..."` untuk bagian yang tidak butuh torch.
- Bagian `seed.py` yang menyentuh `torch.manual_seed`/CUDA **belum diverifikasi eksekusi nyata** di sandbox ini.
- `device.py` **belum diverifikasi eksekusi nyata** sama sekali (butuh torch).

➡️ **Tindakan yang disarankan untuk Anda:** clone/copy repo ini ke environment dengan `torch` terinstall (lokal, Colab, atau `pip install -r requirements.txt` di environment yang ada network), lalu jalankan:
```bash
pytest tests/ -v
```
Saya akan bisa membantu debug jika ada test yang gagal begitu hasil run itu di-share balik.

## Keputusan/Deviasi
- Tidak ada deviasi dari DS-01/DS-03. Config placeholder `ts2vec.pinned_commit` dan `ts2vec.fallback_fork` di `base.yaml` sengaja diisi string `REPLACE_WITH_...` karena commit hash asli belum ditentukan penulis — ini **item terbuka**, bukan penyimpangan protokol (ADR-001 mengharuskan commit dipin sebelum M8/training dimulai, bukan sebelum M0 selesai).

## Item Terbuka yang Butuh Keputusan/Tindakan Anda
1. **Isi `ts2vec.pinned_commit` dan `ts2vec.fallback_fork`** di `configs/base.yaml` + `requirements.txt` + `environment.yml` (cari commit TS2Vec resmi yang mau dipakai, dan buat fork GitHub pribadi sebagai fallback — sesuai Risk R-01).
2. **Jalankan `pytest tests/ -v` di environment ber-network** untuk memvalidasi penuh (termasuk bagian torch) sebelum lanjut ke M1.
3. Konfirmasi apakah Anda ingin repo ini langsung di-push ke GitHub sekarang (Milestone 1 hampir selesai) atau menunggu sampai M1–M6 (Milestone 2) selesai.

## TODO List Terbaru (state penuh)

- [x] M0 — Project Bootstrap **[DONE — kode selesai, verifikasi parsial nyata, verifikasi penuh butuh env ber-torch]**
- [ ] M1 — Data Acquisition
- [ ] M2 — Data Validation
- [ ] M7 — TS2Vec Wrapper (bisa paralel dengan M3-M6)
- [ ] M3 — Temporal Alignment
- [ ] M4 — Feature Engineering
- [ ] M5 — Temporal Split
- [ ] M6 — Window Generation
- [ ] **[GATE] V-LEAK-001 s.d. V-LEAK-004** — wajib lulus sebelum M8
- [ ] M8 — Branch Training
- [ ] M9 — Fusion
- [ ] M10.5 — External Baselines (HMM + KM-PCA) — bisa paralel, hanya butuh M6
- [ ] M10 — HDBSCAN Clustering
- [ ] M11 — Evaluation
- [ ] M12 — Visualization (paralel M14)
- [ ] M14 — Statistical Analysis (paralel M12)
- [ ] M13 — Experiment Runner
- [ ] M15 — Paper Artifact Generator

## Instruksi untuk Sesi Berikutnya
**Mulai dari:** Milestone 1 gating question — "Can another researcher clone this repository and install all dependencies in one command?" — jawab ini dulu (tergantung item terbuka #1 di atas), baru lanjut M1 (Data Acquisition).
**File yang perlu dilihat pertama:** `docs/IMP-01_Implementation_Roadmap_v1.1.md` bagian M1, dan `configs/base.yaml` bagian `dataset:` untuk parameter akuisisi data Binance.
**Catatan:** M1 akan butuh `ccxt` untuk fetch data Binance — di sandbox ini tidak ada akses network, jadi M1 kemungkinan sama seperti M0: kode ditulis lengkap dan siap jalan, tapi eksekusi nyata (download data sungguhan) harus dilakukan Anda di environment dengan akses internet.
