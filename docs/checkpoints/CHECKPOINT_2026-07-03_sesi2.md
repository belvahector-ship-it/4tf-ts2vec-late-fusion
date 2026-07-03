# CHECKPOINT — Sesi 2 — 2026-07-03

## Status Global
Milestone saat ini: **2 — Data Pipeline Complete** (in progress)
Modul terakhir DONE: **M1 — Data Acquisition** (kode lengkap, verifikasi parsial nyata dijalankan)
Modul sedang IN PROGRESS: tidak ada — menunggu instruksi lanjut ke M2

## Keputusan Anda yang Diterapkan Sesi Ini

1. **TS2Vec commit hash:** menggunakan repo original — dikonfirmasi via web search bahwa URL resmi sekarang `https://github.com/zhihanyue/ts2vec` (author username berubah dari `yuezhihan` → `zhihanyue`, masih repo AAAI-22 yang sama, terverifikasi lewat README/API/citation yang cocok). **Commit hash literal BELUM bisa saya ambil** (GitHub commit page tidak expose SHA penuh lewat search snippet, dan saya tidak menebak hash demi keamanan reproducibility). Placeholder `COMMIT_HASH`/`REPLACE_WITH_PINNED_COMMIT_HASH` sudah diupdate di `configs/base.yaml`, `requirements.txt`, `environment.yml`, `README.md` dengan instruksi konkret: jalankan `git ls-remote https://github.com/zhihanyue/ts2vec HEAD`.
2. **Fork fallback:** placeholder tetap ada, akan diisi setelah Milestone 1 sesuai keputusan Anda.
3. **pytest:** di-skip sesuai instruksi (keterbatasan kondisi saat ini).
4. **Milestone tunda push ke GitHub:** sampai Milestone 2 selesai, sesuai instruksi Anda.
5. **Artefak checkpoint:** sudah dibagikan sebagai file terpisah untuk diunduh (`CHECKPOINT_LATEST.md` + salinan bertanggal) di luar bundle zip repo.

## Yang Sudah Selesai Sesi Ini

### Kode (`src/data/`)
- `acquisition.py` — modul lengkap M1:
  - `BinanceDownloader` class: `download()`, `_fetch_with_retry()` (exponential backoff, IMP-01 Risk R-07), `save_parquet()`, `compute_checksum()` (SHA-256)
  - `check_row_count_tolerance()` — toleransi 5% sesuai DS-04 V-DATA-001, dengan `EXPECTED_ROW_COUNTS` persis dari DS-02 Stage 0 (15m≈140,256; 1h≈35,064; 4h≈8,766; 1d≈1,461)
  - `build_manifest()`, `save_manifest()`, `load_manifest()` — schema manifest.json sesuai DS-02 Stage 0 (symbol, exchange, download_timestamp, ccxt_version, timeframes{rows,start,end,file,sha256})
  - `is_already_downloaded()` — logic idempotency (skip re-download jika checksum cocok)
  - `run_acquisition()` — orkestrator penuh untuk 4 timeframe
  - Semua exception informatif (`AcquisitionError`, `ValueError`), tidak ada `print()` — hanya `logging` (sesuai DoD M1, diverifikasi dengan test statis)
  - `ExchangeError` (non-transient) gagal langsung tanpa retry sia-sia; `NetworkError`/`RateLimitExceeded` (transient) di-retry dengan backoff eksponensial

### CLI (`scripts/`)
- `run_m1_acquisition.py` — entrypoint tipis, baca config, panggil `run_acquisition()`, exit code informatif (0/1), logging ringkasan hasil per timeframe.

### Test (`tests/test_acquisition.py`)
- ~30 test case: row count tolerance, checksum determinism, Parquet round-trip, manifest build/save/load, idempotency (4 skenario), download dengan mocked ccxt exchange (single-page, multi-retry, exchange error tanpa retry, retries exhausted), dan static check "no print() calls".

## Verifikasi yang BENAR-BENAR Dijalankan (bukan klaim)

✅ **Sintaks semua file valid** (`acquisition.py`, `run_m1_acquisition.py`, `test_acquisition.py`)
✅ **`check_row_count_tolerance()`** dijalankan manual nyata — batas 5% toleransi benar untuk semua 4 timeframe
✅ **`compute_checksum()`** dijalankan manual nyata — deterministik, berbeda untuk konten berbeda, valid SHA-256 hex (64 karakter)
✅ **`build_manifest()` + `save_manifest()` + `load_manifest()`** roundtrip dijalankan manual nyata — manifest valid JSON, field lengkap sesuai schema DS-02
✅ **`is_already_downloaded()`** — 4 skenario (match/mismatch/no-manifest/missing-file) semua dijalankan manual nyata dan benar
✅ **`DownloadResult.__post_init__`** — row_count/start/end dihitung benar, diverifikasi manual nyata
✅ **Logging aktif** (bukan print) — terbukti dari output log saat `save_manifest()` dipanggil di test manual

## Yang BELUM Bisa Diverifikasi Nyata di Sandbox Ini

⚠️ Sandbox **tidak punya `ccxt` dan `pyarrow` terinstall, dan tanpa akses network** untuk `pip install` maupun fetch data Binance sungguhan. Akibatnya:
- `BinanceDownloader.download()` **jalur nyata (bukan mock)** — belum pernah dieksekusi.
- `_fetch_with_retry()` dengan exception class asli `ccxt.NetworkError`/`ccxt.RateLimitExceeded`/`ccxt.ExchangeError` — test yang saya tulis pakai mock, **belum dijalankan via pytest** karena `ccxt` tidak ter-install (test itu sendiri butuh `import ccxt` untuk exception classes).
- `save_parquet()`/`pd.read_parquet()` — butuh `pyarrow`, belum diverifikasi nyata.
- **Download data BTC/USDT sungguhan 2020-2023 dari Binance** — jelas butuh network, harus dilakukan Anda.

➡️ **Tindakan yang disarankan:** di environment ber-network (lokal/Colab), jalankan:
```bash
pip install -r requirements.txt
pytest tests/test_acquisition.py -v
python scripts/run_m1_acquisition.py --config configs/base.yaml
```
Expected output: 4 file Parquet di `data/raw/` + `data/raw/manifest.json`, dengan row count dalam toleransi 5% dari `EXPECTED_ROW_COUNTS`.

## Keputusan/Deviasi
- Tidak ada deviasi dari DS-01/DS-02/DS-03. URL repo TS2Vec diperbarui dari `yuezhihan` ke `zhihanyue` berdasarkan konfirmasi web search — dicatat sebagai koreksi faktual, bukan deviasi protokol.

## Item Terbuka yang Butuh Keputusan/Tindakan Anda
1. **Isi commit hash TS2Vec** — jalankan `git ls-remote https://github.com/zhihanyue/ts2vec HEAD`, tempel ke 3 file (`configs/base.yaml`, `requirements.txt`, `environment.yml`).
2. **Jalankan `pytest tests/ -v` + `scripts/run_m1_acquisition.py`** di environment ber-network untuk validasi penuh M0+M1, termasuk download data sungguhan.
3. Fork fallback TS2Vec — buat setelah Milestone 1 selesai (sesuai rencana Anda).

## TODO List Terbaru (state penuh)

- [x] M0 — Project Bootstrap **[DONE]**
- [x] M1 — Data Acquisition **[DONE — kode lengkap, verifikasi parsial nyata, eksekusi network penuh menunggu Anda]**
- [ ] M2 — Data Validation ← **lanjut di sini**
- [ ] M7 — TS2Vec Wrapper (bisa paralel dengan M3-M6)
- [ ] M3 — Temporal Alignment
- [ ] M4 — Feature Engineering
- [ ] M5 — Temporal Split
- [ ] M6 — Window Generation
- [ ] **[GATE] V-LEAK-001 s.d. V-LEAK-004** — wajib lulus sebelum M8
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
**Mulai dari:** M2 — Data Validation (IMP-01 v1.1, DS-02 Stage 1, DS-04 §3.1 V-DATA-001/V-DATA-002).
**File yang perlu dilihat pertama:** `docs/DS-02_Data_Flow_Specification.md` bagian "Stage 1: Data Validation" (tabel Validation Checks + LC-1), dan `docs/IMP-01_Implementation_Roadmap_v1.1.md` bagian M2.
**Catatan:** M2 murni logic pandas (tidak butuh ccxt/pyarrow untuk sebagian besar check), jadi kemungkinan besar bisa diverifikasi lebih nyata di sandbox ini dibanding M1.
