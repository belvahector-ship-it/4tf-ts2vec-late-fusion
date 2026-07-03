# MIGRATION_TO_CLAUDE_CODE.md

**Baca file ini PERTAMA, sebelum membaca file lain di repo ini, jika
Anda adalah Claude Code (atau AI lain) yang baru mulai sesi di repo
ini.**

## Konteks

Proyek ini dikerjakan dalam serangkaian sesi Claude.ai (chat biasa,
bukan Claude Code) dari M0 hingga M6, di sebuah sandbox tanpa akses
network dan tanpa `torch`/`pytest`/`ccxt` terinstall. Setiap keputusan,
setiap bug yang ditemukan dan diperbaiki, dan setiap verifikasi yang
dijalankan (atau TIDAK bisa dijalankan karena keterbatasan sandbox)
tercatat lengkap di `docs/CHECKPOINT_LATEST.md` dan
`docs/checkpoints/CHECKPOINT_2026-07-03_sesi*.md`.

**Jangan menulis ulang modul M0–M6 dari nol.** Semuanya sudah selesai,
diverifikasi (sejauh mungkin di sandbox lama), dan disepakati desainnya
dengan penulis proyek — termasuk dua putaran audit dokumen desain yang
menemukan dan memperbaiki 3 bug logika di spesifikasi asli. Menulis
ulang akan membuang kerja itu dan berisiko mengulang bug yang sudah
diperbaiki.

## Langkah wajib di awal sesi

1. **Baca `docs/CHECKPOINT_LATEST.md` secara penuh.** Ini ringkasan
   status paling akhir: modul mana yang selesai, apa yang sudah
   diverifikasi, apa yang belum, dan instruksi eksplisit "mulai dari
   mana" di bagian bawah file.
2. **Baca `docs/AUDIT_LC4_ADDENDUM.md`.** Ini audit paling kritis
   dalam proyek — menjelaskan kenapa desain M6 (window generation)
   berbeda dari asumsi naif, dan kenapa `N_test_windows = 8,760`
   (bukan ~8,713) dengan overlap window persis 47.
3. **Baca `docs/IMP-01_v1.3.md`** (roadmap implementasi, versi
   terbaru) untuk spesifikasi modul M7 dan seterusnya. **Jangan pakai
   versi IMP-01 lain jika ada di histori — v1.3 adalah yang final.**
4. Cek versi dokumen desain yang dipakai: **DS-01 v1.1, DS-02 v1.2,
   DS-03 v1.2, DS-04 v1.1, IMP-01 v1.3**. Kalau ada nama file dengan
   angka versi lebih rendah tercecer di suatu tempat, itu bukan
   sumber kebenaran — pakai yang disebut di sini.

## Yang BENAR-BENAR sudah selesai (M0–M6)

| Modul | Status kode | Status verifikasi |
|---|---|---|
| M0 — Project Bootstrap | Selesai | Terverifikasi sebagian (logic config/seed manual, torch belum) |
| M1 — Data Acquisition | Selesai | Terverifikasi sebagian (logic murni via mock; network real belum pernah jalan) |
| M2 — Data Validation | Selesai | **Terverifikasi penuh** (33/33 assertion nyata) |
| M3 — Temporal Alignment | Selesai | **Terverifikasi penuh** (37/37 assertion, 2 bug ditemukan+diperbaiki) |
| M4 — Feature Engineering | Selesai | **Terverifikasi penuh** (29/29 assertion, 1 bug tanggal diperbaiki) |
| M5 — Temporal Split | Selesai | **Terverifikasi penuh** (21/21 assertion) |
| M6 — Window Generation | Selesai | **Terverifikasi penuh** (23/23 assertion, desain direvisi total setelah audit LC-4) |

"Terverifikasi penuh" di atas berarti: logic dijalankan nyata dengan
Python (`pandas`/`numpy` tersedia di sandbox lama), TAPI **belum pernah
lewat `pytest` sungguhan** karena `pytest` sendiri tidak terinstall di
sandbox lama. Semua file test (`tests/test_*.py`) sudah lengkap dan
seharusnya lulus — tapi ini klaim yang HARUS Anda verifikasi ulang,
bukan diasumsikan benar begitu saja.

## LANGKAH PERTAMA yang wajib Anda lakukan di Claude Code

Sebelum menulis kode baru untuk M7, jalankan ini sebagai sanity check —
karena Claude Code punya akses network/torch/pytest penuh yang tidak
pernah ada sebelumnya, dan environment yang berbeda (versi
pandas/numpy/python) bisa memunculkan bug baru, persis seperti yang
terjadi 2 kali sebelumnya (lihat CHECKPOINT sesi 4 dan sesi 6 untuk
detail bug yang ditemukan gara-gara perbedaan versi pandas):

```bash
# 1. Install dependencies
pip install -r requirements.txt
# atau: conda env create -f environment.yml && conda activate market-state-discovery

# 2. Jalankan SELURUH test suite M0-M6 sebagai baseline
pytest tests/ -v --tb=short

# 3. Kalau ada test yang GAGAL, JANGAN buru-buru "perbaiki" test-nya.
#    Cek dulu apakah ini bug di kode src/, atau memang perbedaan versi
#    environment yang butuh penyesuaian yang hati-hati (seperti kasus
#    pandas 3.x datetime64[us] vs [ns] yang sudah 2x ditemukan).
#    Laporkan ke user/dokumentasikan di checkpoint sebelum mengubah apa pun.
```

Kalau semua test lulus: lanjut ke M7 sesuai spesifikasi
`docs/IMP-01_v1.3.md`. Kalau ada yang gagal: itu prioritas utama
sebelum M7, karena M7-M15 semua dibangun di atas fondasi M0-M6.

## Status TS2Vec (M7 siap dimulai)

- Commit sudah di-pin: `b0088e14a99706c05451316dc6db8d3da9351163`
  (lihat `configs/base.yaml` blok `ts2vec:` untuk provenance lengkap)
- Source code referensi (read-only) ada di
  `third_party_reference/ts2vec/` — **JANGAN import dari sini**, ini
  hanya untuk dibaca. Runtime dependency yang sesungguhnya di-install
  via `pip install -r requirements.txt` (yang akan clone dari GitHub
  di commit yang sama).
- API kunci yang sudah dikonfirmasi dari pembacaan source:
  - `TS2Vec(input_dims, output_dims=320, hidden_dims=64, depth=10, device='cuda', lr=0.001, batch_size=16, ...)`
  - `.fit(train_data, n_epochs=None, n_iters=None, verbose=False)` — `train_data` shape `[n_instance, n_timestamps, n_features]`
  - `.encode(data, encoding_window=None, ...)` — **WAJIB set `encoding_window='full_series'`** untuk mendapat 1 vector per window `[N, output_dims]` (default tanpa parameter ini mengembalikan representasi per-timestamp `[N, T, output_dims]`, BUKAN yang kita mau sesuai DS-03)
  - `.save(fn)` / `.load(fn)` — menyimpan `self.net.state_dict()` (bukan `self._net`) karena pakai `torch.optim.swa_utils.AveragedModel`
- Dependency inti TS2Vec cuma butuh `torch`+`numpy` (sudah ada di
  `requirements.txt` kita). `scipy`/`pandas`/`sklearn`/`Bottleneck`/
  `statsmodels` di `requirements.txt` resmi TS2Vec HANYA dipakai oleh
  `tasks/`/`datautils.py`/`train.py` mereka (benchmark harness UCR/UEA)
  yang tidak kita pakai sama sekali.
- **Perhatian versi**: TS2Vec resmi minta `torch==1.8.1`, proyek kita
  pin `torch==2.3.1`. Source code intinya (`ts2vec.py`+`models/`) hanya
  pakai torch API dasar yang stabil lintas versi (Conv1d, AdamW,
  DataLoader, `torch.optim.swa_utils.AveragedModel`), jadi kemungkinan
  besar aman — **tapi ini WAJIB diverifikasi nyata di M7**, bukan
  diasumsikan. Kalau ada incompatibility, dokumentasikan sebagai
  deviation di checkpoint, jangan diam-diam downgrade torch (itu akan
  merusak requirement M0's pinned torch version dan mempengaruhi
  seluruh pipeline M8+).

## Protokol kerja yang disepakati (lanjutkan pola yang sama)

Dari `docs/CHECKPOINT_LATEST.md`, protokol yang sudah berjalan:
1. TODO list eksplisit di awal setiap sesi/modul.
2. Tulis kode → tulis test → **jalankan test nyata** (bukan simulasi
   manual — sekarang Anda punya `pytest` sungguhan, gunakan).
3. Kalau menemukan inkonsistensi di dokumen desain (DS-01 s.d. DS-04,
   IMP-01), **JANGAN diam-diam pilih salah satu interpretasi**. Susun
   bukti (cross-check angka/formula), lalu **tanya user** sebelum
   melanjutkan — persis seperti 3 kasus audit yang sudah terjadi
   (lihat `docs/AUDIT_REPORT_DS01-DS04_IMP01.md` dan
   `docs/AUDIT_LC4_ADDENDUM.md` sebagai contoh polanya).
4. Update `docs/CHECKPOINT_LATEST.md` di akhir sesi (dan simpan salinan
   bertanggal di `docs/checkpoints/`), dengan bagian eksplisit: apa
   yang selesai, apa yang terverifikasi NYATA (bukan klaim), apa yang
   belum, dan instruksi "mulai dari mana" untuk sesi berikutnya.
5. Realistis: kalau sesuatu tidak bisa dieksekusi/diverifikasi (jarang
   terjadi sekarang karena Claude Code lebih leluasa dari sandbox lama,
   tapi tetap mungkin, misal training GPU yang butuh berjam-jam),
   katakan itu terus terang, jangan klaim "berhasil" tanpa bukti.

## Urutan modul berikutnya (per IMP-01 v1.3)

```
M7  — TS2Vec Wrapper       (siap mulai — commit sudah di-pin)
M5, M10.5 sudah/bisa paralel dengan M7 (M5 selesai; M10.5 belum, hanya butuh M5)
M8  — Branch Training      (butuh M6 + M7)
M9  — Fusion                (butuh M8)
M10 — HDBSCAN Clustering    (butuh M9)
M11 — Evaluation            (butuh M10 + M10.5)
M12, M14 — paralel setelah M11
M13 — Experiment Runner
M15 — Paper Artifact Generator
```

Gate wajib sebelum M8 mulai training sungguhan: pastikan
`pytest tests/test_alignment.py tests/test_temporal_split.py
tests/test_window_generation.py -v` semua lulus (ini adalah V-LEAK-001
s.d. V-LEAK-004, gate ilmiah paling kritis di seluruh proyek).
