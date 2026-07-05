# Jawaban Terstruktur: Struktur 3TF/4TF & Arah Pengembangan Embedding

> Ringkasan lanjutan dari `docs/PIPELINE_SCIENTIFIC_DISSECTION.md` dan `docs/M10_CLUSTERING_EXPLORATION.md`. Tidak ada kode yang diubah — ini sintesis dari bukti yang sudah diukur.

---

## 1. Apakah 3TF/4TF kehilangan struktur cluster, atau berubah jadi struktur lain?

**Jawaban singkat: BUKAN kehilangan struktur — berubah bentuk struktur, dari "blob terpisah" menjadi "manifold/trajectory kontinu".** Ini bukan interpretasi optimis semata; tiga baris bukti independen menunjukkannya:

| Bukti | Apa yang diukur | Hasil |
|---|---|---|
| **Hopkins statistic** | Kecenderungan cluster (0.5 = acak, →1 = terstruktur kuat) | **0.91–0.92 di SEMUA kondisi** — termasuk 4TF. Struktur tidak hilang. |
| **Cross-seed ARI** | Apakah partisi (KMeans) stabil lintas seed training berbeda | 3TF: 0.51–0.55, 4TF: 0.51–0.54 — jauh di atas random (~0), berarti struktur itu **reproducible**, bukan noise |
| **Kruskal-Wallis ekonomi** | Apakah struktur itu bermakna (return/volatilitas beda antar cluster) | p<1e-13 di SEMUA kondisi; separasi return **MENGUAT** di 3TF/4TF (KW-H 67.6→83.0) |

Jadi struktur ada dan bahkan makin bermakna — yang berubah adalah **geometrinya**: dari density-island (yang dibutuhkan HDBSCAN) menjadi **trajectory/manifold kontinu berdimensi lebih tinggi** (dimensi intrinsik 90%-varians naik 27→39→61→72 dari 1TF→4TF, distance-concentration CV turun 0.172→0.111).

**Mengapa ini secara matematis diharapkan, bukan kebetulan:**
- Window overlap stride-1 membuat embedding berturutan nyaris identik → lintasan Lipschitz-kontinu (delay embedding, teorema Takens 1981; Perea & Harer 2015). Ini sudah ada sejak 1TF.
- Menambah timeframe menambahkan subspace yang **hampir ortogonal** untuk TF kasar (CCA terukur: 15m↔1h≈0.99 nyaris duplikat, tapi 4h↔1d≤0.72 komplementer) → manifold produk yang lebih tinggi dimensinya, bukan cluster yang "pecah".
- HDBSCAN butuh **lembah densitas** (celah kosong antar blob) — itu yang hilang. Struktur *varians/arah* (yang dibutuhkan KMeans/GMM, dan yang membawa makna ekonomi) **tidak hilang**, malah menguat.

**Downstream task yang cocok untuk struktur trajectory/manifold (bukan cluster terpisah):**
1. **Partitioning berbasis varians** (sudah direkomendasikan): KMeans/GMM k tetap — memotong manifold jadi region, bukan mencari pulau densitas.
2. **Analisis trajectory/dinamika**: karena strukturnya adalah lintasan kontinu, task seperti *state-transition modeling* (Markov transition antar region), *trajectory forecasting* (ke mana arah pergerakan di manifold berikutnya), atau *persistent homology* (topological data analysis pada sliding-window point cloud, sesuai Perea & Harer) menjadi relevan justru KARENA strukturnya kontinu.
3. **Manifold learning eksplisit**: UMAP/PCA sebagai representasi utama (bukan hanya alat visualisasi), lalu density atau graph-based method pada manifold berdimensi rendah tsb.
4. **Downstream supervised/semi-supervised probing**: karena embedding kaya informasi (return separation kuat), bisa langsung dipakai sebagai fitur untuk task prediktif (bukan hanya cluster), misal regresi/klasifikasi arah return jangka pendek.

Kesimpulan bagian ini: **3TF/4TF tidak "gagal" — mereka menuntut kerangka downstream yang berbeda dari yang awalnya diasumsikan (density clustering).** Ini temuan yang bisa dijadikan bagian penting narasi paper, bukan disembunyikan sebagai kegagalan.

---

## 2. Apa yang bisa dikembangkan dari embedding ini? (dari hasil eksplorasi & traceback)

Berdasarkan seluruh rantai sebab-akibat (Stage 1–10 di dissection report), berikut arah pengembangan yang **punya dasar bukti**, diurutkan dari yang paling siap-pakai:

### A. Segera dieksekusi (data sudah ada, tinggal jalankan)
- **KMeans fixed-k=4 lintas kondisi** (rekomendasi M10 yang sudah diajukan) — memanfaatkan struktur varians yang terbukti kuat & bermakna secara ekonomi.
- **Analisis sensitivitas k∈{3,5,6,8}** untuk menguji robustness temuan 4-state (bull/bear/calm/choppy).

### B. Pengembangan metodologis (memperkuat kontribusi riset)
- **Diagnostik struktur sebagai kontribusi berdiri sendiri**: toolkit yang sudah dibangun (Hopkins, PCA participation-ratio, distance-CV, CCA antar-view, temporal-purity cluster) bisa dijadikan *metode umum* untuk memprediksi "apakah embedding ini cocok untuk density clustering atau partitioning" — berguna di luar proyek ini (Framing C di dissection report).
- **Cross-Timeframe Attention** (disebut di percakapan sebelumnya sebagai kontribusi utama riset): daripada concat+random-projection (yang terbukti hanya quasi-isometry netral), attention-based fusion bisa secara adaptif mem-bobot subspace TF berdasarkan relevansi konteks — berpotensi mengurangi penambahan dimensi yang "boros" (recall: 15m↔1h nyaris duplikat tapi tetap menyumbang 64 dimensi penuh; attention bisa mengompresi redundansi ini).
- **Topological Data Analysis (TDA)** pada sliding-window point cloud — karena struktur sudah terbukti berupa trajectory kontinu (Perea & Harer), persistent homology bisa mengukur "bentuk" manifold market-state secara formal, memberi metrik baru yang lebih cocok daripada silhouette/DBI untuk data non-blob.

### C. Perluasan riset (framing baru, potensi paper terpisah)
- **Studi "information dial"**: karena jumlah timeframe berperan sebagai parameter kontrol tunggal (INV-001) yang memetakan langsung ke geometri terukur (dimensi intrinsik, distance-CV), ini bisa diperluas jadi studi sistematis "bagaimana representation geometry berevolusi seiring skala ditambah" — general untuk time-series SSL, tidak terbatas pada crypto (Framing A di dissection report).
- **Uji ulang dengan output_dim TS2Vec berbeda** (32/128/256, bukan hanya 64) — dissection report memprediksi trade-off dimensi-vs-clusterability yang sama akan muncul *dalam satu branch*, bukan hanya lintas kondisi; ini eksperimen terkontrol yang murah untuk memvalidasi teori alignment/uniformity secara langsung.
- **Bandingkan fusion strategy**: concat+JL-projection (saat ini) vs learned attention vs weighted-concat berbasis CCA-similarity — mengukur mana yang menghasilkan intrinsic-dim paling efisien per bit informasi ekonomi yang didapat.

### Prioritas realistis
Kalau tujuan jangka pendek adalah **menyelesaikan M10→M11→paper**, jalur **A** cukup dan sudah terverifikasi kuat oleh bukti. Jalur **B/C** adalah *bonus scientific value* — cocok dieksplorasi setelah pipeline utama selesai, atau dijadikan kontribusi tambahan/paper kedua, memanfaatkan infrastruktur (checkpoint, embedding, diagnostik) yang sudah ada dan tidak perlu dibangun ulang.

---

*Rujukan bukti: `experiments/m10_exploration/*.csv`, probe cross-branch CCA/PCA (sesi ini), `docs/PIPELINE_SCIENTIFIC_DISSECTION.md` Stage 2/5/8.*
