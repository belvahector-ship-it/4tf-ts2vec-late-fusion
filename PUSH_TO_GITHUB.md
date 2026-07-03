# PUSH_TO_GITHUB.md

Panduan push repo ini ke GitHub untuk pertama kali, sebelum membuka
Claude Code.

## Langkah

1. **Buat repo kosong di GitHub** (jangan centang "Initialize with
   README" — biarkan benar-benar kosong), catat URL-nya, misal:
   `https://github.com/USERNAME/market-state-discovery.git`

2. **Extract zip yang diberikan**, lalu jalankan di terminal:

```bash
cd market-state-discovery   # folder hasil extract zip

git init
git add .
git commit -m "M0-M6 complete: data pipeline (Claude.ai sessions 1-7, TS2Vec commit pinned)"
git branch -M main
git remote add origin https://github.com/USERNAME/market-state-discovery.git
git push -u origin main
```

3. **Verifikasi**: buka URL repo Anda di browser, pastikan semua
   folder (`src/`, `tests/`, `configs/`, `docs/`, dst.) muncul.

## Setelah push berhasil

Buka Claude Code di folder yang sama:

```bash
cd market-state-discovery
claude
```

Lalu ketik prompt ini persis (atau modifikasi sesuai kebutuhan):

```
Baca MIGRATION_TO_CLAUDE_CODE.md terlebih dahulu sebelum melakukan apa
pun. Ikuti instruksinya secara berurutan: jalankan setup_and_verify.sh
untuk memasang dependency dan menjalankan baseline test M0-M6, laporkan
hasilnya ke saya, baru kita putuskan langkah berikutnya (kemungkinan
besar lanjut ke M7 sesuai docs/IMP-01_v1.3.md).
```

## Catatan tentang file besar / .gitignore

`.gitignore` di repo ini sudah dikonfigurasi untuk mengecualikan
`data/`, `checkpoints/`, `experiments/`, `outputs/`, `logs/` dari git
(lihat isi file `.gitignore`) — jadi push pertama ini seharusnya kecil
dan cepat (~250KB, murni kode+dokumen). Folder-folder besar (dataset
Binance, model checkpoint, dst.) akan terisi nanti seiring modul
M1/M8/dst dijalankan sungguhan, dan tetap tidak ikut ter-push ke GitHub
kecuali Anda sengaja mengubah `.gitignore`.

## Kalau Anda tidak familiar dengan command line

Pakai **GitHub Desktop** (aplikasi GUI, gratis):
1. Install dari https://desktop.github.com
2. "Add Local Repository" → pilih folder hasil extract zip
3. Klik "Publish repository"
4. Selesai — tidak perlu ketik command apa pun

Setelah itu, Claude Code tetap bisa Anda buka dari folder yang sama
seperti langkah di atas.
