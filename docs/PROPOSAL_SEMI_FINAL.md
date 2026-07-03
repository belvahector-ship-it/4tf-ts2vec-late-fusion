> **STATUS: PROPOSAL SEMI-FINAL — Snapshot Resmi Sesi 26 (Stop Point)**
> **Ini adalah representasi JALUR B (Riset Penuh menuju SINTA 1/2/3)** — bukan versi ringkas 6-halaman. Editing dihentikan sementara atas permintaan eksplisit penulis untuk menguji hasil eksperimen aktual sebelum melanjutkan revisi proposal.
>
> **Cakupan revisi yang sudah masuk ke snapshot ini:**
> - Fase 0-9: Fondasi, Audit, Abstract/Keywords, Bab I-III lengkap, Expected Contributions, Jadwal, Referensi awal, Final Assembly pertama
> - Fase 7B: 23 referensi awal diverifikasi Scopus/DOI (8 kesalahan ditemukan & diperbaiki, termasuk 1 kasus fabrikasi identitas penulis "Karim et al." yang ternyata tidak pernah ada)
> - Fase 10: Blok 3 & 7 diperkuat (argumen struktural + triangulasi evolusioner), fix nama "Abu Hashish"
> - Fase 12: 6 subbab diperkuat dengan Master Kompilasi Literatur (262 entri corpus), 6 referensi baru ditambahkan dengan tag transparansi "belum diverifikasi independen"
> - Fase 13 (task 13.1-13.3): JL Lemma direframe jujur (bukan klaim jaminan formal, tapi argumen kontrol eksperimen netral), n=5 power diakui eksplisit di Keterbatasan, housekeeping status leftover dibersihkan
>
> **Total referensi:** 27 entri (21 terverifikasi penuh Fase 7B + 6 baru Fase 12 belum diverifikasi independen)
> **Total panjang:** ~2680 baris markdown (setara ~25-30 halaman dokumen terformat)
>
> **TIDAK termasuk dalam snapshot ini:** hasil eksperimen aktual (proposal ini masih tahap desain/proposal, belum ada data hasil training/clustering nyata), proceeding 6 halaman (Jalur A, terpisah, belum dikerjakan).
>
> **Item terbuka yang butuh keputusan/tindakan penulis:**
> 1. Klarifikasi apakah AUDIT_CHECKLIST_PROPOSAL.md eksternal menemukan mismatch sitasi independen dari yang sudah diperbaiki Fase 7B (pertanyaan 11.3, masih menggantung)
> 2. 6 referensi baru Fase 12 opsional untuk diverifikasi Scopus/DOI independen (pola kerja Fase 7B)
> 3. Tabel 2.1 opsional diperluas ke 10 entri (catatan asli penulis sendiri)
>
> **Lihat `10_CHECKPOINT_SESI26.md` untuk ringkasan status lengkap dan rencana kerja 2-jalur (Proceeding 6-halaman vs Riset Penuh).**

---

# Multi-Resolution Temporal Representation Learning for Self-Supervised Cryptocurrency Market State Discovery: A Controlled Empirical Study

**Penulis:** Belva Fahrozi Chiangmaitri (P31202502702) **Lembaga:**
MTI_48, Universitas Dian Nuswantoro

## Abstrak

yang saling melengkapi: resolusi yang lebih halus menangkap fluktuasi
jangka pendek, sedangkan resolusi yang lebih kasar menangkap tren dan
rezim struktural. Di sisi lain, metode pembelajaran representasi
*self-supervised* seperti TS2Vec telah menunjukkan performa yang kuat
pada *benchmark* deret waktu umum, namun belum dievaluasi dengan
resolusi temporal sebagai variabel eksperimen yang dikontrol, dan
penelitian *multi-timeframe* finansial yang ada masih didominasi
orientasi *forecasting*, bukan pembelajaran representasi. Kesenjangan
ini melandasi sebuah *controlled empirical study* yang mengisolasi
resolusi temporal sebagai satu-satunya variabel independen pada empat
kondisi eksperimen (1TF--4TF), menggunakan encoder TS2Vec yang dilatih
secara independen dan digabungkan melalui *late concatenation fusion*
berdimensi tetap. Representasi laten yang dihasilkan dikelompokkan
menggunakan HDBSCAN untuk menemukan *market state*, dievaluasi melalui
kualitas *clustering* geometrik (*Silhouette Score*, *Davies-Bouldin
Index*, *Calinski-Harabasz Index*) maupun validitas ekonomi (distribusi
*return* dan profil risiko tiap *cluster*). Pengujian hipotesis
mengikuti protokol *pre-registered* dengan tiga perbandingan berpasangan
(2TF, 3TF, 4TF terhadap baseline 1TF) menggunakan uji Wilcoxon
*signed-rank* satu sisi dengan koreksi Holm-Bonferroni pada lima *random
seed*. Penelitian ini tidak mengusulkan arsitektur model baru; tujuannya
adalah memberikan bukti empiris mengenai kontribusi input temporal
multi-resolusi terhadap kualitas representasi *market state*
*self-supervised*, sekaligus menyediakan *baseline* reproduktif bagi
penelitian lanjutan dalam pembelajaran representasi deret waktu
finansial.

**Kata Kunci:** pembelajaran mandiri (*self-supervised learning*);
pembelajaran representasi; deret waktu multi-resolusi; penemuan kondisi

## Abstract

Cryptocurrency markets operate continuously and exhibit substantially
higher volatility than conventional asset classes, making *market state*
--- the latent condition describing price behavior over a given period
--- a central construct for risk management, anomaly detection, and
investment decision-making. Most existing approaches to market state
discovery rely on a single temporal resolution, despite evidence that
different timeframes carry complementary information: finer resolutions
capture short-term fluctuations, while coarser resolutions capture
structural trends and regimes. Meanwhile, self-supervised representation
learning methods such as TS2Vec have demonstrated strong performance on
general time-series benchmarks, but have not been evaluated with
temporal resolution treated as a controlled experimental variable, and
existing multi-timeframe financial studies remain predominantly
forecasting-oriented rather than representation-oriented. This gap
motivates a controlled empirical study that isolates temporal resolution
as the sole independent variable across four experimental conditions
(1TF--4TF), using independently trained TS2Vec encoders combined through
fixed-dimension late concatenation fusion. The resulting latent
representations are clustered using HDBSCAN to discover market states,
evaluated through both geometric clustering quality (Silhouette Score,
Davies-Bouldin Index, Calinski-Harabasz Index) and economic validity
(return distribution and risk profile per cluster). Hypothesis testing
follows a pre-registered protocol with three pairwise comparisons (2TF,
3TF, 4TF vs. 1TF baseline) using one-sided Wilcoxon signed-rank tests
with Holm-Bonferroni correction across five random seeds. This study
does not propose a new model architecture; instead, it aims to provide
empirical evidence on whether multi-resolution temporal input
meaningfully improves the quality of self-supervised market state
representations, offering a reproducible baseline for future research in
financial time-series representation learning.

**Keywords:** self-supervised learning; representation learning;
multi-resolution time series; market state discovery; cryptocurrency;
contrastive learning; clustering validation

# BAB I

## PENDAHULUAN

### 1.1 Latar Belakang

Cryptocurrency telah berkembang menjadi salah satu kelas aset yang
menarik perhatian investor, institusi keuangan, maupun peneliti dalam
beberapa tahun terakhir. Berbeda dengan pasar saham yang memiliki jam
perdagangan terbatas, pasar aset kripto beroperasi selama 24 jam tanpa
henti dengan karakteristik volatilitas yang relatif lebih tinggi
dibandingkan kelas aset konvensional (Koki, Leonardos, & Piliouras,
2022). Pergerakan harga dapat berubah secara cepat sebagai respons
terhadap berbagai faktor, seperti kondisi makroekonomi, sentimen pasar,
perkembangan teknologi, maupun dinamika permintaan dan penawaran.
Karakteristik tersebut menjadikan analisis perilaku pasar cryptocurrency
lebih kompleks dibandingkan banyak kelas aset lainnya, sehingga
mendorong berkembangnya berbagai penelitian untuk memahami dinamika
pasar secara lebih baik.

Salah satu konsep yang banyak digunakan untuk menjelaskan dinamika
tersebut adalah *market state* atau kondisi pasar. Secara umum, *market
state* menggambarkan karakteristik perilaku pasar pada periode tertentu,
seperti kondisi *bullish*, *bearish*, *sideways*, maupun volatilitas
tinggi, dan secara klasik dimodelkan sebagai variabel laten yang berubah
mengikuti probabilitas transisi (Hamilton, 1989). Informasi mengenai
kondisi pasar tidak hanya bermanfaat bagi investor dalam memahami
perubahan tren, tetapi juga menjadi fondasi bagi berbagai sistem
analisis berbasis kecerdasan buatan, seperti prediksi harga, manajemen
risiko, deteksi anomali, hingga pengambilan keputusan investasi (Giudici
& Abu Hashish, 2020). Oleh karena itu, kemampuan memperoleh representasi
kondisi pasar yang mampu menggambarkan karakteristik pergerakan harga
secara akurat menjadi salah satu topik yang terus berkembang dalam
penelitian pasar keuangan.

Berbagai penelitian telah mengembangkan metode untuk mengidentifikasi
kondisi pasar, mulai dari pendekatan statistik seperti *Hidden Markov
Model* (HMM) (Hamilton, 1989; Koki et al., 2022), metode *clustering*
(Bucci & Ciciretti, 2022), hingga pendekatan *deep learning* yang mampu
mempelajari pola secara otomatis dari data historis. Meskipun
menunjukkan hasil yang menjanjikan, sebagian besar pendekatan tersebut
masih membangun representasi pasar menggunakan satu resolusi temporal
(*single timeframe*) (Sobreiro et al., 2026). Keterbatasan pendekatan
*single-timeframe* bersifat struktural, bukan sekadar preferensi
metodologis: setiap proses agregasi *candle* ke resolusi tertentu ---
misalnya penggabungan data 15 menit menjadi 1 jam --- secara inheren
membuang derajat kebebasan (*degrees of freedom*) yang terkandung pada
resolusi yang lebih halus, sementara resolusi yang lebih halus itu
sendiri tidak memiliki jendela pengamatan yang cukup panjang untuk
menangkap pola tren dan musiman yang baru terlihat pada skala waktu
yang lebih kasar. Dengan kata lain, representasi satu-resolusi secara
matematis hanya mampu mengakses satu titik pada spektrum
frekuensi-temporal pasar, sedangkan perilaku harga cryptocurrency
tersusun atas komponen dengan periodisitas berbeda yang saling tumpang
tindih (Sobreiro et al., 2026). Konsekuensinya, representasi yang
dibangun dari satu *timeframe* berpotensi kehilangan informasi yang
hanya teramati pada skala waktu lain, terlepas dari seberapa baik model
dilatih pada resolusi tersebut.

Bukti empiris pada domain finansial memperkuat argumen struktural
tersebut: pengujian langsung terhadap beberapa resolusi temporal secara
konsisten menunjukkan bahwa performa model bersifat *bergantung
horizon* (*horizon-dependent*) --- tidak ada satu resolusi tunggal yang
unggul secara universal di seluruh kondisi pasar, dan kombinasi
beberapa resolusi umumnya mengungguli *baseline* satu-skala ketika
diuji secara langsung pada data yang sama (Sobreiro et al., 2026).
Pola ini juga konsisten pada studi *multi-timeframe* lain: perbandingan
pada horizon 15 menit hingga 1 hari melaporkan bahwa model *tree-based*
unggul pada resolusi pendek sementara model berbasis dekomposisi
musiman lebih kompetitif pada resolusi harian (Ennagoura et al., 2026)
[SUMBER: MASTER_KOMPILASI, belum diverifikasi Scopus/DOI]; perbandingan
pada horizon 30 menit hingga 4 jam melaporkan performa terbaik justru
pada resolusi menengah (Khurana, Singh, & Garg, 2023) [SUMBER:
MASTER_KOMPILASI, belum diverifikasi Scopus/DOI]; dan perbandingan
antara skala harian tunggal dengan kombinasi harian-mingguan melaporkan
bahwa pendekatan multi-skala secara konsisten mengungguli pendekatan
satu-skala pada metrik risiko-imbal hasil (Huang, Song, & Chen, 2023)
[SUMBER: MASTER_KOMPILASI, belum diverifikasi Scopus/DOI]. Pola yang
berulang pada studi-studi independen ini --- performa terbaik berpindah
tergantung horizon yang diuji, dan kombinasi resolusi secara konsisten
mengungguli resolusi tunggal saat dibandingkan langsung --- memperkuat
argumen bahwa satu resolusi temporal tunggal tidak cukup untuk
menangkap dinamika pasar secara utuh. Kesenjangan antara kebutuhan
menangkap dinamika pasar dari berbagai horizon waktu dan pendekatan
yang masih didominasi oleh *single timeframe* menunjukkan bahwa masih
terdapat ruang penelitian untuk mengevaluasi bagaimana informasi dari
beberapa resolusi temporal memengaruhi kualitas representasi kondisi
pasar --- bukan hanya performa prediksi seperti pada penelitian
*forecasting* yang telah ada, melainkan kualitas representasi laten itu
sendiri.

Perkembangan *deep learning* mendorong munculnya pendekatan
*representation learning*, yaitu metode yang berupaya mempelajari
representasi data secara otomatis tanpa bergantung pada rekayasa fitur
(*feature engineering*) yang kompleks. Pada domain deret waktu (*time
series*), pendekatan ini dinilai mampu menangkap pola temporal yang
sulit diperoleh melalui metode konvensional. Salah satu perkembangan
penting dalam bidang tersebut adalah *Self-Supervised Learning* (SSL),
yaitu paradigma pembelajaran yang memanfaatkan struktur alami data
sebagai sumber supervisi sehingga tidak memerlukan proses pelabelan
secara manual (Van den Oord, Li, & Vinyals, 2018). Pendekatan ini
menjadi semakin relevan karena sebagian besar data deret waktu tersedia
dalam jumlah besar, tetapi tidak memiliki label yang memadai.

Di antara berbagai metode *Self-Supervised Learning*, TS2Vec merupakan
salah satu pendekatan yang banyak digunakan karena mampu mempelajari
representasi deret waktu yang bersifat umum (*universal representation*)
melalui mekanisme *hierarchical contextual contrastive learning* (Yue et
al., 2022). Representasi yang dihasilkan terbukti dapat digunakan pada
berbagai tugas hilir (*downstream tasks*), seperti klasifikasi,
*forecasting*, deteksi anomali, maupun *clustering* (Yue et al., 2022).
Keberhasilan tersebut menunjukkan bahwa kualitas representasi laten
memiliki peran penting dalam meningkatkan performa berbagai metode
analisis deret waktu. Namun demikian, penelitian TS2Vec maupun sebagian
besar penelitian SSL lainnya masih berfokus pada pengembangan metode
pembelajaran representasi, sementara pengaruh variasi resolusi temporal
sebagai sumber informasi masukan belum dievaluasi secara khusus.
Sintesis literatur yang telah dilakukan juga menunjukkan bahwa
penelitian *multi-timeframe* pada domain keuangan umumnya berorientasi
pada tugas *forecasting* (Sobreiro et al., 2026), bukan pada
pembelajaran representasi (*representation learning*).

Berdasarkan kondisi tersebut, masih terdapat kesenjangan penelitian
mengenai sejauh mana kombinasi beberapa resolusi temporal mampu
meningkatkan kualitas representasi laten yang dipelajari oleh model
*Self-Supervised Learning*. Pertanyaan ini menjadi penting karena
apabila informasi dari berbagai *timeframe* memang memberikan kontribusi
yang signifikan terhadap kualitas representasi, maka temuan tersebut
dapat menjadi dasar dalam pengembangan model analisis pasar yang lebih
efektif. Sebaliknya, apabila penambahan resolusi temporal tidak
memberikan peningkatan yang berarti, maka penggunaan satu *timeframe*
saja dapat menjadi pilihan yang lebih sederhana dan efisien --- sebuah
kemungkinan yang didukung oleh bukti bahwa representasi satu-resolusi
dengan bobot yang dibagikan antar-*channel* dapat memberikan performa
kompetitif pada sejumlah tugas (Nie, Nguyen, Sinthong, & Kalagnanam,
2023). Dengan demikian, penelitian ini tidak berfokus pada pengembangan
arsitektur baru, melainkan pada investigasi empiris terhadap kontribusi
resolusi temporal dalam pembelajaran representasi pasar.

Penelitian ini mengusulkan pendekatan *Multi-Branch TS2Vec* dengan skema
*Fixed-Dimension Late Fusion* untuk mengevaluasi pengaruh penggunaan
satu hingga beberapa resolusi temporal terhadap kualitas representasi
pasar cryptocurrency. Seluruh kondisi eksperimen dirancang menggunakan
arsitektur, dimensi representasi, dan prosedur evaluasi yang sama
sehingga resolusi temporal menjadi satu-satunya variabel yang diamati.
Representasi laten yang dihasilkan selanjutnya digunakan untuk proses
*market state discovery* menggunakan HDBSCAN (Campello, Moulavi, &
Sander, 2013; McInnes, Healy, & Astels, 2017) dan dievaluasi berdasarkan
kualitas *clustering* serta validitas ekonomi dari setiap kondisi yang
dihasilkan. Rancangan tersebut disusun sebagai *controlled empirical
study* sehingga diharapkan mampu memberikan bukti empiris mengenai
kontribusi resolusi temporal terhadap pembelajaran representasi pasar
cryptocurrency.

Melalui penelitian ini diharapkan dapat diperoleh pemahaman yang lebih
baik mengenai hubungan antara resolusi temporal dan kualitas
representasi laten pada data pasar cryptocurrency. Selain memberikan
bukti empiris terhadap efektivitas penggunaan *static multi-resolution
temporal input*, penelitian ini diharapkan dapat menjadi dasar bagi
pengembangan penelitian lanjutan di bidang *market state representation*
maupun *representation learning* pada data deret waktu keuangan.

### 1.2 Identifikasi Masalah

Berdasarkan latar belakang yang telah diuraikan, penelitian ini
mengidentifikasi permasalahan yang masih menjadi perhatian dalam
pembelajaran representasi (*representation learning*) pada data pasar
cryptocurrency. Identifikasi dibedakan antara karakteristik domain yang
melatarbelakangi kebutuhan penelitian dan kesenjangan literatur yang
menjadi dasar kontribusi penelitian ini.

**Karakteristik domain yang melatarbelakangi kebutuhan penelitian:**

1.  Karakteristik pasar cryptocurrency yang memiliki volatilitas tinggi
    dan beroperasi selama 24 jam menyebabkan perubahan kondisi pasar
    berlangsung secara dinamis. Kondisi tersebut menuntut representasi
    data yang mampu menggambarkan perilaku pasar secara lebih
    komprehensif dari berbagai horizon waktu pengamatan.

**Kesenjangan literatur:**

2.  Sebagian besar penelitian *market state discovery* masih membangun
    representasi pasar menggunakan satu resolusi temporal (*single
    timeframe*), sehingga informasi yang diperoleh berpotensi belum
    sepenuhnya menggambarkan dinamika pasar dari berbagai horizon waktu.

3.  Penelitian mengenai *Self-Supervised Learning*, khususnya TS2Vec,
    telah menunjukkan kemampuan yang baik dalam mempelajari representasi
    deret waktu. Namun, penelitian tersebut lebih berfokus pada
    pengembangan metode pembelajaran representasi dan belum mengevaluasi
    pengaruh penggunaan beberapa resolusi temporal terhadap kualitas
    representasi yang dihasilkan.

4.  Penelitian *multi-timeframe* pada domain keuangan umumnya diarahkan
    untuk meningkatkan akurasi *forecasting*, sedangkan kajian mengenai
    kontribusi *multi-resolution temporal input* terhadap kualitas
    representasi laten untuk *market state discovery* masih relatif
    terbatas. Akibatnya, belum terdapat bukti empiris yang menunjukkan
    apakah penambahan jumlah resolusi temporal benar-benar meningkatkan
    kualitas representasi laten, atau justru menambah informasi yang
    tidak memberikan kontribusi signifikan terhadap proses tersebut.

Ketiga kesenjangan literatur di atas (poin 2--4) secara langsung menjadi
dasar perumusan pertanyaan penelitian pada Subbab 1.4, sedangkan
karakteristik domain (poin 1) menjadi konteks yang menjustifikasi
relevansi praktis dari penelitian ini.

Berdasarkan identifikasi tersebut, penelitian ini berfokus untuk
menginvestigasi pengaruh penggunaan resolusi temporal secara bertahap
terhadap kualitas representasi laten yang dipelajari oleh model
*Self-Supervised Learning*. Penelitian dilakukan menggunakan protokol
eksperimen yang terkontrol sehingga perubahan kualitas representasi
dapat dikaitkan dengan variasi resolusi temporal sebagai variabel utama
penelitian.

### 1.3 Research Gap

Perkembangan *Self-Supervised Learning* telah memberikan kemajuan yang
signifikan dalam pembelajaran representasi data deret waktu. Berbagai
metode seperti *Contrastive Predictive Coding* (CPC) (Van den Oord, Li,
& Vinyals, 2018), *Temporal Neighborhood Coding* (TNC) (Tonekaboni,
Eytan, & Goldenberg, 2021), TS-TCC (Eldele et al., 2021), CoST (Woo,
Liu, Sahoo, Kumar, & Hoi, 2022), hingga TS2Vec (Yue et al., 2022)
menunjukkan bahwa representasi laten yang dipelajari secara mandiri
mampu mendukung berbagai *downstream task*, seperti klasifikasi,
*forecasting*, deteksi anomali, maupun *clustering*. Di antara metode
tersebut, TS2Vec menjadi salah satu pendekatan yang banyak digunakan
karena mampu menghasilkan *universal time series representation* melalui
mekanisme *hierarchical contextual contrastive learning* (Yue et al.,
2022).

Meskipun demikian, sebagaimana dirangkum pada Tabel 2.1 (Subbab 2.8),
sebagian besar penelitian tersebut berfokus pada pengembangan arsitektur
model dan strategi pembelajaran representasi, serta divalidasi pada
*benchmark* domain umum seperti data manusia (UCR/HAR) atau deret waktu
non-finansial (Eldele et al., 2021; Woo et al., 2022). Variasi resolusi
temporal umumnya diperlakukan sebagai bagian dari karakteristik dataset,
bukan sebagai variabel penelitian yang dievaluasi secara sistematis ---
termasuk pada TS2Vec sendiri, yang tidak diuji pada data cryptocurrency
maupun pada variasi resolusi temporal sebagai variabel terkontrol (Yue
et al., 2022). Dengan kata lain, penelitian terdahulu lebih banyak
menjawab *bagaimana* membangun representasi yang baik, tetapi belum
secara khusus mengkaji faktor *apa* yang memengaruhi kualitas
representasi tersebut, khususnya dari aspek resolusi temporal.

Di sisi lain, penelitian *multi-timeframe* pada domain keuangan telah
menunjukkan bahwa informasi dari beberapa horizon waktu dapat
meningkatkan pemahaman terhadap perilaku pasar (Sobreiro et al., 2026).
Namun, sebagian besar penelitian tersebut berorientasi pada peningkatan
performa *forecasting* atau prediksi harga menggunakan fitur rekayasa
manual, bukan pada pembelajaran representasi laten untuk *market state
discovery* (Sobreiro et al., 2026). Metode identifikasi kondisi pasar
yang mapan pada domain finansial, khususnya HMM, di sisi lain tidak
memanfaatkan kerangka *representation learning* modern dan tidak
mengintegrasikan berbagai resolusi temporal secara sistematis (Koki,
Leonardos, & Piliouras, 2022). Akibatnya, masih belum tersedia bukti
empiris yang menjelaskan apakah penggunaan beberapa resolusi temporal
benar-benar memberikan kontribusi terhadap kualitas representasi yang
dipelajari oleh model *Self-Supervised Learning*.

Kedua paragraf sebelumnya menggambarkan dua jalur perkembangan yang
secara historis berjalan terpisah. Jalur pertama --- *representation
learning* berbasis SSL (CPC, TNC, TS-TCC, CoST, hingga TS2Vec) ---
berkembang dari kebutuhan mempelajari struktur laten deret waktu secara
umum tanpa label, namun divalidasi pada *benchmark* domain generik yang
tidak memiliki struktur multi-resolusi pasar finansial sebagai bagian
dari desain eksperimennya. Jalur kedua --- pemodelan kondisi pasar
finansial (HMM) dan analisis *multi-timeframe* --- berkembang dari
kebutuhan praktis industri keuangan untuk menangkap dinamika pasar dari
berbagai horizon waktu, namun tetap bertumpu pada representasi
statistik klasik atau fitur rekayasa manual, tanpa memanfaatkan
kerangka *representation learning* modern. Kedua jalur ini berkembang
menuju arah yang saling melengkapi namun belum pernah bertemu secara
eksplisit: jalur pertama memiliki metode pembelajaran representasi yang
matang tetapi belum diuji pada struktur multi-resolusi pasar finansial,
sedangkan jalur kedua memiliki kebutuhan multi-resolusi yang jelas
tetapi belum menggunakan representasi yang dipelajari secara
*self-supervised*.

Titik temu yang belum tergarap ini bukan sekadar celah observasional,
melainkan konsekuensi logis dari perbedaan orientasi kedua jalur
tersebut: penelitian *representation learning* mengoptimalkan
*bagaimana* representasi dipelajari (arsitektur, fungsi objektif,
strategi augmentasi), sedangkan penelitian *multi-timeframe* finansial
mengoptimalkan *akurasi prediksi* pada tugas hilir tertentu. Belum ada
penelitian yang memosisikan resolusi temporal itu sendiri sebagai
variabel penelitian yang diuji pengaruhnya terhadap kualitas
representasi laten melalui protokol eksperimen yang terkontrol --- bukan
sebagai karakteristik dataset yang *given*, dan bukan pula sebagai
fitur tambahan untuk meningkatkan akurasi *forecasting*. Berdasarkan
sintesis tersebut (uraian lengkap pada Tabel 2.1 dan Tabel 2.2, Subbab
2.8--2.9), penelitian ini mengidentifikasi kesenjangan pada titik temu
kedua jalur perkembangan tersebut: pemanfaatan resolusi temporal
sebagai variabel independen yang dievaluasi secara sistematis dalam
pembelajaran representasi *self-supervised* untuk *market state
discovery*. Akibatnya, pengaruh penggunaan satu maupun beberapa
resolusi temporal terhadap kualitas representasi laten masih belum
dapat dijelaskan secara empiris.

Kesenjangan ini tidak berdiri sebagai simpulan tunggal, melainkan
konvergen dari beberapa jalur pembuktian yang independen satu sama
lain. Penelusuran literatur *representation learning* untuk deret waktu
menunjukkan pola perkembangan lima tahap yang konsisten --- dari
representasi prediktif berbasis *contrastive loss* (CPC), representasi
sadar-temporal untuk data non-stasioner (TNC), representasi sadar-konteks
dengan augmentasi khusus deret waktu (TS-TCC), representasi sadar-struktur
yang mendisentangle komponen musiman-tren (CoST), hingga representasi
universal lintas-tugas (TS2Vec) --- sebagaimana telah dirujuk pada
paragraf pertama subbab ini. Pada setiap tahap tersebut, resolusi
temporal secara konsisten diperlakukan sebagai karakteristik bawaan
dataset, bukan variabel yang dimanipulasi secara terkontrol untuk diuji
pengaruhnya. Pada domain yang lebih spesifik pada deret waktu
multi-resolusi, sejumlah studi independen secara konsisten melaporkan
bahwa performa model bersifat *bergantung horizon* --- tidak ada satu
resolusi tunggal yang unggul secara universal, dan kombinasi beberapa
resolusi umumnya mengungguli pendekatan satu-skala ketika diuji langsung
pada data yang sama, sebagaimana telah diuraikan pada Subbab 1.1. Namun,
mekanisme baku untuk menggabungkan representasi dari beberapa resolusi
temporal secara optimal juga belum tersedia dalam literatur yang ada,
menunjukkan bahwa ruang penelitian ini masih terbuka baik dari sisi
*apakah* multi-resolusi memberi kontribusi maupun *bagaimana*
representasi tersebut sebaiknya digabungkan.

Konvergensi ini juga tampak pada domain regime pasar: metode
identifikasi kondisi pasar yang mapan secara statistik (HMM dan
turunannya) belum memanfaatkan kerangka *representation learning*
modern (Koki et al., 2022), sementara metode *clustering* untuk
*market state discovery* --- meskipun metodologinya telah matang secara
teknis --- belum banyak diuji pada representasi laten hasil SSL
multi-resolusi. Analisis cakupan-bukti (*evidence-coverage analysis*)
yang menelusuri irisan antara literatur SSL, representasi regime pasar,
dan pemodelan multi-timeframe secara konsisten menemukan bahwa sel
irisan "SSL Encoders × Market Regime" dan "Integrated Multi-Timeframe
Market Representation" masih kosong pada hampir seluruh kombinasi yang
ditelusuri [Additional Reference Needed --- temuan ini berasal dari
analisis sintesis penulis sendiri terhadap korpus literatur yang
dikumpulkan, bukan dari satu artikel primer tunggal; perlu dicantumkan
sebagai catatan metodologis pada Lampiran atau dijadikan bagian dari
deskripsi metode tinjauan pustaka di Subbab 2.8 apabila diperlukan
transparansi lebih lanjut]. Ketiga jalur pembuktian ini --- evolusi
metodologi *representation
learning*, bukti empiris *horizon-dependent performance*, dan
kesenjangan pada domain *market state discovery* --- secara independen
mengarah pada kesimpulan yang sama, sehingga kesenjangan yang diklaim
penelitian ini bukan sekadar kombinasi belum-dicoba antara dua bidang
yang berdiri sendiri, melainkan titik temu yang secara konsisten belum
tergarap dari berbagai sudut pandang literatur yang berbeda.

Penelitian ini berupaya mengisi kesenjangan tersebut melalui *controlled
empirical study* dengan mengevaluasi penggunaan satu hingga beberapa
resolusi temporal menggunakan arsitektur dan prosedur eksperimen yang
identik. Dengan pendekatan tersebut, penelitian ini diharapkan mampu
memberikan bukti empiris mengenai kontribusi resolusi temporal terhadap
kualitas representasi laten pada *market state discovery*, sekaligus
menyediakan dasar bagi penelitian lanjutan di bidang *market state
representation* pada data pasar cryptocurrency.

### 1.4 Rumusan Masalah

Berdasarkan latar belakang, identifikasi masalah, dan *research gap*
yang telah diuraikan, penelitian ini dirumuskan dalam beberapa
pertanyaan penelitian sebagai berikut.

1.  Apakah penggunaan beberapa resolusi temporal (*static
    multi-resolution temporal input*) dapat meningkatkan kualitas
    representasi laten yang dipelajari oleh model *Self-Supervised
    Learning* dibandingkan penggunaan satu resolusi temporal
    (*single-resolution*) pada *market state discovery* cryptocurrency?

2.  Bagaimana pengaruh jumlah resolusi temporal terhadap kualitas dan
    stabilitas representasi laten yang dihasilkan oleh model
    *Self-Supervised Learning*?

3.  Apakah *market state* yang diperoleh dari representasi laten
    tersebut memiliki karakteristik ekonomi yang berbeda secara
    signifikan berdasarkan distribusi *return* dan profil risikonya?

Ketiga rumusan masalah tersebut menjadi dasar dalam penyusunan metode
penelitian dan desain eksperimen. Penelitian ini tidak berfokus pada
pengembangan arsitektur *deep learning* baru, melainkan pada evaluasi
empiris terhadap pengaruh resolusi temporal sebagai variabel penelitian
dalam pembelajaran representasi pasar cryptocurrency.

### 1.5 Tujuan Penelitian

Berdasarkan rumusan masalah yang telah disusun, penelitian ini bertujuan
untuk:

1.  Menganalisis pengaruh penggunaan beberapa resolusi temporal (*static
    multi-resolution temporal input*) terhadap kualitas representasi
    laten yang dipelajari oleh model *Self-Supervised Learning* pada
    *market state discovery* cryptocurrency.

2.  Mengevaluasi hubungan antara jumlah resolusi temporal dan kualitas
    representasi laten berdasarkan metrik evaluasi *clustering*,
    meliputi *Silhouette Score*, *Davies-Bouldin Index* (DBI), dan
    *Calinski-Harabasz Index* (CH).

3.  Menganalisis apakah *market state* yang dihasilkan dari representasi
    laten memiliki karakteristik ekonomi yang berbeda secara signifikan
    berdasarkan distribusi *return* dan profil risiko masing-masing
    *cluster*.

4.  Memberikan bukti empiris mengenai kontribusi resolusi temporal
    terhadap pembelajaran representasi pasar cryptocurrency melalui
    protokol eksperimen yang terkontrol, menggunakan tiga perbandingan
    berpasangan yang telah ditetapkan sebelumnya (2TF, 3TF, dan 4TF
    terhadap baseline 1TF).

### 1.6 Kontribusi Penelitian

Penelitian ini diharapkan dapat memberikan kontribusi pada tiga tingkat:
konseptual, metodologis, dan empiris/praktis.

**Kontribusi Konseptual**

1.  Memberikan bukti empiris mengenai pengaruh penggunaan resolusi
    temporal terhadap kualitas representasi laten yang dipelajari oleh
    model *Self-Supervised Learning* pada *market state discovery*
    cryptocurrency --- memposisikan resolusi temporal sebagai variabel
    penelitian yang dievaluasi secara sistematis, bukan sekadar
    karakteristik dataset (lihat Subbab 1.3). Hasil penelitian ini
    diharapkan dapat memperkaya kajian *representation learning*,
    khususnya pada domain data deret waktu keuangan.

**Kontribusi Metodologis**

2.  Menyajikan protokol eksperimen yang terkontrol --- termasuk skema
    pengujian statistik *pre-registered* dengan tiga perbandingan
    berpasangan dan koreksi Holm-Bonferroni --- untuk mengevaluasi
    pengaruh resolusi temporal dengan menjaga faktor-faktor lain tetap
    konsisten. Pendekatan ini diharapkan dapat menjadi acuan bagi
    penelitian selanjutnya yang menginvestigasi faktor-faktor yang
    memengaruhi kualitas representasi data deret waktu.

3.  Menyediakan *baseline* yang bersifat reproduktif (*reproducible*)
    --- dengan protokol *random seed*, *split* data, dan hiperparameter
    yang terdokumentasi penuh (lihat Subbab 3.9) --- sehingga dapat
    dimanfaatkan sebagai dasar pengembangan penelitian pada bidang
    *market state representation* maupun *representation learning* di
    masa mendatang.

**Kontribusi Empiris dan Praktis**

4.  Memberikan bukti empiris mengenai kontribusi resolusi temporal
    terhadap pembelajaran representasi pasar cryptocurrency melalui
    protokol eksperimen yang terkontrol, menggunakan tiga perbandingan
    berpasangan yang telah ditetapkan sebelumnya (2TF, 3TF, dan 4TF
    terhadap baseline 1TF). Temuan ini diharapkan menjadi pertimbangan
    bagi peneliti maupun pengembang sistem analisis pasar dalam
    menentukan strategi penyusunan data masukan (*input representation*)
    pada model *Self-Supervised Learning*.

5.  Menyediakan representasi laten yang berpotensi menjadi landasan bagi
    berbagai *downstream task* di luar *market state discovery*, seperti
    *forecasting*, klasifikasi, maupun deteksi anomali. Pemanfaatan pada
    *downstream task* tersebut tidak menjadi bagian dari penelitian ini
    (lihat Subbab 1.7, Batasan Penelitian) namun diharapkan menjadi arah
    pengembangan penelitian selanjutnya.

### 1.7 Batasan Penelitian

Agar penelitian memiliki ruang lingkup yang jelas serta menjaga
konsistensi antar skenario eksperimen, maka penelitian ini dibatasi pada
beberapa aspek berikut.

1.  Penelitian menggunakan data historis BTC/USDT yang diperoleh dari
    Binance sebagai representasi pasar cryptocurrency.

2.  Periode penelitian dibatasi pada data 1 Januari 2020 hingga 31
    Desember 2023 dengan pembagian data *training* dan *testing*
    menggunakan pendekatan *walk-forward split*.

3.  Resolusi temporal yang digunakan terdiri atas 15 menit, 1 jam, 4
    jam, dan 1 hari. Kombinasi resolusi temporal disusun secara bertahap
    mulai dari satu hingga beberapa *timeframe* sesuai dengan skenario
    eksperimen yang telah ditetapkan.

4.  Data yang digunakan hanya berupa data OHLCV (*Open, High, Low,
    Close, Volume*) yang ditransformasikan menjadi fitur dasar tanpa
    menggunakan *technical indicator*. Pembatasan ini bertujuan untuk
    meminimalkan pengaruh informasi hasil *smoothing* maupun *windowing*
    sehingga representasi yang dipelajari model berasal dari
    karakteristik data pasar itu sendiri.

5.  Penelitian menggunakan satu pendekatan *Self-Supervised Learning*
    yang sama pada seluruh skenario eksperimen, sedangkan proses *market
    state discovery* dilakukan menggunakan metode *clustering* yang
    sama. Penelitian ini tidak membandingkan berbagai arsitektur
    pembelajaran representasi maupun berbagai algoritma *clustering*.

6.  Seluruh skenario eksperimen menggunakan prosedur *preprocessing*,
    arsitektur model, mekanisme pembelajaran representasi, proses
    *fusion*, serta konfigurasi pelatihan yang sama. Perbedaan antar
    skenario hanya terletak pada jumlah resolusi temporal yang digunakan
    sebagai data masukan, sehingga perubahan hasil pengujian dapat
    diinterpretasikan sebagai pengaruh resolusi temporal.

7.  Penelitian tidak berfokus pada pengembangan arsitektur *deep
    learning* baru maupun optimasi performa model, tetapi pada
    investigasi empiris mengenai pengaruh resolusi temporal terhadap
    kualitas representasi laten.

8.  Ruang lingkup penelitian dibatasi pada pembelajaran representasi
    (*representation learning*) dan *market state discovery*.
    Pemanfaatan representasi laten pada *forecasting*, klasifikasi,
    deteksi anomali, maupun *downstream task* lainnya tidak menjadi
    bagian dari penelitian ini, namun diharapkan dapat menjadi arah
    pengembangan penelitian selanjutnya.

9.  Proyeksi dimensi akhir menggunakan matriks acak tetap (*fixed random
    projection*, lihat Subbab 2.6) memetakan seluruh kondisi ke ruang
    256-dimensi yang sama. Karena dimensi masukan konkatenasi berbeda
    per kondisi (64/128/192/256), rank efektif representasi akhir turut
    berbeda meskipun dimensi nominalnya identik. Hal ini merupakan
    konsekuensi matematis yang melekat pada pendekatan *fixed random
    projection* dan bersifat konservatif terhadap H1 --- yaitu kondisi
    dengan resolusi tunggal memiliki ruang representasi efektif yang
    lebih kecil, sehingga jika kondisi multi-resolusi tetap unggul,
    temuan tersebut justru semakin kuat secara argumentatif.
    Keterbatasan ini didokumentasikan lebih lanjut sebagai bagian dari
    desain proyeksi pada Subbab 2.6.

# BAB II

## TINJAUAN PUSTAKA

### 2.1 Cryptocurrency Market

Cryptocurrency merupakan aset digital yang dibangun di atas teknologi
blockchain dan diperdagangkan melalui jaringan yang bersifat
terdesentralisasi. Berbeda dengan pasar keuangan konvensional yang
memiliki jam perdagangan terbatas, pasar cryptocurrency beroperasi
selama 24 jam setiap hari sehingga proses pembentukan harga berlangsung
secara kontinu (Koki, Leonardos, & Piliouras, 2022). Karakteristik
tersebut menyebabkan dinamika pasar berubah dengan cepat sebagai respons
terhadap berbagai faktor, seperti kondisi makroekonomi, sentimen
investor, perkembangan teknologi, regulasi, maupun aktivitas
perdagangan.

Selain beroperasi tanpa henti, pasar cryptocurrency juga dikenal
memiliki tingkat volatilitas yang relatif tinggi dibandingkan banyak
kelas aset lainnya (Koki et al., 2022; Giudici & Abu Hashish, 2020).
Perubahan harga yang terjadi dalam waktu singkat menghasilkan pola
pergerakan yang kompleks dan bervariasi pada berbagai horizon waktu.
Kondisi ini menjadikan cryptocurrency sebagai salah satu objek yang
banyak digunakan untuk mengevaluasi berbagai metode analisis data deret
waktu, termasuk pembelajaran representasi (*representation learning*).
**\[Additional Reference Needed\]** --- klaim spesifik bahwa
cryptocurrency "banyak digunakan" sebagai objek uji *representation
learning* secara luas (di luar konteks *regime detection* yang sudah
dirujuk di atas) belum memiliki rujukan langsung di pool referensi saat
ini; disarankan penulis menambahkan 1 sitasi survei atau studi
*benchmark* yang secara eksplisit menggunakan data cryptocurrency untuk
*representation learning* umum (bukan *forecasting*).

Berbagai penelitian pada domain cryptocurrency umumnya berfokus pada
prediksi harga (*price forecasting*), manajemen risiko, deteksi anomali,
maupun identifikasi kondisi pasar (*market state*) (Sobreiro et al.,
2026; Koki et al., 2022). Dalam beberapa tahun terakhir, perhatian
penelitian mulai bergeser dari sekadar meningkatkan akurasi prediksi
menuju upaya membangun representasi data yang mampu menggambarkan
karakteristik pasar secara lebih komprehensif --- pergeseran yang
tercermin dari berkembangnya metode *representation learning* berbasis
*deep learning* seperti TS2Vec (Yue et al., 2022). Pergeseran tersebut
didasarkan pada pemahaman bahwa kualitas representasi data menjadi
fondasi bagi berbagai *downstream task*, seperti klasifikasi,
*forecasting*, deteksi anomali, dan *clustering* (Yue et al., 2022).

Pada penelitian ini, cryptocurrency dipilih sebagai objek penelitian
karena karakteristik pasar yang dinamis memberikan lingkungan yang
sesuai untuk mengevaluasi kualitas representasi laten yang dipelajari
oleh model *Self-Supervised Learning*. Fokus penelitian tidak diarahkan
pada prediksi harga maupun pengembangan strategi perdagangan, tetapi
pada analisis bagaimana representasi pasar dibentuk dari informasi yang
berasal dari berbagai resolusi temporal.

### 2.2 Market State

*Market state* merupakan konsep yang digunakan untuk menggambarkan
kondisi pasar pada suatu periode berdasarkan karakteristik pergerakan
harga, volatilitas, maupun dinamika transaksi. Dalam konteks pasar
keuangan, *market state* sering dikaitkan dengan kondisi seperti
*bullish*, *bearish*, *sideways*, maupun periode dengan volatilitas
tinggi, dan secara klasik dimodelkan sebagai variabel laten yang berubah
mengikuti probabilitas transisi antar-rezim (Hamilton, 1989). Berbeda
dengan pengamatan terhadap perubahan harga secara individual, konsep
*market state* berupaya menangkap pola perilaku pasar secara lebih
menyeluruh sehingga perubahan karakteristik pasar dapat diamati dalam
suatu representasi yang lebih bermakna.

Informasi mengenai *market state* memiliki peran penting dalam berbagai
aplikasi analisis keuangan. Representasi kondisi pasar dapat
dimanfaatkan sebagai dasar dalam analisis risiko, pengambilan keputusan
investasi, sistem pendukung keputusan, maupun sebagai tahap awal
berbagai metode pembelajaran mesin (Giudici & Abu Hashish, 2020). Oleh
karena itu, kualitas representasi yang digunakan untuk membangun *market
state* menjadi faktor yang sangat menentukan terhadap kualitas analisis
yang dihasilkan.

Pendekatan untuk memperoleh *market state* telah berkembang melalui
beberapa tahap yang saling membangun satu sama lain. Tahap paling awal
memodelkan pasar sebagai kumpulan kondisi statistik berbeda yang dapat
diinferensi secara probabilistik, bukan sekadar derau acak (Hamilton,
1989). Perkembangan berikutnya memformalkan kondisi pasar sebagai
*hidden state* dengan probabilitas transisi antar-rezim melalui *Hidden
Markov Model* (HMM), termasuk penerapannya secara khusus pada domain
cryptocurrency untuk mengidentifikasi rezim *bull*, *bear*, dan kondisi
volatilitas tinggi (Koki, Leonardos, & Piliouras, 2022). Sejumlah
pengembangan lanjutan memperluas kerangka HMM dengan probabilitas
transisi yang bergantung pada variabel eksogen [SUMBER:
MASTER_KOMPILASI, belum di 23-entri-final], sementara pendekatan lain
mengalihkan masalah identifikasi rezim menjadi masalah *unsupervised
clustering* --- misalnya melalui *clustering* hierarkis pada kovariansi
terealisasi (Bucci & Ciciretti, 2022). Perkembangan lebih lanjut
memperkenalkan *jump model*, yang menggabungkan *clustering* dengan
penalti perpindahan status sehingga rezim dinyatakan sebagai probabilitas
kontinu, bukan label diskrit --- pendekatan yang terbukti lebih *robust*
pada data dengan distribusi rezim yang tidak seimbang (Aydınhan, Kolm,
Mulvey, & Shu, 2024) [SUMBER: MASTER_KOMPILASI, belum diverifikasi
Scopus/DOI]. Tahap paling mutakhir beralih ke representasi laten yang
dipelajari melalui *embedding* non-linear untuk menangkap struktur
pasar berdimensi tinggi yang tidak dapat sepenuhnya ditangkap oleh
statistik klasik seperti *mean*, varians, atau matriks kovariansi
(Orton & Gebbie, 2024) [SUMBER: MASTER_KOMPILASI, belum diverifikasi
Scopus/DOI]. Perkembangan *deep learning* memungkinkan proses
pembelajaran representasi pada tahap ini dilakukan secara otomatis,
sehingga pembentukan *market state* tidak lagi sepenuhnya bergantung
pada fitur yang dirancang secara manual maupun pada asumsi struktural
yang kaku seperti pada HMM klasik.

Dalam penelitian ini, *market state* diperoleh melalui proses
*clustering* terhadap representasi laten yang dihasilkan oleh model
*Self-Supervised Learning*. Dengan demikian, penelitian tidak bertujuan
memberikan label kondisi pasar secara manual, tetapi mengevaluasi sejauh
mana representasi laten yang dipelajari model mampu menghasilkan
kelompok *market state* yang memiliki karakteristik geometrik dan
ekonomi yang berbeda.

### 2.3 Time Series Representation Learning

*Representation learning* merupakan pendekatan dalam pembelajaran mesin
yang bertujuan mempelajari representasi data secara otomatis tanpa
bergantung sepenuhnya pada proses *feature engineering*. Berbeda dengan
pendekatan konvensional yang mengharuskan perancangan fitur berdasarkan
pengetahuan domain, *representation learning* memungkinkan model
mempelajari karakteristik penting data secara langsung selama proses
pelatihan sehingga representasi yang dihasilkan lebih adaptif terhadap
struktur data yang dipelajari.

Pada data deret waktu (*time series*), pembelajaran representasi menjadi
lebih menantang karena data memiliki ketergantungan temporal, panjang
urutan yang bervariasi, serta pola yang dapat berubah pada berbagai
skala waktu. Representasi yang dihasilkan tidak hanya harus
mempertahankan informasi setiap pengamatan, tetapi juga mampu menangkap
hubungan temporal yang membentuk karakteristik keseluruhan deret waktu.
Kebutuhan inilah yang mendasari mengapa *representation learning*
diperlukan pada domain ini: alih-alih merancang fitur secara manual
untuk setiap skala waktu dan setiap jenis pola pasar, model dilatih
untuk menemukan sendiri struktur laten yang relevan langsung dari data
historis.

Secara umum, *representation learning* bertujuan mempelajari suatu
fungsi pemetaan yang mengubah data masukan ke dalam ruang representasi
laten berdimensi lebih rendah tanpa menghilangkan informasi penting yang
terkandung di dalamnya. Proses tersebut dapat dinyatakan sebagai
berikut.

**Persamaan (2.1) --- Representation Learning**

$$\mathbf{z} = f_{\theta}\left( \mathbf{x} \right)$$

Keterangan:

-   $\mathbf{x} \in \mathbb{R}^{T \times F}$ : data masukan (*input time
    series*) dengan $T$ langkah waktu dan $F$ fitur
-   $f_{\theta}( \cdot )$ : fungsi pemetaan (*representation function*)
    dengan parameter $\theta$
-   $\mathbf{z} \in \mathbb{R}^{K}$ : representasi laten (*latent
    representation*) berdimensi $K$

Persamaan (2.1) menunjukkan bahwa model bertugas memetakan data deret
waktu ke dalam ruang representasi baru yang diharapkan mampu
mempertahankan karakteristik penting dari data asli. Representasi laten
tersebut selanjutnya dapat dimanfaatkan oleh berbagai metode analisis
tanpa harus membangun kembali proses ekstraksi fitur secara manual.

Dalam penelitian ini, fungsi pemetaan $f_{\theta}( \cdot )$
direpresentasikan oleh model TS2Vec, sedangkan representasi laten
$\mathbf{z}$ digunakan sebagai masukan pada proses *market state
discovery*. Penelitian tidak mengevaluasi kemampuan model dalam
melakukan prediksi harga, melainkan menginvestigasi bagaimana variasi
jumlah resolusi temporal memengaruhi kualitas representasi laten yang
dipelajari.

### 2.4 Self-Supervised Learning dan Munculnya Contrastive Learning

*Self-Supervised Learning* (SSL) merupakan paradigma pembelajaran yang
memanfaatkan struktur alami data sebagai sumber supervisi sehingga tidak
memerlukan proses pelabelan secara manual. Berbeda dengan *supervised
learning* yang bergantung pada ketersediaan label, SSL membangun sinyal
pembelajaran (*pretext task*) secara otomatis dari data itu sendiri.
Pendekatan ini menjadi semakin penting pada domain data deret waktu
karena sebagian besar dataset tersedia dalam jumlah besar, tetapi
memiliki keterbatasan label yang dapat digunakan untuk proses pelatihan
--- termasuk pada data pasar keuangan, di mana label kondisi pasar yang
objektif dan disepakati secara universal pada dasarnya tidak tersedia.

Tujuan utama *Self-Supervised Learning* adalah mempelajari representasi
laten yang mampu mempertahankan karakteristik penting data sehingga
dapat dimanfaatkan pada berbagai *downstream task*. Secara umum, proses
pembelajaran pada SSL dilakukan dengan mencari parameter model yang
meminimalkan fungsi objektif yang dibangun dari hubungan antar
representasi data, sebagaimana dinyatakan berikut.

**Persamaan (2.2) --- Proses Optimasi Self-Supervised Learning**

$$\theta^{*} = \arg\min_{\theta}\mathcal{L}(\theta)$$

Keterangan:

-   $\theta$ : parameter model yang dipelajari
-   $\mathcal{L}(\theta)$ : fungsi objektif (*objective function*) yang
    dibangun dari hubungan antar representasi tanpa menggunakan label
    kelas
-   $\theta^{*}$ : parameter optimal yang meminimalkan fungsi objektif

Persamaan (2.2) menunjukkan bahwa proses pembelajaran bertujuan
memperoleh parameter model yang menghasilkan representasi terbaik
berdasarkan fungsi objektif yang dirancang tanpa menggunakan label
kelas. Berbeda dengan *supervised learning*, sinyal pembelajaran pada
SSL diperoleh dari hubungan antar sampel atau struktur temporal yang
terdapat pada data itu sendiri.

Salah satu strategi SSL yang paling berpengaruh dalam mewujudkan
Persamaan (2.2) adalah *contrastive learning*. Pendekatan ini muncul
untuk menjawab tantangan mendasar dalam SSL, yaitu bagaimana merancang
fungsi objektif $\mathcal{L}(\theta)$ yang bermakna tanpa label.
*Contrastive learning* menjawabnya dengan mendefinisikan pasangan
positif (representasi yang seharusnya mirip, misalnya dua potongan
sinyal yang berdekatan atau berasal dari konteks yang sama) dan pasangan
negatif (representasi yang seharusnya berbeda), lalu melatih model untuk
mendekatkan pasangan positif dan menjauhkan pasangan negatif dalam ruang
laten. Peletak dasar pendekatan ini pada deret waktu secara umum adalah
*Contrastive Predictive Coding* (CPC), yang memperkenalkan fungsi
*InfoNCE loss* untuk mempelajari representasi dengan memprediksi
observasi masa depan dalam ruang laten, alih-alih memprediksi nilai
observasi itu sendiri secara langsung (Van den Oord, Li, & Vinyals,
2018). Gagasan CPC --- memaksimalkan *mutual information* antara
representasi konteks dan representasi masa depan melalui perbandingan
kontrastif --- menjadi fondasi konseptual bagi hampir seluruh metode
*contrastive representation learning* untuk deret waktu yang berkembang
setelahnya, termasuk *Temporal Neighborhood Coding* (TNC), TS-TCC, CoST,
dan TS2Vec yang dibahas lebih lanjut pada Subbab 2.5, 2.8, dan 2.9.

*Contrastive learning* menjadi sangat relevan untuk deret waktu
finansial karena dua alasan utama. Pertama, sinyal harga tidak memiliki
label kondisi pasar yang objektif, sehingga pendekatan berbasis prediksi
label (*supervised*) sulit diterapkan tanpa anotasi manual yang
subjektif. Kedua, struktur temporal deret waktu finansial --- kedekatan
waktu cenderung berkorelasi dengan kemiripan kondisi pasar --- secara
alami menyediakan sinyal pasangan positif/negatif yang dibutuhkan
*contrastive learning*, tanpa memerlukan augmentasi data yang rumit
seperti pada domain citra.

Berdasarkan pertimbangan tersebut, penelitian ini menggunakan
*Self-Supervised Learning* berbasis *contrastive learning* sebagai
paradigma utama untuk mempelajari representasi pasar cryptocurrency
tanpa memerlukan label kondisi pasar. Metode spesifik yang digunakan
adalah TS2Vec, yang dipilih dan dijustifikasi lebih lanjut pada Subbab
2.5.

### 2.5 TS2Vec sebagai Baseline

TS2Vec (*Time Series to Vector*) merupakan metode *Self-Supervised
Learning* yang dirancang untuk menghasilkan representasi umum
(*universal representation*) pada data deret waktu (Yue et al., 2022).
Berbeda dengan metode *contrastive* generasi sebelumnya yang hanya
menghasilkan representasi pada satu tingkat granularitas --- CPC pada
tingkat prediksi ke depan (Van den Oord, Li, & Vinyals, 2018), TNC pada
tingkat *neighborhood* temporal (Tonekaboni, Eytan, & Goldenberg, 2021),
TS-TCC pada tingkat augmentasi lemah/kuat (Eldele et al., 2021), dan
CoST pada tingkat dekomposisi musiman-tren dalam domain waktu dan
frekuensi (Woo et al., 2022) --- TS2Vec mampu menghasilkan representasi
pada tingkat *timestamp* maupun tingkat *instance* (sub-*sequence*)
secara simultan melalui mekanisme *hierarchical contextual contrastive
learning*.

Penelitian ini memilih TS2Vec sebagai *baseline* encoder atas dasar
beberapa pertimbangan berikut, bukan semata-mata karena popularitasnya:

1.  **Fleksibilitas panjang urutan dan level representasi.** TS2Vec
    dirancang secara eksplisit untuk menghasilkan representasi pada
    level yang arbitrer (*timestamp*, sub-*series*, maupun *instance*
    penuh) melalui agregasi sederhana atas representasi *timestamp*-nya.
    Karakteristik ini cocok dengan kebutuhan penelitian, di mana
    representasi setiap *window* (bukan setiap *timestamp* individual)
    yang menjadi unit analisis untuk *market state discovery*.
2.  **Hierarchical contrasting mengurangi bias induktif yang kaku.**
    Metode *contrastive* sebelumnya seperti TNC dan TS-TCC memberlakukan
    asumsi struktural yang cukup kuat (misalnya invarian transformasi
    tertentu) untuk menentukan pasangan positif. TS2Vec, melalui kontras
    hierarkis pada berbagai skala, mengurangi ketergantungan pada asumsi
    tersebut sehingga lebih adaptif terhadap karakteristik data yang
    belum diketahui strukturnya secara pasti --- relevan untuk deret
    waktu finansial yang bersifat non-stasioner.
3.  **Validasi empiris yang luas.** TS2Vec telah diuji pada 125 dataset
    UCR dan 29 dataset UEA serta menunjukkan perbaikan signifikan
    dibandingkan metode *state-of-the-art* sebelumnya untuk
    *representation learning* deret waktu tanpa label, dengan kinerja
    yang juga unggul pada tugas *forecasting* dan deteksi anomali (Yue
    et al., 2022).
4.  **Ketersediaan implementasi resmi dan reproduktifitas.** TS2Vec
    memiliki kode sumber resmi yang terbuka, memungkinkan penggunaan
    sebagai instrumen tanpa modifikasi terhadap arsitektur inti ---
    sejalan dengan prinsip *controlled empirical study* pada penelitian
    ini (lihat ADR-001 pada dokumentasi desain arsitektur).

Pada penelitian ini, TS2Vec diposisikan sebagai instrumen pembelajaran
representasi, bukan sebagai metode yang dikembangkan ataupun
dimodifikasi. Implementasi resmi TS2Vec digunakan tanpa modifikasi pada
*source code*, sedangkan seluruh logika eksperimen (pelatihan
per-cabang, konfigurasi, *fusion*, dan evaluasi) diimplementasikan
secara terpisah sebagai modul pembungkus (*wrapper*).

Proses pembelajaran TS2Vec dilakukan menggunakan mekanisme *hierarchical
contextual contrastive learning*, yaitu dengan memaksimalkan kemiripan
representasi yang berasal dari konteks temporal yang sama dan
meminimalkan kemiripan terhadap konteks yang berbeda --- sebuah
perluasan hierarkis dari prinsip kontrastif yang pertama kali
diperkenalkan CPC (Van den Oord, Li, & Vinyals, 2018). Secara umum,
fungsi objektif TS2Vec dinyatakan sebagai berikut.

**Persamaan (2.3) --- Fungsi Objektif TS2Vec** *(representasi yang
disederhanakan)*

$$\mathcal{L}_{\text{TS2Vec}} = \mathcal{L}_{\text{temporal}} + \mathcal{L}_{\text{inst}}$$

Keterangan:

-   $\mathcal{L}_{\text{TS2Vec}}$ : fungsi objektif total TS2Vec,
    dihitung secara hierarkis pada beberapa skala granularitas
-   $\mathcal{L}_{\text{temporal}}$ : *temporal contrastive loss*,
    mengoptimalkan konsistensi representasi pada dimensi waktu
    antar-augmentasi
-   $\mathcal{L}_{\text{inst}}$ : *instance-wise contrastive loss*,
    mengoptimalkan pemisahan representasi antar-instans berbeda pada
    *timestamp* yang sama

**Catatan**: Persamaan (2.3) merupakan representasi yang disederhanakan
untuk keperluan paparan konseptual. Pada implementasi aslinya, kedua
*loss* dihitung secara rekursif pada setiap tingkat hierarki melalui
proses *max-pooling* bertahap sebagaimana dijelaskan pada Yue et
al. (2022). Penelitian ini menggunakan implementasi resmi TS2Vec,
dipinned pada satu *commit* tertentu untuk menjamin reproduktifitas,
tanpa modifikasi terhadap *source code* aslinya.

Setelah proses pelatihan selesai, TS2Vec menghasilkan representasi
temporal untuk setiap langkah waktu. Agar setiap *window*
direpresentasikan oleh satu vektor tetap, dilakukan proses *temporal
max-pooling* yang dinyatakan sebagai berikut.

**Persamaan (2.4) --- Temporal Max-Pooling**

$$\mathbf{z} = MaxPool\left( \mathbf{H} \right) = \underset{t = 1}{\max^{T}}\mathbf{h}_{t}$$

Keterangan:

-   ::: Definition-Term
    $\mathbf{H} = \left\lbrack \mathbf{h}_{1},\mathbf{h}_{2},\ldots,\mathbf{h}_{T} \right\rbrack \in \mathbb{R}^{T \times K}$
    :::

```{=html}
<!-- -->
```
-   representasi temporal yang dihasilkan TS2Vec, dengan $K = 64$ pada
    penelitian ini

```{=html}
<!-- -->
```
-   $MaxPool( \cdot )$ : operasi *temporal max-pooling* secara
    *element-wise* sepanjang dimensi waktu
-   $\mathbf{z} \in \mathbb{R}^{K}$ : representasi laten akhir dari satu
    *window*

Representasi laten hasil *max-pooling* inilah yang digunakan pada
penelitian sebagai masukan untuk proses *late fusion* dan *market state
discovery*.

### 2.6 Multi-Resolution Time Series Representation

Data deret waktu (*time series*) dapat diamati pada berbagai resolusi
temporal (*temporal resolution*) atau *timeframe*, misalnya 15 menit, 1
jam, 4 jam, maupun 1 hari. Literatur *multi-scale time-series modeling*
secara konsisten menunjukkan bahwa representasi satu resolusi
(*single-resolution tokenization*) seringkali tidak memadai karena pola
temporal yang berbeda hidup pada frekuensi atau granularitas yang
berbeda pula; ukuran jendela pengamatan yang lebih pendek cenderung
menangkap pola frekuensi-tinggi yang bersifat lokal, sedangkan jendela
yang lebih panjang lebih sesuai untuk menangkap musiman dan tren jangka
panjang (Zhang et al., 2024; Chen et al., 2024). Setiap resolusi
temporal karenanya menyajikan karakteristik informasi yang berbeda:
*timeframe* yang lebih pendek cenderung menangkap perubahan pasar dalam
jangka pendek (informasi mikro), sedangkan *timeframe* yang lebih
panjang memberikan gambaran tren pasar yang lebih stabil (informasi
struktural/regime) (Wang et al., 2024). Kedua jenis informasi ini
bersifat saling melengkapi, karena skala halus (*fine scale*)
mengkodekan fluktuasi lokal dan dependensi jangka-pendek, sementara
skala kasar (*coarse scale*) mengkodekan tren dan musiman jangka panjang
(Wang et al., 2024; Zhang et al., 2024). Oleh karena itu, penggunaan
beberapa resolusi temporal berpotensi memberikan informasi yang lebih
lengkap dibandingkan hanya menggunakan satu *timeframe*.

Perlu dicatat bahwa keunggulan pendekatan multi-resolusi bukan merupakan
konsensus mutlak dalam literatur: terdapat pula bukti bahwa representasi
satu-resolusi dengan bobot yang dibagikan antar-*channel*
(*channel-independent shared weights*) dapat memberikan performa
kompetitif pada sejumlah tugas *forecasting* (Nie et al., 2023). Temuan
ini menunjukkan bahwa manfaat integrasi multi-resolusi bergantung pada
karakteristik tugas dan data, sehingga relevan untuk dievaluasi secara
empiris pada domain *market state discovery* --- sebagaimana menjadi
fokus penelitian ini --- alih-alih diasumsikan berlaku secara umum.
Kajian literatur yang lebih luas mengonfirmasi bahwa mekanisme baku
untuk menggabungkan *embedding* dari beberapa resolusi temporal secara
optimal memang belum tersedia, sehingga ruang penelitian ini --- baik
dari sisi *apakah* multi-resolusi memberi kontribusi maupun *bagaimana*
representasi tersebut sebaiknya digabungkan --- tetap terbuka [SUMBER:
MASTER_KOMPILASI, belum di 23-entri-final]. Perlu ditegaskan pula bahwa
bukti *information-theoretic* formal mengenai resolusi mana yang secara
matematis perlu dan cukup untuk merepresentasikan dinamika pasar masih
sangat terbatas dibandingkan bukti pada sisi arsitektur model [SUMBER:
MASTER_KOMPILASI, belum di 23-entri-final] --- sehingga penelitian ini
diposisikan secara eksplisit sebagai validasi empiris terkontrol
terhadap kontribusi resolusi temporal, bukan sebagai klaim teori formal
baru mengenai skala optimal pasar.

Penelitian ini memandang resolusi temporal sebagai variabel independen
yang diinvestigasi melalui protokol eksperimen terkontrol. Seluruh
konfigurasi model, proses pelatihan, dimensi representasi, metode
*clustering*, serta prosedur evaluasi dipertahankan tetap pada setiap
kondisi eksperimen sehingga perubahan hasil dapat diinterpretasikan
sebagai pengaruh jumlah resolusi temporal.

Setiap resolusi temporal diproses oleh encoder TS2Vec yang **dilatih
secara independen** --- tidak ada gradien yang dibagikan antar-cabang
pada tahap manapun. Pilihan desain ini konsisten dengan salah satu dari
dua aliran pendekatan yang masih aktif diperdebatkan dalam literatur
multi-resolusi: *shared encoder* dengan bobot yang sama pada seluruh
skala versus *independent/specialized encoder* yang memberi setiap skala
kapasitas pembelajaran tersendiri (Zhang et al., 2024; Wang et al.,
2024). Penelitian ini memilih pendekatan *independent encoder* karena
tujuannya adalah mengisolasi kontribusi representasi setiap resolusi
secara terpisah sebelum digabungkan, bukan mempelajari representasi
gabungan secara *end-to-end* --- pilihan yang selaras dengan prinsip
*late fusion* yang dibahas lebih lanjut pada bagian berikut. Bobot
setiap *branch* dilatih satu kali dan digunakan kembali pada seluruh
kondisi eksperimen yang melibatkan *timeframe* tersebut (misalnya,
*branch* 1h yang sama digunakan pada kondisi 1TF, 2TF, 3TF, dan 4TF).
Secara konseptual: *setiap branch mempelajari representasi
single-resolution secara independen; kondisi multi-resolusi mengevaluasi
efek penggabungan (fusion) representasi yang telah dipelajari secara
independen tersebut --- bukan efek dari pembelajaran multi-resolusi
secara bersama (joint learning)*.

Setelah setiap resolusi temporal menghasilkan representasi laten secara
independen, representasi laten dari masing-masing cabang digabungkan
menggunakan pendekatan *late concatenation fusion*. Proses tersebut
dinyatakan sebagai berikut.

Literatur multi-resolusi mencatat bahwa mekanisme baku untuk
menggabungkan representasi dari beberapa resolusi temporal secara
optimal belum tersedia, dan bahwa perdebatan antara *early fusion*
(penggabungan sejak tahap awal *encoding*) versus *late fusion* (setiap
resolusi dipelajari secara terpisah lalu digabungkan pada tahap akhir)
masih terbuka tanpa konsensus bahwa salah satu pendekatan selalu lebih
unggul (Zhang et al., 2023; Liu & Chen, 2024). Penelitian ini secara
eksplisit memilih *late fusion* melalui konkatenasi deterministik
agar kontribusi setiap resolusi tetap dapat diisolasi dan dievaluasi
secara terpisah sebelum digabungkan, dinyatakan sebagai berikut.

**Persamaan (2.5) --- Late Concatenation Fusion**

$$\mathbf{z}_{\text{fused}} = \left\lbrack \mathbf{z}^{(1)} \parallel \mathbf{z}^{(2)} \parallel \cdots \parallel \mathbf{z}^{(n)} \right\rbrack$$

Keterangan:

-   $\mathbf{z}^{(i)} \in \mathbb{R}^{64}$ : representasi laten dari
    resolusi temporal ke-$i$ yang dihasilkan oleh encoder TS2Vec cabang
    ke-$i$
-   $\parallel$ : operasi konkatenasi (*concatenation*) pada dimensi
    fitur
-   $\mathbf{z}_{\text{fused}} \in \mathbb{R}^{64n}$ : representasi
    hasil penggabungan, dengan $n \in \{ 1,2,3,4\}$ sesuai kondisi
    eksperimen
-   Urutan konkatenasi: 15m → 1h → 4h → 1d, ditetapkan tetap pada
    seluruh eksperimen dan ditegakkan secara terprogram menggunakan
    struktur berurutan (*ordered list*), bukan struktur tanpa urutan
    (*set* atau *dict*)

Karena jumlah resolusi temporal berbeda pada setiap kondisi eksperimen,
dimensi hasil konkatenasi juga berbeda (64, 128, 192, atau 256). Untuk
memastikan seluruh kondisi menghasilkan dimensi representasi yang
identik sehingga perbandingan antar kondisi dapat dilakukan secara adil,
dilakukan proses *fixed random projection* sebagai berikut.

**Persamaan (2.6) --- Fixed Random Projection**
*(Johnson-Lindenstrauss)*

$$\mathbf{z}_{\text{final}} = \mathbf{R}\,\mathbf{z}_{\text{fused}}$$

Keterangan:

-   $\mathbf{R} \in \mathbb{R}^{256 \times 64n}$ : matriks *fixed random
    projection* yang diinisialisasi dengan *random seed* tetap
    ($\text{seed} = 42$) dan **tidak memiliki parameter yang
    dipelajari** selama pelatihan (*zero learnable parameters*); setiap
    baris matriks dinormalisasi ke norma-L2 satuan
-   $\mathbf{z}_{\text{fused}} \in \mathbb{R}^{64n}$ : representasi
    hasil konkatenasi dari tahap sebelumnya
-   $\mathbf{z}_{\text{final}} \in \mathbb{R}^{256}$ : representasi
    akhir berdimensi tetap yang digunakan pada seluruh kondisi
    eksperimen

Penggunaan *fixed random projection* (alih-alih *trainable linear
layer*) bertujuan memastikan bahwa tidak ada parameter tambahan yang
dipelajari selama proses proyeksi. Dengan demikian, perbandingan antar
kondisi eksperimen tidak terkontaminasi oleh kapasitas pembelajaran yang
berbeda --- proyeksi bersifat murni sebagai operasi standardisasi
dimensi, bukan sebagai komponen yang dioptimasi. Matriks $\mathbf{R}$
dihasilkan menggunakan distribusi Gaussian, dengan setiap baris
dinormalisasi ke norma satuan --- konstruksi ini terinspirasi oleh
prinsip Johnson-Lindenstrauss bahwa proyeksi acak linear secara
aproksimatif mempertahankan jarak berpasangan antar titik data (Johnson
& Lindenstrauss, 1984). Perlu ditegaskan bahwa jaminan teoritis formal
Johnson-Lindenstrauss dirumuskan untuk arah reduksi dimensi ($n \gg k$),
sedangkan pada penelitian ini proyeksi diterapkan pada arah yang
beragam bergantung kondisi eksperimen --- termasuk arah peningkatan
dimensi nominal (*dimensional upscaling*) pada kondisi dengan jumlah
*branch* aktif yang lebih sedikit (mis. $64 \rightarrow 256$ pada
kondisi 1TF). Oleh karena itu, penggunaan proyeksi acak pada penelitian
ini **tidak diklaim** sebagai penerapan jaminan matematis
Johnson-Lindenstrauss secara ketat, melainkan sebagai strategi
standardisasi dimensi yang bersifat *data-oblivious* (tidak
beradaptasi secara berbeda antar kondisi) dan tidak menambahkan
parameter yang dipelajari --- dua sifat yang secara metodologis lebih
relevan untuk menjaga validitas kontrol eksperimen dibandingkan jaminan
preservasi jarak itu sendiri. Justifikasi utama pemilihan pendekatan
ini adalah netralitas terhadap desain eksperimen: proyeksi acak tetap
(*fixed*) dipilih karena tidak memperkenalkan variabel bebas tambahan
yang dapat "bercampur" (*confound*) dengan variabel independen utama
penelitian, yaitu jumlah resolusi temporal --- bukan karena diklaim
sebagai metode proyeksi yang optimal secara performa.

Perlu dicatat bahwa matriks proyeksi $\mathbf{R}$ dibangun secara
terpisah untuk setiap kondisi, karena dimensi masukan $64n$ berbeda per
kondisi. Akibatnya, meskipun seluruh kondisi menghasilkan keluaran
berdimensi nominal 256, **rank efektif** dari representasi akhir tetap
dibatasi oleh dimensi masukannya (64, 128, 192, atau 256, bergantung
jumlah *branch* aktif) --- proyeksi acak tidak dapat menambah informasi
yang tidak ada di $\mathbf{z}_{\text{fused}}$. Karakteristik ini
bersifat konservatif terhadap hipotesis penelitian: kondisi dengan
resolusi tunggal memiliki ruang representasi efektif yang secara
matematis lebih sempit, sehingga keunggulan kondisi multi-resolusi (jika
ditemukan) tidak dapat dijelaskan oleh keunggulan dimensi proyeksi
semata. Keterbatasan ini didiskusikan lebih lanjut pada Subbab 1.7
(Batasan Penelitian) dan dilaporkan pada bagian Keterbatasan Penelitian.

### 2.7 Market State Discovery

*Market state discovery* merupakan proses mengidentifikasi kondisi pasar
berdasarkan karakteristik data historis tanpa memerlukan label yang
telah ditentukan sebelumnya. Berbeda dengan pendekatan klasifikasi yang
bergantung pada anotasi manual, *market state discovery* bertujuan
menemukan struktur alami (*natural structure*) yang terdapat pada data
sehingga kondisi pasar diperoleh dari kemiripan karakteristik yang
dipelajari model. Pendekatan ini bertumpu pada beberapa premis yang
secara konsisten didukung dalam literatur *representation learning* dan
*clustering*: kualitas hasil *clustering* ditentukan terutama oleh
kualitas representasi laten yang digunakan sebagai masukan, bukan oleh
algoritma *clustering* semata; kombinasi *deep representation learning*
dengan *clustering* merupakan paradigma yang lebih dominan dibandingkan
penerapan *clustering* langsung pada fitur mentah; dan representasi
yang dipelajari melalui *contrastive learning* cenderung menghasilkan
ruang laten yang lebih *separable*, sehingga lebih mudah membentuk
kelompok yang bermakna secara substantif. Ketiga premis tersebut
merupakan konsensus yang konsisten disimpulkan lintas berbagai studi
representasi dan *clustering* yang ditelaah pada tinjauan pustaka ini
[Additional Reference Needed --- premis ini disintesis dari penelusuran
literatur yang lebih luas dan belum dapat diatribusikan ke satu studi
primer tunggal secara langsung]. Ketiga premis ini menjadi dasar
konseptual mengapa penelitian ini menempatkan kualitas representasi
laten --- bukan pemilihan algoritma *clustering* --- sebagai variabel
utama yang dievaluasi.

Di antara berbagai algoritma *clustering*, HDBSCAN (*Hierarchical
Density-Based Spatial Clustering of Applications with Noise*) menjadi
salah satu metode yang banyak digunakan karena mampu membentuk kelompok
data dengan kepadatan yang berbeda tanpa menentukan jumlah *cluster*
sejak awal (Campello, Moulavi, & Sander, 2013; McInnes, Healy, & Astels,
2017). Selain itu, HDBSCAN juga mampu mengidentifikasi observasi yang
tidak termasuk ke dalam kelompok tertentu (*noise*), sehingga lebih
fleksibel dalam menganalisis data dengan distribusi yang kompleks.
Perbandingan empiris pada data dengan struktur *cluster* tidak beraturan
menunjukkan bahwa HDBSCAN secara konsisten mengungguli K-Means pada
sebagian besar metrik validasi internal (Valles-Coral et al., 2022)
[SUMBER: MASTER_KOMPILASI, belum diverifikasi Scopus/DOI], sementara
K-Means tetap menjadi *baseline* yang kompetitif ketika jumlah *cluster*
dapat ditentukan secara tepat.
Karena struktur *market state* pada data finansial tidak diasumsikan
memiliki bentuk geometris tertentu maupun jumlah kelompok yang diketahui
sejak awal, karakteristik HDBSCAN yang tidak memerlukan asumsi tersebut
menjadikannya pilihan yang lebih sesuai dengan sifat data yang dianalisis
dibandingkan metode *clustering* berbasis partisi seperti K-Means.

Pada penelitian ini, HDBSCAN tidak digunakan sebagai objek yang
dibandingkan dengan algoritma *clustering* lainnya, melainkan sebagai
instrumen untuk mengevaluasi kualitas representasi laten yang dihasilkan
oleh TS2Vec. Seluruh kondisi eksperimen menggunakan algoritma dan
konfigurasi HDBSCAN yang sama --- parameter dikunci dari hasil *grid
search* pada kondisi 1TF (lihat Subbab 3.5) --- sehingga perubahan
kualitas *market state* dapat dikaitkan dengan perubahan kualitas
representasi laten, bukan akibat perbedaan metode *clustering* maupun
perbedaan hiperparameter antar kondisi.

### 2.8 Penelitian Terdahulu

Subbab ini menelusuri perkembangan literatur yang relevan mengikuti lima
pertanyaan yang membentuk argumen penelitian: mengapa *representation
learning* diperlukan, mengapa *contrastive learning* muncul sebagai
pendekatan dominan, mengapa TS2Vec dipilih sebagai *baseline*, mengapa
kesenjangan pada *market state representation* masih terbuka, dan
mengapa penelitian ini diperlukan untuk mengisi kesenjangan tersebut.

**Mengapa *Representation Learning* Diperlukan.** Pendekatan
konvensional dalam analisis deret waktu finansial umumnya bergantung
pada indikator teknikal yang dirancang secara manual (RSI, MACD, *moving
average*, dan sejenisnya), yang masing-masing membawa parameter
*lookback* dan asumsi *smoothing* tertentu. Pendekatan ini rentan
terhadap *bias* rekayasa fitur: pemilihan indikator dan parameternya
cenderung didasarkan pada intuisi atau kebiasaan praktik, bukan pada
bukti bahwa indikator tersebut menangkap struktur pasar yang sebenarnya.
*Representation learning* menawarkan alternatif dengan mempelajari
struktur laten langsung dari data historis (Persamaan 2.1), sehingga
representasi yang dihasilkan lebih adaptif terhadap karakteristik data
yang sedang dianalisis. Pada domain deret waktu secara umum, kebutuhan
ini mendorong berkembangnya berbagai metode *representation learning*
berbasis *deep learning*.

**Mengapa *Contrastive Learning* Muncul.** Ketika *representation
learning* diterapkan tanpa label (*unsupervised*/*self-supervised*),
tantangan utamanya adalah merancang fungsi objektif yang bermakna tanpa
pengawasan eksternal (Persamaan 2.2). *Contrastive learning* menjawab
tantangan ini dengan memanfaatkan struktur intrinsik data itu sendiri
sebagai sumber supervisi. Van den Oord, Li, dan Vinyals (2018)
memperkenalkan *Contrastive Predictive Coding* (CPC), yang mempelajari
representasi dengan memaksimalkan *mutual information* antara
representasi konteks saat ini dan observasi masa depan melalui fungsi
*InfoNCE loss* --- sebuah pendekatan yang terbukti bekerja secara
universal pada domain suara, citra, teks, dan *reinforcement learning*.
Prinsip inti CPC ini --- belajar dengan membedakan pasangan yang
berkorelasi secara alami dari pasangan yang tidak berkorelasi, tanpa
memerlukan label --- menjadi fondasi bagi hampir seluruh metode
*representation learning* deret waktu berikutnya. Tonekaboni, Eytan, dan
Goldenberg (2021) memperluas gagasan ini dengan *Temporal Neighborhood
Coding* (TNC), yang mendefinisikan *neighborhood* temporal berbasis
kemulusan lokal (*local smoothness*) proses generatif sinyal, dengan
fungsi objektif kontrastif yang *debiased* untuk menangani deret waktu
non-stasioner dan multivariat --- karakteristik yang relevan bagi data
deret waktu medis maupun finansial. Woo et al. (2022) selanjutnya
mengembangkan CoST, yang menerapkan *contrastive learning* secara
terpisah pada domain waktu dan domain frekuensi untuk mempelajari
representasi musiman (*seasonal*) dan tren (*trend*) yang terpisah
(*disentangled*), dengan tujuan akhir meningkatkan performa
*forecasting* deret waktu jangka panjang.

**Mengapa TS2Vec Menjadi *Baseline*.** Di antara metode *contrastive*
yang berkembang, TS2Vec (Yue et al., 2022) diposisikan sebagai kerangka
yang lebih umum karena menerapkan *kontras hierarkis* pada berbagai
tingkat granularitas secara simultan, alih-alih hanya pada satu level
representasi seperti pendekatan sebelumnya. Yue et al. (2022) secara
eksplisit mencatat bahwa metode-metode seperti TNC dan TS-TCC (Eldele et
al., 2021) melakukan *contrastive learning* hanya pada level tertentu
dan memberlakukan *bias* induktif yang cukup kuat (misalnya invarian
transformasi tertentu) untuk menentukan pasangan positif, sementara
TS2Vec dirancang untuk menghasilkan representasi yang valid pada level
*timestamp*, sub-*series*, maupun *instance* penuh sekaligus. Validasi
empiris pada 125 dataset UCR dan 29 dataset UEA menunjukkan bahwa TS2Vec
mencapai perbaikan signifikan dibandingkan metode-metode SSL sebelumnya,
termasuk pada tugas hilir seperti klasifikasi, *forecasting*, dan
deteksi anomali. Kombinasi antara fleksibilitas level representasi,
validasi empiris yang luas, dan ketersediaan implementasi resmi
menjadikan TS2Vec pilihan yang tepat sebagai instrumen pembelajaran
representasi (*baseline*) bagi penelitian ini, sebagaimana dijustifikasi
lebih rinci pada Subbab 2.5.

**Mengapa Masih Ada Gap pada *Market State Representation*.** Meninjau
keempat metode *contrastive* di atas (CPC, TNC, CoST, TS2Vec) beserta
metode identifikasi kondisi pasar yang telah mapan (HMM), tampak dua
arah penelitian yang belum bertemu secara langsung:

Pertama, seluruh metode *representation learning* di atas dikembangkan
dan divalidasi pada *benchmark* klasifikasi dan *forecasting* deret
waktu umum (UCR, UEA, ETT, dan sejenisnya), bukan pada tugas *market
state discovery* di domain finansial. Variasi resolusi temporal
(*timeframe*) tidak pernah diperlakukan sebagai variabel eksperimen yang
dikontrol secara sistematis pada metode-metode ini; input yang digunakan
pada eksperimen aslinya bersifat *single-resolution* sesuai dengan
format dataset yang tersedia.

Kedua, pada domain keuangan, penelitian mengenai identifikasi kondisi
pasar melalui HMM tetap aktif dilakukan hingga saat ini. Koki,
Leonardos, dan Piliouras (2022) menerapkan HMM berbasis Bayesian untuk
memodelkan pergantian rezim pasar (*bear*, *stable*, *bull*) pada
Bitcoin, Ether, dan Ripple, menunjukkan bahwa kerangka HMM secara
konsisten mampu memisahkan periode dengan karakteristik risiko dan
profitabilitas yang berbeda. Namun, pendekatan HMM secara struktural
mengasumsikan sejumlah kecil rezim laten dengan dinamika transisi
Markovian, dan tidak dirancang untuk mengintegrasikan informasi dari
berbagai resolusi temporal sekaligus. Di sisi lain, penelitian
*multi-timeframe* pada domain finansial --- misalnya rekayasa fitur
multi-resolusi untuk prediksi harga Bitcoin yang menggabungkan empat
resolusi temporal (Sobreiro et al., 2026) --- secara konsisten diarahkan
untuk meningkatkan akurasi *forecasting* atau klasifikasi sinyal
*entry*, bukan untuk mengevaluasi kualitas representasi laten yang
dipelajari secara *self-supervised*.

Dengan demikian, terdapat kesenjangan yang jelas: metode *representation
learning* kontrastif (CPC, TNC, CoST, TS2Vec) belum diuji secara
terkontrol terhadap variasi resolusi temporal pada domain *market state
discovery*, sementara penelitian *multi-timeframe* finansial yang sudah
mempertimbangkan berbagai resolusi belum menyentuh ranah pembelajaran
representasi tanpa label untuk tujuan *clustering* kondisi pasar.

**Mengapa Penelitian Ini Diperlukan.** Berdasarkan kesenjangan tersebut,
diperlukan sebuah studi yang secara eksplisit memperlakukan jumlah
resolusi temporal sebagai variabel independen yang diuji secara
terkontrol, menggunakan encoder *representation learning* yang telah
tervalidasi secara luas (TS2Vec), pada tugas yang belum banyak
dieksplorasi dalam konteks ini (*market state discovery* berbasis
*clustering*), dengan protokol statistik yang *pre-registered* untuk
menjamin validitas kesimpulan. Penelitian ini memposisikan diri secara
langsung pada irisan ketiga bidang tersebut --- *contrastive
representation learning*, evaluasi *multi-resolution input*, dan *market
state discovery* --- sebagaimana dirangkum pada Tabel 2.1 dan Tabel 2.2.

**Tabel 2.1 Ringkasan Penelitian Terdahulu**

  ---------------------------------------------------------------------------------------------------------------------------------------------
  Penulis &     Metode/Arsitektur   Domain/Dataset     Tugas Utama          Metrik Evaluasi          Keterbatasan Terkait   Relevansi terhadap
  Tahun                                                                                              Penelitian Ini         Penelitian Ini
  ------------- ------------------- ------------------ -------------------- ------------------------ ---------------------- -------------------
  Van den Oord, Contrastive         Suara, citra,      Representation       Akurasi klasifikasi      Tidak spesifik deret   Fondasi konseptual
  Li, & Vinyals Predictive Coding   teks, RL           learning tujuan umum linear pada representasi waktu finansial; tidak contrastive
  (2018)        (CPC), InfoNCE loss (multi-domain)                          yang dipelajari          mengevaluasi           learning yang
                                                                                                     multi-resolusi         mendasari seluruh
                                                                                                                            metode berikutnya
                                                                                                                            (Subbab 2.4)

  Tonekaboni,   Temporal            Deret waktu medis  Representation       Akurasi                  Dirancang untuk data   Menunjukkan
  Eytan, &      Neighborhood Coding & simulasi         learning untuk deret klasifikasi/clustering   medis;                 pentingnya
  Goldenberg    (TNC)               non-stasioner      waktu non-stasioner  hilir                    single-resolution;     menangani
  (2021)                                                                                             tidak diuji pada data  non-stasioneritas
                                                                                                     finansial              --- relevan untuk
                                                                                                                            data crypto

  Eldele et     TS-TCC (Temporal &  Deret waktu        Representation       Akurasi klasifikasi      Bias induktif          Pembanding metode
  al. (2021)    Contextual          sensor/aktivitas   learning,                                     augmentasi             contrastive level
                Contrasting)        manusia (UCR/HAR)  klasifikasi                                   weak/strong;           tunggal terhadap
                                                       few-label                                     single-resolution      TS2Vec yang
                                                                                                                            hierarkis

  Woo et        CoST (Contrastive   ETT, Electricity,  Forecasting deret    MSE/MAE forecasting      Berorientasi           Menunjukkan bahwa
  al. (2022)    Seasonal-Trend)     Weather            waktu jangka panjang                          forecasting, bukan     capaian metode
                                    (forecasting                                                     clustering/market      contrastive umumnya
                                    benchmark)                                                       state;                 diukur lewat
                                                                                                     single-resolution      forecasting, bukan
                                                                                                                            discovery kondisi
                                                                                                                            laten

  Yue et        TS2Vec              125 dataset UCR,   Universal            Akurasi klasifikasi, MSE Tidak diuji pada       Dipilih sebagai
  al. (2022)    (hierarchical       29 dataset UEA,    representation       forecasting, F1 deteksi  crypto; tidak          encoder baseline
                contextual          forecasting &      learning             anomali                  mengevaluasi variasi   penelitian ini
                contrastive         anomaly benchmark                                                resolusi temporal      (Subbab 2.5)
                learning)                                                                            sebagai variabel       
                                                                                                     terkontrol             

  Koki,         Bayesian Hidden     Bitcoin, Ether,    Deteksi rezim pasar  Log-likelihood,          Rezim laten diskrit    Digunakan sebagai
  Leonardos, &  Markov Model        Ripple (return     (bear/stable/bull)   karakteristik rezim      dengan asumsi          baseline eksternal
  Piliouras                         series)                                 (mean, variance return)  Markovian; tidak       (HMM) untuk
  (2022)                                                                                             mengintegrasikan       pembanding pada
                                                                                                     multi-resolusi; tidak  penelitian ini
                                                                                                     berbasis               (Subbab 3.6)
                                                                                                     representation         
                                                                                                     learning               

  Sobreiro,     Multi-timeframe     BTC/USDT, 4        Klasifikasi sinyal   ROC-AUC, return backtest Fitur rekayasa manual  Bukti bahwa
  Martinho,     feature             resolusi temporal  entry trading                                 (bukan representation  penelitian
  Martins, &    engineering +       (15m, 4h, 1d, 3d)                                                learning); tujuan      multi-timeframe
  Vardasca      Random Forest/ML                                                                     forecasting/trading,   finansial
  (2026)        klasik                                                                               bukan market state     berorientasi
                                                                                                     discovery              forecasting,
                                                                                                                            memperkuat research
                                                                                                                            gap
  ---------------------------------------------------------------------------------------------------------------------------------------------

*(Catatan: tabel di atas memuat penelitian inti yang menjadi dasar
argumen gap. Untuk kebutuhan submission SINTA/Scopus, tabel ini
disarankan diperluas hingga minimal 10 entri dengan menambahkan studi
pendukung lain sesuai arahan pembimbing/reviewer, misalnya studi HMM
tambahan atau studi clustering K-Means pada domain crypto.)*

### 2.9 Sintesis Literatur dan State of the Art

Sintesis pada Subbab 2.8 menunjukkan tiga pengamatan utama. **Pertama**,
garis keturunan metode *contrastive representation learning* untuk deret
waktu --- dari CPC (2018), ke TNC dan TS-TCC (2021), hingga CoST dan
TS2Vec (2022) --- secara konsisten dikembangkan dan divalidasi pada
domain umum (klasifikasi *benchmark*, *forecasting* deret waktu
non-finansial), bukan pada tugas *market state discovery*. **Kedua**,
metode identifikasi kondisi pasar yang mapan pada domain finansial,
khususnya HMM, tidak memanfaatkan kerangka *representation learning*
modern dan tidak mengintegrasikan berbagai resolusi temporal secara
sistematis. **Ketiga**, penelitian *multi-timeframe* pada domain
finansial yang sudah ada cenderung berorientasi pada *forecasting* atau
klasifikasi sinyal, menggunakan fitur rekayasa manual, bukan
representasi yang dipelajari secara *self-supervised*.

Berdasarkan ketiga pengamatan tersebut, penelitian ini menempatkan diri
pada irisan yang belum banyak dieksplorasi: menggunakan encoder
*contrastive representation learning* yang tervalidasi (TS2Vec) sebagai
instrumen, memperlakukan jumlah resolusi temporal sebagai variabel
independen yang dikontrol secara eksperimental (bukan sekadar
karakteristik dataset), dan mengevaluasi hasilnya pada tugas *market
state discovery* berbasis *clustering* dengan validasi ekonomi ---
sebuah kombinasi yang belum ditemukan secara langsung pada literatur
yang ditinjau.

**Tabel 2.2 Posisi Penelitian terhadap State of the Art**

  ----------------------------------------------------------------------------------------
  Penelitian        Contrastive   Multi-Resolusi   Market             Ablation   Domain
                    SSL           Temporal         State/Clustering   Jumlah TF  Crypto
  ----------------- ------------- ---------------- ------------------ ---------- ---------
  CPC (Van den Oord ✓             ---              ---                ---        ---
  et al., 2018)                                                                  

  TNC (Tonekaboni   ✓             ---              ---                ---        ---
  et al., 2021)                                                                  

  TS-TCC (Eldele et ✓             ---              ---                ---        ---
  al., 2021)                                                                     

  CoST (Woo et al., ✓             ---              ---                ---        ---
  2022)                                                                          

  TS2Vec (Yue et    ✓             ---              ---                ---        ---
  al., 2022)                                                                     

  HMM Regime        ---           ---              ✓                  ---        ✓
  Detection (Koki                                                                
  et al., 2022)                                                                  

  Multi-Timeframe   ---           ✓                ---                ---        ✓
  Feature                                                                        
  Engineering                                                                    
  (Sobreiro et al.,                                                              
  2026)                                                                          

  **Penelitian      **✓**         **✓ (statis,     **✓**              **✓**      **✓**
  Ini**                           terkontrol)**                                  
  ----------------------------------------------------------------------------------------

Berbeda dengan penelitian terdahulu yang berfokus pada pengembangan
model *contrastive* baru atau peningkatan performa
*forecasting*/klasifikasi, penelitian ini mengevaluasi secara empiris
apakah perubahan jumlah resolusi temporal memberikan pengaruh terhadap
kualitas representasi laten yang dipelajari model *representation
learning* yang sudah ada. Kontribusi penelitian tidak berupa arsitektur
*contrastive learning* baru, melainkan bukti empiris --- melalui
protokol eksperimen terkontrol dan pengujian statistik *pre-registered*
--- mengenai peran resolusi temporal dalam pembelajaran representasi
pasar cryptocurrency.

### 2.10 Kerangka Pemikiran

Penelitian ini berangkat dari karakteristik pasar cryptocurrency yang
memiliki dinamika tinggi dan berlangsung pada berbagai resolusi
temporal. Setiap resolusi temporal diperkirakan membawa informasi yang
berbeda mengenai perilaku pasar sehingga penggunaan beberapa resolusi
temporal berpotensi menghasilkan representasi yang lebih kaya
dibandingkan penggunaan satu resolusi temporal.

Di sisi lain, perkembangan *Self-Supervised Learning* berbasis
*contrastive learning* (Subbab 2.4) memungkinkan model mempelajari
representasi data tanpa memerlukan label. TS2Vec dipilih sebagai
instrumen pembelajaran representasi karena mampu menghasilkan
representasi laten yang dapat dimanfaatkan pada berbagai *downstream
task* (Subbab 2.5). Pada penelitian ini, TS2Vec tidak diposisikan
sebagai objek penelitian, tetapi sebagai alat untuk mempelajari
representasi pasar cryptocurrency.

Hasil sintesis literatur (Subbab 2.8--2.9) menunjukkan bahwa sebagian
besar penelitian *contrastive representation learning* belum diuji pada
konteks *multi-resolution market state discovery*, sedangkan penelitian
*multi-timeframe* finansial yang ada berorientasi pada *forecasting*.
Oleh karena itu, penelitian ini menempatkan jumlah resolusi temporal
sebagai variabel independen, sedangkan seluruh komponen eksperimen
lainnya dipertahankan tetap agar perubahan hasil dapat diinterpretasikan
sebagai pengaruh resolusi temporal.

Secara konseptual, penelitian ini dibangun atas asumsi bahwa perubahan
jumlah resolusi temporal akan memengaruhi informasi yang dipelajari oleh
model sehingga berpotensi menghasilkan kualitas representasi laten yang
berbeda. Representasi tersebut selanjutnya digunakan untuk membentuk
*market state* melalui proses *clustering* dan dievaluasi menggunakan
metrik geometrik serta validasi karakteristik ekonomi.

**Gambar 2.2 Kerangka Pemikiran Penelitian**

![](media/image1.png){width="5.5in" height="7.209211504811899in"}

**Fig. 2.** Conceptual framework of the proposed multi-resolution
temporal representation learning approach for cryptocurrency market
state discovery.

### 2.11 Hipotesis Penelitian

Berdasarkan rumusan masalah, kajian teori, dan sintesis literatur,
hipotesis penelitian dirumuskan sebagai berikut. Mengikuti prinsip
*pre-registered analysis* untuk menghindari *post-hoc condition
selection* (yaitu memilih kondisi "terbaik" setelah melihat hasil, yang
berisiko menimbulkan *p-hacking*), hipotesis dirumuskan sebagai **tiga
perbandingan berpasangan yang telah ditetapkan sebelum eksperimen
dijalankan**, masing-masing membandingkan satu kondisi multi-resolusi
terhadap *baseline* 1TF.

**H0 (Hipotesis Nol)**

Tidak terdapat perbedaan kualitas *clustering* yang signifikan secara
statistik antara kondisi resolusi tunggal (1TF) dan kondisi
multi-resolusi, sebagaimana diukur oleh *Silhouette Score*
($p \geq 0,05$, Wilcoxon *signed-rank test* satu sisi, setelah koreksi
*multiple comparison*).

**H1 (Hipotesis Alternatif / Primer)**

Input temporal multi-resolusi menghasilkan kualitas *clustering* yang
signifikan lebih baik secara statistik dibandingkan input resolusi
tunggal. Hipotesis ini dievaluasi secara terpisah melalui tiga
perbandingan berpasangan yang telah ditetapkan sebelumnya
(*pre-specified*):

-   **H1a:** 2TF menghasilkan *Silhouette Score* yang signifikan lebih
    tinggi dibandingkan 1TF
-   **H1b:** 3TF menghasilkan *Silhouette Score* yang signifikan lebih
    tinggi dibandingkan 1TF
-   **H1c:** 4TF menghasilkan *Silhouette Score* yang signifikan lebih
    tinggi dibandingkan 1TF

Ketiga perbandingan diuji menggunakan Wilcoxon *signed-rank test* satu
sisi (*alternative* = *greater*), dengan tingkat signifikansi
$\alpha = 0,05$ setelah koreksi Holm-Bonferroni diterapkan terhadap
ketiga nilai-$p$ secara bersamaan.

**H2 (Hipotesis Eksploratif)**

Hubungan antara jumlah resolusi temporal aktif dan kualitas *clustering*
bersifat tidak monoton (*non-monotonic*) --- penambahan resolusi
temporal melampaui titik tertentu berpotensi tidak meningkatkan, atau
bahkan menurunkan, kualitas *clustering*. Hipotesis ini bersifat
eksploratif dan **tidak memengaruhi kesimpulan statistik primer** yang
didasarkan pada H1a--H1c. H2 dievaluasi secara deskriptif melalui kurva
*Silhouette Score* terhadap jumlah *timeframe* aktif (Gambar 4, lihat
Subbab 3.7).

Pengujian hipotesis primer (H1a--H1c) dilakukan menggunakan metrik
*Silhouette Score* sebagai metrik utama, didukung oleh *Davies-Bouldin
Index* dan *Calinski-Harabasz Index* sebagai metrik pelengkap untuk
memperkuat interpretasi kualitas *clustering*. Selanjutnya, hasil
*clustering* divalidasi melalui analisis karakteristik ekonomi pada
setiap *market state* yang terbentuk (Kruskal-Wallis test, $p < 0,05$)
untuk memastikan bahwa representasi laten yang dihasilkan tidak hanya
terpisah secara geometrik, tetapi juga memiliki makna dalam
merepresentasikan perilaku pasar cryptocurrency.

# BAB III

## METODOLOGI PENELITIAN

### 3.1 Desain Penelitian

Penelitian ini menggunakan pendekatan kuantitatif dengan metode
eksperimen yang dirancang sebagai *controlled empirical study*.
Pendekatan ini dipilih untuk menginvestigasi pengaruh jumlah resolusi
temporal terhadap kualitas representasi laten yang dipelajari oleh model
*Self-Supervised Learning* pada *cryptocurrency market state discovery*.
Berbeda dengan penelitian yang berorientasi pada pengembangan model
baru, penelitian ini menempatkan resolusi temporal sebagai variabel
independen yang dievaluasi secara sistematis melalui serangkaian
eksperimen terkontrol.

Seluruh skenario eksperimen menggunakan konfigurasi yang identik,
meliputi dataset, proses *preprocessing*, arsitektur TS2Vec, mekanisme
*representation learning*, strategi *late fusion*, dimensi representasi
akhir, algoritma *clustering*, prosedur evaluasi, serta metode analisis
statistik. Dengan demikian, jumlah resolusi temporal menjadi
satu-satunya variabel yang dibedakan antar kondisi eksperimen sehingga
setiap perubahan hasil dapat diinterpretasikan sebagai pengaruh dari
variasi resolusi temporal, bukan akibat perubahan konfigurasi sistem.

Penelitian terdiri atas empat kondisi eksperimen utama (1TF, 2TF, 3TF,
4TF) yang disusun secara kumulatif berdasarkan *anchor* 1 jam, ditambah
tiga *baseline* sekunder berresolusi tunggal ({15m}, {4h}, {1d}) untuk
mengisolasi kontribusi masing-masing *timeframe* secara individual,
serta dua metode pembanding eksternal (HMM dan K-Means+PCA). Setiap
kondisi dibangun secara kumulatif sehingga penambahan resolusi temporal
dilakukan secara bertahap tanpa mengubah konfigurasi eksperimen lainnya.
Pendekatan tersebut memungkinkan evaluasi dilakukan secara konsisten
terhadap hubungan antara jumlah resolusi temporal dan kualitas
representasi laten yang dihasilkan.

Secara umum, penelitian dilaksanakan melalui sembilan tahapan utama,
yaitu akuisisi data, validasi data, *temporal alignment*, *feature
engineering*, *temporal split*, *window generation*, pembelajaran
representasi menggunakan TS2Vec, pembentukan *market state* melalui
HDBSCAN, serta evaluasi dan analisis statistik terhadap hasil eksperimen
(lihat Subbab 3.3 untuk rincian lengkap). Seluruh tahapan tersebut
membentuk satu alur penelitian yang terintegrasi mulai dari data
historis hingga penarikan kesimpulan ilmiah.

**Variabel Penelitian**

Penelitian ini menggunakan struktur variabel sebagai berikut.

**Tabel 3.0 Struktur Variabel Penelitian**

  -----------------------------------------------------------------------
  Jenis Variabel          Komponen                Nilai / Konfigurasi
  ----------------------- ----------------------- -----------------------
  Variabel independen     Jumlah resolusi         {1h}, {15m,1h},
                          temporal aktif          {15m,1h,4h},
                                                  {15m,1h,4h,1d}

  Variabel dependen       Kualitas representasi   Silhouette Score,
                          laten                   Davies-Bouldin Index,
                                                  Calinski-Harabasz
                                                  Index, jumlah
                                                  *cluster*, persentase
                                                  *noise*, nilai-$p$
                                                  Kruskal-Wallis

  Variabel terkendali     Dataset, fitur, *window Identik pada semua
                          size*, *stride*,        kondisi
                          normalisasi, encoder    
                          TS2Vec, strategi *late  
                          fusion*, dimensi        
                          representasi (256-dim), 
                          *fixed random           
                          projection*, HDBSCAN,   
                          prosedur evaluasi,      
                          analisis statistik      
  -----------------------------------------------------------------------

Dengan struktur tersebut, setiap perubahan pada variabel dependen dapat
secara kausal dikaitkan dengan variasi variabel independen (jumlah
resolusi temporal), karena seluruh faktor perancu telah dikendalikan
secara eksplisit.

### 3.2 Objek dan Dataset Penelitian

Objek penelitian ini adalah pasar cryptocurrency Bitcoin (BTC/USDT) yang
diperdagangkan pada bursa Binance. Bitcoin dipilih karena memiliki
likuiditas yang tinggi, volume perdagangan yang relatif stabil, serta
ketersediaan data historis yang lengkap sehingga banyak digunakan
sebagai objek penelitian pada domain analisis pasar cryptocurrency.
Seluruh data historis diperoleh melalui Binance menggunakan pustaka
CCXT, dengan *python-binance* sebagai alternatif apabila diperlukan.

Dataset yang digunakan berupa data OHLCV (*Open, High, Low, Close,
Volume*) selama periode 1 Januari 2020 hingga 31 Desember 2023. Rentang
waktu tersebut dipilih agar mencakup berbagai kondisi pasar, termasuk
fase *bullish*, *bearish*, dan konsolidasi, sehingga representasi yang
dipelajari model berasal dari dinamika pasar yang beragam.

**Tabel 3.1 Karakteristik Dataset Penelitian**

  -----------------------------------------------------------------------
  Komponen                            Keterangan
  ----------------------------------- -----------------------------------
  Objek penelitian                    Bitcoin (BTC/USDT)

  Bursa                               Binance

  Jenis data                          OHLCV (*Open, High, Low, Close,
                                      Volume*)

  Sumber data                         Binance melalui pustaka CCXT

  Periode data                        1 Januari 2020 -- 31 Desember 2023
  -----------------------------------------------------------------------

Untuk mengevaluasi pengaruh resolusi temporal, penelitian menggunakan
empat resolusi temporal, yaitu 15 menit (15m), 1 jam (1h), 4 jam (4h),
dan 1 hari (1d). Keempat resolusi tersebut dipilih untuk
merepresentasikan informasi pasar dari jangka pendek hingga jangka
panjang. Seluruh data kemudian diselaraskan menggunakan *1-hour anchor*
sehingga setiap observasi pada seluruh resolusi temporal
merepresentasikan titik waktu yang sama tanpa menimbulkan *look-ahead
leakage*.

**Tabel 3.2 Konfigurasi Resolusi Temporal**

  -----------------------------------------------------------------------
  Timeframe               Representasi            Keterangan
  ----------------------- ----------------------- -----------------------
  15 menit (15m)          Jangka pendek           Diagregasi ke *1-hour
                                                  anchor* (4 candle → 1
                                                  candle)

  1 jam (1h)              Jangka menengah         *Anchor timeframe* ---
                                                  digunakan langsung

  4 jam (4h)              Jangka menengah-panjang Diselaraskan
                                                  menggunakan
                                                  *forward-fill* (1
                                                  candle → 4 jam)

  1 hari (1d)             Jangka panjang          Diselaraskan
                                                  menggunakan
                                                  *forward-fill* (1
                                                  candle → 24 jam)
  -----------------------------------------------------------------------

Untuk menjaga validitas eksperimen, penelitian tidak menggunakan
*technical indicator* seperti RSI, MACD, maupun Bollinger Bands.
Keputusan ini diambil agar informasi yang dipelajari model berasal dari
variasi resolusi temporal, bukan dari proses *smoothing* maupun
parameter *window* yang melekat pada indikator teknikal. Dataset
kemudian dipisahkan menggunakan *walk-forward temporal split* menjadi
data pelatihan (2020--2022) dan data pengujian (2023).

**Tabel 3.3 Konfigurasi Dataset Penelitian**

  -----------------------------------------------------------------------
  Komponen                            Konfigurasi
  ----------------------------------- -----------------------------------
  Metode penyelarasan                 1-hour temporal alignment

  15m → 1h                            Agregasi OHLCV (OHLC: open dari
                                      candle pertama, high dari max, low
                                      dari min, close dari candle
                                      terakhir, volume: sum)

  4h → 1h                             Forward-fill (tanpa interpolasi)

  1d → 1h                             Forward-fill (tanpa interpolasi)

  Feature set                         7 fitur turunan OHLCV

  Technical Indicator                 Tidak digunakan

  Metode pembagian data               Walk-forward temporal split

  Data pelatihan                      2020-01-01 -- 2022-12-31 (\~26.269
                                      observasi jam setelah penghapusan
                                      baris NaN)

  Data pengujian                      2023-01-01 -- 2023-12-31 (\~8.760
                                      observasi jam)

  Shuffle                             Tidak digunakan
  -----------------------------------------------------------------------

### 3.3 Tahapan Penelitian

Secara umum, penelitian terdiri atas sembilan tahapan utama sebagaimana
ditunjukkan pada Gambar 3.2 berikut.

    Akuisisi Data
          │
          ▼
    Validasi Data
          │
          ▼
    Temporal Alignment
          │
          ▼
    Feature Engineering
          │
          ▼
    Temporal Split
          │
          ▼
    Window Generation
          │
          ▼
    Representation Learning (TS2Vec, per-branch independen)
          │
          ▼
    Late Fusion + Fixed Random Projection → Market State Discovery (HDBSCAN)
          │
          ▼
    Evaluasi Geometrik + Validasi Ekonomi + Analisis Statistik

**Tabel 3.4 Tahapan Penelitian**

  -----------------------------------------------------------------------
  Tahap                   Proses                  Output
  ----------------------- ----------------------- -----------------------
  1                       Akuisisi data historis  Raw OHLCV Dataset
                          OHLCV dari Binance      

  2                       Validasi integritas     Validated Dataset
                          data                    

  3                       Temporal alignment      Aligned Dataset
                          menggunakan 1-hour      
                          anchor                  

  4                       Feature engineering     Feature Dataset
                          menggunakan tujuh fitur 
                          turunan OHLCV           

  5                       Walk-forward temporal   Train & Test Dataset
                          split menjadi data      
                          pelatihan dan pengujian 

  6                       Pembentukan sliding     Window Dataset
                          window dan normalisasi  
                          per-window z-score      

  7                       Pembelajaran            Latent Embeddings per
                          representasi            Branch
                          menggunakan TS2Vec pada 
                          setiap timeframe        
                          (independen)            

  8                       Late Fusion, Fixed      Market State Labels
                          Random Projection,      
                          Market State Discovery  
                          menggunakan HDBSCAN     

  9                       Evaluasi geometrik,     Hasil Eksperimen
                          validasi ekonomi, dan   
                          pengujian statistik     
  -----------------------------------------------------------------------

### 3.4 Pembelajaran Representasi (*Representation Learning*)

Tahap pembelajaran representasi bertujuan menghasilkan representasi
laten (*latent representation*) dari setiap resolusi temporal
menggunakan TS2Vec. Sebelum memasuki tahap ini, dataset hasil *feature
engineering* dibentuk menjadi kumpulan *sliding window* dengan panjang
48 observasi dan *stride* sebesar 1. Setiap *window* hanya memuat tujuh
fitur turunan OHLCV dari satu resolusi temporal. Selanjutnya, setiap
*window* dinormalisasi menggunakan *per-window z-score normalization*.

#### 3.4.1 Feature Engineering

Tujuh fitur direkayasa secara langsung dari data OHLCV tanpa menggunakan
*technical indicator*. Seluruh fitur dirancang untuk merepresentasikan
karakteristik pergerakan harga pada setiap periode secara independen
dari parameter *smoothing* atau *lookback* yang berpotensi mengandung
informasi dari resolusi temporal lain.

**Persamaan (3.1) sampai (3.7) --- Definisi Tujuh Fitur OHLCV**

$$f_{1} = \frac{o_{t} - c_{t - 1}}{c_{t - 1}}\quad\text{(open\_return)}$$

$$f_{2} = \frac{h_{t} - o_{t}}{o_{t}}\quad\text{(high\_return)}$$

$$f_{3} = \frac{l_{t} - o_{t}}{o_{t}}\quad\text{(low\_return)}$$

$$f_{4} = \frac{c_{t} - o_{t}}{o_{t}}\quad\text{(close\_return)}$$

$$f_{5} = \frac{v_{t} - \mu_{v}^{(20)}}{\sigma_{v}^{(20)}}\quad\text{(volume\_zscore)}$$

$$f_{6} = \frac{h_{t} - l_{t}}{o_{t}}\quad\text{(hl\_range)}$$

$$f_{7} = \frac{\left| c_{t} - o_{t} \right|}{h_{t} - l_{t} + \epsilon}\quad\text{(body\_ratio)},\quad\epsilon = 10^{- 8}$$

Keterangan:

-   $o_{t},h_{t},l_{t},c_{t}$ : harga *open*, *high*, *low*, *close*
    pada periode $t$
-   $v_{t}$ : volume perdagangan pada periode $t$
-   $\mu_{v}^{(20)},\sigma_{v}^{(20)}$ : rata-rata dan simpangan baku
    volume pada *rolling window* 20 periode, dihitung pada resolusi asli
    (*native*) masing-masing *timeframe*
-   $\epsilon$ : konstanta kecil untuk menghindari pembagian dengan nol
    pada $f_{7}$

#### 3.4.2 Per-Window Z-Score Normalization

Setiap *sliding window* dinormalisasi secara independen menggunakan
*z-score normalization* agar model tidak menerima informasi absolut dari
harga, melainkan hanya karakteristik distribusi relatif dari setiap
*window*. Normalisasi ini hanya menggunakan statistik dari dalam
*window* itu sendiri, sehingga tidak memerlukan statistik dari data
*training* dan tidak berisiko menimbulkan kebocoran informasi
(*leakage*) lintas *split*.

**Persamaan (3.8) --- Per-Window Z-Score Normalization**

$${\widehat{x}}_{t,f} = \frac{x_{t,f} - \mu_{f}^{(w)}}{\sigma_{f}^{(w)} + \epsilon}$$

Keterangan:

-   $x_{t,f}$ : nilai fitur ke-$f$ pada langkah waktu ke-$t$ dalam
    *window* $w$

-   $\mu_{f}^{(w)} = \frac{1}{W}\sum_{t = 1}^{W}x_{t,f}$ : rata-rata
    fitur ke-$f$ dalam *window* $w$

-   ::: Definition-Term
    $\sigma_{f}^{(w)} = \sqrt{\frac{1}{W}\sum_{t = 1}^{W}\left( x_{t,f} - \mu_{f}^{(w)} \right)^{2}}$
    :::

```{=html}
<!-- -->
```
-   simpangan baku fitur ke-$f$ dalam *window* $w$

```{=html}
<!-- -->
```
-   $W = 48$ : panjang *window*

**Tabel 3.5 Konfigurasi Data Masukan TS2Vec**

  -----------------------------------------------------------------------
  Komponen                            Konfigurasi
  ----------------------------------- -----------------------------------
  Panjang *window*                    48 observasi

  Jumlah fitur                        7 fitur turunan OHLCV

  Bentuk data                         Tensor 3D (*Window × Time Step ×
                                      Feature*)

  Normalisasi                         Per-window z-score

  Stride                              1
  -----------------------------------------------------------------------

Setiap resolusi temporal diproses menggunakan encoder TS2Vec yang
dilatih secara independen tanpa berbagi parameter maupun gradien dengan
encoder lainnya. Pendekatan ini memastikan bahwa setiap encoder hanya
mempelajari karakteristik temporal dari resolusi yang bersangkutan.
Konfigurasi encoder ditunjukkan pada Tabel 3.6.

**Tabel 3.6 Konfigurasi Encoder TS2Vec**

  -----------------------------------------------------------------------
  Komponen                            Konfigurasi
  ----------------------------------- -----------------------------------
  Input dimension                     7

  Hidden dimension                    64

  Output dimension                    64

  Depth (dilated CNN layers)          10

  Kernel size                         3

  Mask ratio                          0.5 (default TS2Vec)

  Optimizer                           AdamW

  Learning rate                       $1 \times 10^{- 3}$

  Weight decay                        $1 \times 10^{- 4}$

  Batch size                          8

  Maksimum epoch                      50

  Early stopping patience             10
  -----------------------------------------------------------------------

**Tabel 3.7 Keluaran Encoder TS2Vec per Branch**

  -----------------------------------------------------------------------
  Tahap                               Bentuk Keluaran
  ----------------------------------- -----------------------------------
  Data masukan                        $\lbrack 48 \times 7\rbrack$

  Representasi temporal TS2Vec        $\lbrack 48 \times 64\rbrack$

  *Temporal max-pooling*              $\lbrack 64\rbrack$
  -----------------------------------------------------------------------

### 3.5 Fusion Representasi dan *Market State Discovery*

Setelah setiap resolusi temporal menghasilkan representasi laten secara
independen, seluruh representasi digabungkan menggunakan pendekatan
*late concatenation fusion*. Berbeda dengan *early fusion* yang
menggabungkan data pada tingkat masukan, pendekatan ini menggabungkan
representasi setelah setiap encoder menyelesaikan proses pembelajaran.
Urutan penggabungan ditetapkan secara tetap (15m → 1h → 4h → 1d) dan
diterapkan secara konsisten pada seluruh kondisi eksperimen.

Karena jumlah resolusi temporal berbeda pada setiap kondisi eksperimen,
dimensi hasil konkatenasi juga berbeda. Oleh karena itu, seluruh
representasi hasil *late fusion* diproyeksikan ke dimensi tetap 256
menggunakan *fixed random projection* (lihat Persamaan 2.6). Pendekatan
ini mempertahankan dimensi representasi akhir tetap sama tanpa
menambahkan parameter yang dipelajari. Sebagaimana dijelaskan pada
Subbab 2.6, proyeksi ini tidak mengubah rank efektif representasi ---
keterbatasan ini diakui secara eksplisit dan dibahas pada bagian
Keterbatasan Penelitian.

**Tabel 3.8 Konfigurasi Fusion Representasi**

  -----------------------------------------------------------------------
  Kondisi           Timeframe Aktif   Dimensi           Dimensi Akhir
                                      Konkatenasi       
  ----------------- ----------------- ----------------- -----------------
  1TF               {1h}              64                256

  2TF               {15m, 1h}         128               256

  3TF               {15m, 1h, 4h}     192               256

  4TF               {15m, 1h, 4h, 1d} 256               256
  -----------------------------------------------------------------------

Representasi akhir hasil *fusion* kemudian digunakan sebagai masukan
pada proses *market state discovery* menggunakan HDBSCAN. Untuk menjaga
validitas eksperimen, penelitian menggunakan protokol dua tahap pada
proses penentuan parameter HDBSCAN. Tahap pertama merupakan eksperimen
utama, yaitu melakukan *grid search* hanya pada kondisi 1TF untuk
memperoleh kombinasi parameter terbaik. Parameter yang diperoleh
kemudian dikunci (*locked*) dan digunakan tanpa perubahan pada seluruh
kondisi eksperimen berikutnya. Sebagai analisis tambahan (*robustness
check*, opsional), penelitian juga dapat melakukan *sensitivity
analysis* dengan melakukan *grid search* secara independen pada setiap
kondisi eksperimen. Analisis ini, jika dilakukan, tidak digunakan
sebagai dasar penarikan kesimpulan utama dan dilaporkan secara terpisah.

**Tabel 3.9 Konfigurasi HDBSCAN**

  -----------------------------------------------------------------------
  Komponen                            Konfigurasi
  ----------------------------------- -----------------------------------
  Algoritma                           HDBSCAN

  Metrik jarak                        Euclidean

  Grid search                         Dilakukan hanya pada kondisi 1TF
                                      (Stage 1 --- primer)

  Parameter utama                     min_cluster_size
                                      $\in \{ 50,100,200\}$; min_samples
                                      $\in \{ 5,10,20\}$

  Parameter eksperimen utama          Dikunci berdasarkan Silhouette
                                      tertinggi pada 1TF (dengan
                                      $2 \leq k \leq 8$); digunakan tetap
                                      untuk 2TF, 3TF, 4TF

  Jumlah cluster maksimum             8 (constraint interpretabilitas)

  *Sensitivity analysis* (Stage 2)    Opsional, *robustness check*;
                                      dilaporkan terpisah sebagai
                                      analisis tambahan dan tidak
                                      memengaruhi kesimpulan primer
  -----------------------------------------------------------------------

### 3.6 Skenario Eksperimen

Penelitian menggunakan empat kondisi eksperimen utama yang disusun
secara kumulatif. Kondisi 1TF digunakan sebagai *baseline*, sedangkan
kondisi 2TF, 3TF, dan 4TF digunakan untuk mengevaluasi pengaruh
penambahan resolusi temporal terhadap kualitas representasi laten.

**Tabel 3.10 Kondisi Eksperimen Utama**

  -----------------------------------------------------------------------
  Kondisi                 Timeframe Aktif         Tujuan
  ----------------------- ----------------------- -----------------------
  1TF                     {1h}                    Baseline penelitian

  2TF                     {15m, 1h}               Evaluasi pengaruh
                                                  penambahan resolusi
                                                  granular

  3TF                     {15m, 1h, 4h}           Evaluasi pengaruh
                                                  penambahan resolusi
                                                  mid-scale

  4TF                     {15m, 1h, 4h, 1d}       Evaluasi pengaruh
                                                  multi-resolution penuh
  -----------------------------------------------------------------------

Selain empat kondisi utama, penelitian juga menggunakan tiga *baseline*
sekunder, yaitu eksperimen yang hanya menggunakan satu *timeframe* 15
menit, 4 jam, dan 1 hari. *Baseline* sekunder ini tidak digunakan untuk
menjawab pertanyaan penelitian utama, tetapi memberikan gambaran
kontribusi masing-masing resolusi temporal terhadap kualitas
representasi secara individual. Dengan demikian, total terdapat **tujuh
kondisi unik berbasis TS2Vec** (empat kondisi primer kumulatif + tiga
*baseline* sekunder), karena kondisi 1TF dan *baseline* {1h} merujuk
pada kondisi dan *branch* yang identik dan tidak dihitung dua kali.

Penelitian juga menggunakan dua metode pembanding eksternal, yaitu HMM
dan K-Means + PCA, yang dijalankan dengan protokol evaluasi dan protokol
*seed* yang identik dengan kondisi utama untuk memastikan perbandingan
yang adil.

**Tabel 3.11 Variabel yang Dikendalikan (Controlled Variables)**

  -----------------------------------------------------------------------
  Komponen                            Konfigurasi
  ----------------------------------- -----------------------------------
  Arsitektur encoder                  TS2Vec (output 64-dim per branch)

  Dimensi embedding akhir             256

  Window size                         48

  Stride                              1

  Jumlah fitur                        7 fitur turunan OHLCV

  Normalisasi                         Per-window z-score

  Optimizer                           AdamW, lr=$10^{- 3}$, wd=$10^{- 4}$

  Batch size                          8

  Maksimum epoch                      50

  Early stopping patience             10

  Fusion                              Late concatenation + fixed random
                                      projection (seed=42)

  Algoritma clustering                HDBSCAN (parameter dikunci dari
                                      1TF)

  Metrik jarak                        Euclidean

  Dataset                             BTC/USDT Binance (2020--2023)

  Random seed                         5 seed: {42, 123, 456, 789, 1024}
  -----------------------------------------------------------------------

Seluruh eksperimen dijalankan menggunakan lima *random seed* yang
berbeda, diterapkan secara identik pada seluruh sumber keacakan (Python,
NumPy, PyTorch, CUDA) dan pada kedua metode pembanding eksternal. Nilai
setiap metrik dilaporkan dalam bentuk rata-rata (*mean*) dan simpangan
baku (*standard deviation*) untuk meningkatkan reliabilitas hasil
penelitian.

**Tabel 3.11a Total Jumlah Eksperimen**

  -----------------------------------------------------------------------
  Komponen          Jumlah Kondisi    Jumlah Seed       Total Run
  ----------------- ----------------- ----------------- -----------------
  Kondisi primer    4                 5                 20
  (1TF, 2TF, 3TF,                                       
  4TF)                                                  

  Baseline sekunder 3                 5                 15
  ({15m}, {4h},                                         
  {1d})                                                 

  **Subtotal TS2Vec **7**             **5**             **35**
  (7 kondisi                                            
  unik)**                                               

  Baseline          2                 5                 10
  eksternal (HMM,                                       
  K-Means+PCA)                                          

  **Total           **9**             **5**             **45**
  Keseluruhan**                                         
  -----------------------------------------------------------------------

### 3.7 Evaluasi Kualitas Representasi

Evaluasi pada penelitian ini bertujuan untuk mengukur pengaruh jumlah
resolusi temporal terhadap kualitas representasi laten yang dihasilkan
oleh TS2Vec. Evaluasi dilakukan menggunakan dua pendekatan, yaitu
evaluasi geometrik dan validasi karakteristik ekonomi.

#### 3.7.1 Evaluasi Geometrik

Evaluasi geometrik mengukur kualitas pemisahan *cluster* pada ruang
representasi laten menggunakan tiga metrik: *Silhouette Score*,
*Davies-Bouldin Index*, dan *Calinski-Harabasz Index*. Seluruh metrik
dihitung pada *embedding* hasil pelatihan dengan titik *noise* (*label*
$= - 1$ dari HDBSCAN) dikecualikan dari perhitungan.

**Persamaan (3.9) --- Silhouette Score** *(Rousseeuw, 1987)*

$$s(i) = \frac{b(i) - a(i)}{\max\{ a(i),b(i)\}}$$

$$\text{Silhouette} = \frac{1}{N}\sum_{i = 1}^{N}s(i)$$

Keterangan:

-   $a(i)$ : rata-rata jarak titik $i$ terhadap seluruh titik lain dalam
    *cluster* yang sama (*intra-cluster distance*)
-   $b(i)$ : rata-rata jarak minimum titik $i$ terhadap titik-titik pada
    *cluster* tetangga terdekat (*inter-cluster distance*)
-   $N$ : total jumlah titik data (tidak termasuk titik *noise*)
-   Rentang: $\lbrack - 1,1\rbrack$; nilai lebih tinggi menunjukkan
    kualitas *clustering* yang lebih baik

**Persamaan (3.10) --- Davies-Bouldin Index** *(Davies & Bouldin, 1979)*

$$\text{DBI} = \frac{1}{K}\sum_{i = 1}^{K}\max_{j \neq i}\left( \frac{\sigma_{i} + \sigma_{j}}{d\left( c_{i},c_{j} \right)} \right)$$

Keterangan:

-   $K$ : jumlah *cluster* (tidak termasuk *noise*)
-   $\sigma_{i}$ : rata-rata jarak titik-titik dalam *cluster* $i$
    terhadap centroid $c_{i}$
-   $d\left( c_{i},c_{j} \right)$ : jarak Euclidean antara centroid
    *cluster* $i$ dan $j$
-   Interpretasi: nilai lebih rendah menunjukkan pemisahan *cluster*
    yang lebih baik

**Persamaan (3.11) --- Calinski-Harabasz Index** *(Calinski & Harabasz,
1974)*

$$\text{CH} = \frac{tr\left( B_{K} \right)/(K - 1)}{tr\left( W_{K} \right)/(N - K)}$$

Keterangan:

-   $B_{K}$ : matriks *between-cluster scatter* --- mengukur separasi
    antar *cluster*
-   $W_{K}$ : matriks *within-cluster scatter* --- mengukur kekompakan
    dalam *cluster*
-   $tr( \cdot )$ : jejak (*trace*) matriks
-   Interpretasi: nilai lebih tinggi menunjukkan *cluster* yang lebih
    terseparasi dan kompak

**Tabel 3.12 Metrik Evaluasi Geometrik**

  -----------------------------------------------------------------------
  Metrik                  Tujuan Evaluasi         Interpretasi
  ----------------------- ----------------------- -----------------------
  Silhouette Score        Mengukur kekompakan dan Semakin tinggi semakin
                          pemisahan *cluster*;    baik
                          metrik utama untuk      
                          pengujian H1a--H1c      

  Davies-Bouldin Index    Mengukur kesamaan antar Semakin rendah semakin
                          *cluster*; metrik       baik
                          pendukung               

  Calinski-Harabasz Index Mengukur rasio separasi Semakin tinggi semakin
                          dan kekompakan          baik
                          *cluster*; metrik       
                          pendukung               
  -----------------------------------------------------------------------

#### 3.7.2 Validasi Karakteristik Ekonomi

Validasi karakteristik ekonomi bertujuan memastikan bahwa kelompok
*market state* yang terbentuk tidak hanya terpisah secara matematis,
tetapi juga memiliki karakteristik pasar yang berbeda sehingga dapat
diinterpretasikan sebagai kondisi pasar yang bermakna. Validasi
dilakukan menggunakan analisis statistik deskriptif terhadap distribusi
*return* pada setiap *market state*, serta pengujian perbedaan
distribusi menggunakan Kruskal-Wallis test ($p < 0,05$).

**Tabel 3.13 Validasi Karakteristik Ekonomi**

  -----------------------------------------------------------------------
  Komponen                            Tujuan
  ----------------------------------- -----------------------------------
  *Return* rata-rata (*mean return*)  Mengidentifikasi arah pergerakan
                                      dominan pada setiap *market state*

  Volatilitas (*standard deviation    Mengukur tingkat fluktuasi harga
  return*)                            pada setiap *market state*

  *Skewness return*                   Mengevaluasi asimetri distribusi
                                      *return* per *market state*

  Kruskal-Wallis test                 Menguji apakah distribusi *return*
                                      antar *market state* berbeda secara
                                      signifikan
  -----------------------------------------------------------------------

### 3.8 Analisis Statistik dan Pengujian Hipotesis

Analisis statistik dilakukan untuk menguji apakah perbedaan kualitas
representasi laten antar kondisi eksperimen bersifat signifikan secara
statistik. Setiap kondisi eksperimen dijalankan menggunakan lima *random
seed* yang berbeda. Nilai setiap metrik evaluasi kemudian diringkas
dalam bentuk rata-rata (*mean*) dan simpangan baku (*standard
deviation*) di antara kelima *seed* tersebut.

Mengikuti prinsip pengujian *pre-registered* sebagaimana dirumuskan pada
Subbab 2.11, perbedaan antar kondisi eksperimen diuji menggunakan **tiga
perbandingan berpasangan yang ditetapkan sebelum eksperimen
dijalankan**: 2TF vs 1TF, 3TF vs 1TF, dan 4TF vs 1TF. Pendekatan ini
dipilih untuk menggantikan strategi *post-hoc* "kondisi multi-TF terbaik
vs. 1TF", karena strategi *post-hoc* tersebut memilih kondisi pembanding
berdasarkan hasil yang telah diamati terlebih dahulu --- sebuah praktik
yang berisiko menimbulkan *p-hacking* dan melemahkan validitas
kesimpulan statistik. Setiap perbandingan diuji menggunakan *Wilcoxon
Signed-Rank Test* satu sisi, dipilih karena tidak mengharuskan data
mengikuti distribusi normal serta sesuai untuk membandingkan dua kondisi
berpasangan pada sampel kecil ($n = 5$ *seed*).

**Persamaan (3.12) --- Statistik Uji Wilcoxon Signed-Rank**

$$W = \min\left( W^{+},W^{-} \right)$$

dengan:

$$W^{+} = \sum_{i:\, d_{i} > 0}^{}R_{i},\quad\quad W^{-} = \sum_{i:\, d_{i} < 0}^{}R_{i}$$

Keterangan:

-   $d_{i} = x_{\text{multi-TF},i} - x_{\text{1TF},i}$ : selisih
    *Silhouette Score* antara kondisi multi-TF yang diuji (2TF, 3TF,
    atau 4TF) dan kondisi 1TF pada *seed* berpasangan ke-$i$ (indeks
    *seed* yang sama dibandingkan antar kondisi)
-   $R_{i}$ : peringkat (*rank*) dari $\left| d_{i} \right|$ setelah
    selisih nol diabaikan
-   $W^{+}$ : jumlah peringkat dari selisih bernilai positif
-   $W^{-}$ : jumlah peringkat dari selisih bernilai negatif
-   Pengujian bersifat satu sisi (*alternative* = *greater*), menguji
    apakah kondisi multi-TF secara konsisten lebih unggul dibandingkan
    1TF
-   Keputusan: tolak $H_{0}$ untuk perbandingan tersebut apabila
    $p < \alpha = 0,05$ (setelah koreksi Holm-Bonferroni)

Karena penelitian melakukan tiga pengujian secara bersamaan (2TF vs 1TF,
3TF vs 1TF, 4TF vs 1TF), koreksi *multiple comparison* diterapkan
menggunakan metode **Holm-Bonferroni** terhadap ketiga nilai-$p$
tersebut secara bersamaan, untuk mengendalikan kemungkinan terjadinya
*family-wise error rate* (Holm, 1979). Sebagai pelengkap, *effect size*
dilaporkan menggunakan koefisien korelasi *rank-biserial* $r$ untuk
setiap perbandingan.

**Tabel 3.14 Konfigurasi Analisis Statistik**

  -----------------------------------------------------------------------
  Komponen                            Konfigurasi
  ----------------------------------- -----------------------------------
  Jumlah *random seed*                5 (diterapkan identik pada semua
                                      kondisi dan baseline eksternal)

  Ringkasan hasil                     Mean ± Standard Deviation antar 5
                                      seed

  Uji statistik                       Wilcoxon Signed-Rank Test, satu
                                      sisi (*alternative = greater*)

  Perbandingan yang diuji             2TF vs 1TF; 3TF vs 1TF; 4TF vs 1TF
  (*pre-specified*)                   

  Pemasangan data (*pairing*)         Berdasarkan indeks *seed* yang sama
                                      antar kondisi (*paired by seed
                                      index*)

  Koreksi *multiple comparison*       Holm-Bonferroni, diterapkan
                                      terhadap ketiga nilai-$p$ secara
                                      bersamaan

  Tingkat signifikansi                $\alpha = 0,05$ (sebelum dan
                                      sesudah koreksi)

  Effect size                         Rank-biserial correlation $r$,
                                      dilaporkan untuk setiap
                                      perbandingan

  Pengujian validitas ekonomi         Kruskal-Wallis test pada distribusi
  (pendukung)                         *return* per *cluster*, $p < 0,05$
  -----------------------------------------------------------------------

Hasil analisis statistik digunakan untuk mendukung interpretasi terhadap
pengaruh jumlah resolusi temporal terhadap kualitas representasi laten.
Penelitian tidak bertujuan menentukan konfigurasi terbaik secara
universal melalui seleksi *post-hoc*, melainkan menguji secara
independen apakah masing-masing penambahan resolusi temporal (2TF, 3TF,
4TF) menghasilkan perbedaan yang signifikan dibandingkan *baseline* 1TF
pada konfigurasi eksperimen yang telah ditetapkan sebelumnya. Hipotesis
eksploratif H2 (hubungan non-monotonik, lihat Subbab 2.11) dilaporkan
secara deskriptif sebagai temuan tambahan dan tidak memengaruhi
kesimpulan terhadap H1a--H1c.

# EXPECTED CONTRIBUTIONS

Penelitian ini diharapkan memberikan kontribusi pada tiga tingkat:
konseptual, metodologis, dan empiris/praktis.

**Konseptual.** Memposisikan resolusi temporal sebagai variabel
penelitian yang dievaluasi secara sistematis dalam pembelajaran
representasi *self-supervised* untuk *market state discovery* pada
domain cryptocurrency --- bukan sekadar karakteristik dataset seperti
pada mayoritas studi terdahulu (lihat Subbab 1.3, 2.9).

**Metodologis.** Menyajikan protokol eksperimen terkontrol dengan skema
pengujian statistik *pre-registered* (tiga perbandingan berpasangan,
koreksi Holm-Bonferroni) serta *baseline* yang sepenuhnya reproduktif
--- lima *random seed* terdokumentasi, diterapkan identik ke seluruh
sumber keacakan (Subbab 3.8, 3.9).

**Empiris dan Praktis.** Memberikan bukti empiris mengenai kontribusi
resolusi temporal terhadap kualitas representasi pasar cryptocurrency
melalui tiga perbandingan pre-specified (2TF, 3TF, 4TF vs. baseline
1TF), sekaligus menyediakan representasi laten yang berpotensi menjadi
landasan bagi *downstream task* di luar cakupan penelitian ini
(*forecasting*, klasifikasi, deteksi anomali), sebagaimana dibatasi
eksplisit pada Subbab 1.7.

*(Uraian lengkap dengan konteks argumentatif penuh tersedia di Subbab
1.6.)*

# RESEARCH SCHEDULE

**Tabel 3.15 Jadwal Pelaksanaan Penelitian (6 Bulan)**

  -----------------------------------------------------------------------------
  No       Tahapan        B1       B2       B3       B4       B5       B6
           Penelitian                                                  
  -------- -------------- -------- -------- -------- -------- -------- --------
  1        Identifikasi   ✓                                            
           masalah                                                     

  2        Studi          ✓        ✓                                   
           literatur                                                   

  3        Penyusunan     ✓        ✓                                   
           proposal                                                    

  4        Akuisisi &              ✓        ✓                          
           persiapan                                                   
           dataset                                                     

  5        Implementasi                     ✓        ✓                 
           model                                                       

  6        Pelaksanaan                               ✓        ✓        
           eksperimen                                                  

  7        Evaluasi &                                         ✓        
           analisis                                                    
           statistik                                                   

  8        Penyusunan                                         ✓        ✓
           laporan akhir                                               
  -----------------------------------------------------------------------------

*Keterangan: B1--B6 = Bulan ke-1 hingga Bulan ke-6. Tanda ✓ menunjukkan
tahapan aktif dilaksanakan pada bulan terkait.*

Dengan demikian, seluruh rangkaian metodologi penelitian dirancang untuk
mengevaluasi pengaruh resolusi temporal terhadap kualitas representasi
laten secara objektif, mulai dari proses akuisisi data, pembelajaran
representasi, pembentukan *market state*, hingga analisis statistik
terhadap hasil eksperimen. Setiap komponen dalam rancangan ini telah
diatur sedemikian rupa sehingga confound akibat perbedaan konfigurasi
model, dimensi representasi, metode *clustering*, maupun prosedur
evaluasi dapat dikendalikan secara eksplisit, dan strategi pengujian
statistik *pre-registered* (Subbab 2.11 dan 3.8) memastikan bahwa
kesimpulan penelitian tidak terkontaminasi oleh seleksi kondisi secara
*post-hoc*.

## KETERBATASAN PENELITIAN

Bagian ini merangkum keterbatasan yang melekat pada desain penelitian,
sebagaimana telah disinggung pada bagian-bagian sebelumnya, untuk
transparansi penuh kepada pembaca dan reviewer.

1.  **Rank efektif representasi tidak seragam antar kondisi.**
    Sebagaimana dijelaskan pada Subbab 2.6, *fixed random projection*
    tidak dapat menambah informasi yang tidak ada pada representasi
    hasil konkatenasi. Akibatnya, meskipun seluruh kondisi menghasilkan
    keluaran berdimensi nominal 256, rank efektifnya dibatasi oleh
    dimensi konkatenasi asli (64/128/192/256). Bias ini bersifat
    konservatif terhadap H1 --- yaitu memperkecil, bukan memperbesar,
    peluang ditemukannya keunggulan kondisi multi-resolusi --- sehingga
    temuan positif terhadap H1 tidak dapat dijelaskan oleh artefak
    dimensi proyeksi.

2.  **Window size W=48 merepresentasikan rentang waktu nyata yang
    berbeda per timeframe.** Dengan *stride*=1 yang seragam, *window*
    berukuran 48 pada resolusi 15 menit mencakup 12 jam data, sedangkan
    pada resolusi 1 hari mencakup 48 hari data. Hal ini merupakan
    variabel majemuk (*compound variable*) yang disengaja: penelitian
    ini secara eksplisit tidak menyamakan cakupan waktu riil antar
    *timeframe*, karena menyamakan cakupan waktu akan menghilangkan
    justru karakteristik yang ingin dievaluasi.

3.  **Autokorelasi akibat *forward-fill* pada resolusi 4 jam dan 1
    hari.** Proses *forward-fill* yang digunakan untuk menyelaraskan
    resolusi 4 jam dan 1 hari ke *anchor* 1 jam menghasilkan nilai yang
    identik berulang pada beberapa langkah waktu berurutan (4 langkah
    untuk 4h, 24 langkah untuk 1d). Karakteristik ini menghasilkan
    autokorelasi buatan yang melekat pada desain *forward-fill*, namun
    diterapkan secara identik pada seluruh kondisi sehingga tidak
    menjadi sumber bias antar kondisi.

4.  **Tumpang tindih window pertama pada periode pengujian.** Akibat
    desain *sliding window* dengan *stride*=1, hingga 47 *window*
    pertama pada *test set* memiliki cakupan waktu yang sebagian
    bertumpang tindih dengan periode *training*. Hal ini merupakan
    praktik standar pada evaluasi *self-supervised learning* berbasis
    deret waktu dan tidak menimbulkan kebocoran harga masa depan, karena
    tidak ada label atau statistik dari masa depan yang digunakan.

5.  **Parameter HDBSCAN dikunci dari kondisi 1TF.** Parameter
    min_cluster_size dan min_samples dipilih melalui *grid search* hanya
    pada kondisi 1TF, kemudian diterapkan tanpa perubahan pada seluruh
    kondisi lain. Pendekatan ini menjamin validitas eksperimen
    (parameter klaster tidak menjadi variabel perancu), namun berarti
    parameter tersebut belum tentu optimal untuk kondisi multi-resolusi.

6.  **Cakupan literatur pembanding contrastive learning bersifat
    representatif, bukan ekshaustif.** Tinjauan pada Subbab 2.8 berfokus
    pada empat metode *contrastive representation learning* yang paling
    relevan secara historis dan metodologis (CPC, TNC, CoST, TS2Vec)
    beserta pembanding *market state discovery* (HMM) dan penelitian
    *multi-timeframe* finansial. Metode-metode lain seperti TS-TCC,
    BTSF, atau varian *contrastive* terbaru turut disinggung namun tidak
    dibahas secara mendalam.

7.  **Jumlah *random seed* (n=5) membatasi *statistical power* uji
    signifikansi.** Uji Wilcoxon *signed-rank* satu sisi pada n=5
    memiliki *p*-value minimum teoritis sebesar 1/32 (\\approx 0.03125),
    yang mendekati ambang signifikansi \\alpha=0.05 dan menjadi lebih
    ketat lagi setelah koreksi Holm-Bonferroni untuk tiga perbandingan
    (H1a--H1c). Dengan demikian, penelitian ini memiliki keterbatasan
    daya statistik (*statistical power*) untuk mendeteksi efek berukuran
    kecil hingga sedang; kegagalan menolak $H_{0}$ pada suatu
    perbandingan tidak dapat langsung diinterpretasikan sebagai bukti
    tidak adanya pengaruh resolusi temporal, melainkan kemungkinan juga
    mencerminkan keterbatasan daya uji pada ukuran sampel *seed* yang
    digunakan. Jumlah *seed* ditetapkan mempertimbangkan keterbatasan
    sumber daya komputasi untuk pelatihan SSL berulang, sebagaimana umum
    dijumpai pada studi *representation learning* berskala penelitian
    tunggal.

---

## DAFTAR REFERENSI

*(Daftar referensi berikut telah diperluas untuk mendukung Subbab 2.4,
2.5, dan 2.8 dengan sitasi CPC, TNC, CoST, TS-TCC, HMM crypto, dan
multi-timeframe finansial. Tabel 2.1 tetap disarankan diperluas hingga
minimal 10 entri dengan referensi tambahan sesuai arahan pembimbing
sebelum proposal disubmit.)*

> Aydınhan, A. O., Kolm, P. N., Mulvey, J. M., & Shu, Y. (2024).
> Identifying patterns in financial markets: Extending the statistical
> jump model for regime identification. *Annals of Operations
> Research*. https://doi.org/10.1007/s10479-024-06035-z [Belum
> diverifikasi Scopus/DOI resolver secara independen --- bersumber dari
> kompilasi literatur penulis, lihat MASTER_BIBLIOGRAPHY_VALIDASI.md]
>
> Bucci, A., & Ciciretti, V. (2022). Market regime detection via
> realized covariances. *Economic Modelling*, 111, 105832.
> https://doi.org/10.1016/j.econmod.2022.105832
>
> Calinski, T., & Harabasz, J. (1974). A dendrite method for cluster
> analysis. *Communications in Statistics*, 3(1), 1--27.
> https://doi.org/10.1080/03610927408827101
>
> Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). Density-based
> clustering based on hierarchical density estimates. *PAKDD 2013*,
> Lecture Notes in Computer Science, 7819, 160--172.
> https://doi.org/10.1007/978-3-642-37456-2_14
>
> Chen, P., Zhang, Y., Cheng, Y., Shu, Y., Wang, Y., Wen, Q., Yang, B.,
> & Guo, C. (2024). Pathformer: Multi-scale transformers with adaptive
> pathways for time series forecasting. *arXiv preprint
> arXiv:2402.05956*.
>
> Davies, D. L., & Bouldin, D. W. (1979). A cluster separation measure.
> *IEEE Transactions on Pattern Analysis and Machine Intelligence*,
> 1(2), 224--227. https://doi.org/10.1109/TPAMI.1979.4766909
>
> Eldele, E., Ragab, M., Chen, Z., Wu, M., Kwoh, C. K., Li, X., & Guan,
> C. (2021). Time-series representation learning via temporal and
> contextual contrasting. *Proceedings of the Thirtieth International
> Joint Conference on Artificial Intelligence (IJCAI-21)*, 2352--2359.
> https://doi.org/10.24963/ijcai.2021/324
>
> Ennagoura, D., Kehal, K. E., Merzouk, S., Abdelhamid, B., Bossoufi,
> B., Fahssi, K. E., Far, M. E., & Bennani, M. T. (2026). Multi-timeframe
> forecasting of Ethereum prices: A comparative study of statistical and
> deep learning models. *ICHORA 2026*.
> https://doi.org/10.1109/ICHORA69329.2026.11537071 [Belum diverifikasi
> Scopus/DOI resolver secara independen --- bersumber dari kompilasi
> literatur penulis, lihat MASTER_BIBLIOGRAPHY_VALIDASI.md]
>
> Giudici, P., & Abu Hashish, I. (2020). A hidden Markov model to detect
> regime changes in cryptoasset markets. *Quality and Reliability
> Engineering International*, 36(6), 2057--2065.
> https://doi.org/10.1002/qre.2673
>
> Hamilton, J. D. (1989). A new approach to the economic analysis of
> nonstationary time series and the business cycle. *Econometrica*,
> 57(2), 357--384.
>
> Holm, S. (1979). A simple sequentially rejective multiple test
> procedure. *Scandinavian Journal of Statistics*, 6(2), 65--70.
>
> Huang, Y., Song, Y., & Chen, Z. (2023). A multi-scaling reinforcement
> learning trading system based on multi-scaling convolutional neural
> networks. *Mathematics*, 11(11), 2467.
> https://doi.org/10.3390/math11112467 [Belum diverifikasi Scopus/DOI
> resolver secara independen --- bersumber dari kompilasi literatur
> penulis, lihat MASTER_BIBLIOGRAPHY_VALIDASI.md]
>
> Johnson, W. B., & Lindenstrauss, J. (1984). Extensions of Lipschitz
> mappings into a Hilbert space. *Contemporary Mathematics*, 26,
> 189--206.
>
> Khurana, S. S., Singh, P., & Garg, N. (2023). Revolutionize AI trading
> bots with AutoML-based multi-timeframe Bitcoin price prediction. *SN
> Computer Science*, 4, 495. https://doi.org/10.1007/s42979-023-01941-8
> [Belum diverifikasi Scopus/DOI resolver secara independen ---
> bersumber dari kompilasi literatur penulis, lihat
> MASTER_BIBLIOGRAPHY_VALIDASI.md]
>
> Koki, C., Leonardos, S., & Piliouras, G. (2022). Exploring the
> predictability of cryptocurrencies via Bayesian hidden Markov models.
> *Research in International Business and Finance*, 59, 101554.
> https://doi.org/10.1016/j.ribaf.2021.101554
>
> Liu, J., & Chen, S. (2024). TimesURL: Self-supervised contrastive
> learning for universal time series representation learning.
> *Proceedings of the AAAI Conference on Artificial Intelligence*,
> 38(12), 13918--13926. https://doi.org/10.1609/aaai.v38i12.29299
>
> McInnes, L., Healy, J., & Astels, S. (2017). hdbscan: Hierarchical
> density based clustering. *Journal of Open Source Software*, 2(11),
> 205. https://doi.org/10.21105/joss.00205
>
> Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A time
> series is worth 64 words: Long-term forecasting with transformers.
> *International Conference on Learning Representations (ICLR 2023)*.
>
> Orton, A., & Gebbie, T. (2024). Representation learning for regime
> detection in block hierarchical financial markets. *arXiv preprint
> arXiv:2410.22346*. [Belum diverifikasi Scopus/DOI resolver secara
> independen --- bersumber dari kompilasi literatur penulis, lihat
> MASTER_BIBLIOGRAPHY_VALIDASI.md]
>
> Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the
> interpretation and validation of cluster analysis. *Journal of
> Computational and Applied Mathematics*, 20, 53--65.
> https://doi.org/10.1016/0377-0427(87)90125-7
>
> Sobreiro, P., Martinho, D., Martins, R., & Vardasca, R. (2026).
> Multi-timeframe feature engineering for Bitcoin market prediction: A
> price-level-agnostic machine learning approach. *Forecasting*, 8(3),
> 40. https://doi.org/10.3390/forecast8030040
>
> Tonekaboni, S., Eytan, D., & Goldenberg, A. (2021). Unsupervised
> representation learning for time series with temporal neighborhood
> coding. *International Conference on Learning Representations (ICLR
> 2021)*.
>
> Valles-Coral, M., Salazar-Ramirez, L., Injante, R., Hernandez-Torres,
> E., Juárez-Díaz, J., Navarro-Cabrera, J., Pinedo, L., & Vidaurre-Rojas,
> P. (2022). Density-based unsupervised learning algorithm to
> categorize college students into dropout risk levels. *Data*, 7(11),
> 165. https://doi.org/10.3390/data7110165 [Belum diverifikasi
> Scopus/DOI resolver secara independen --- bersumber dari kompilasi
> literatur penulis, lihat MASTER_BIBLIOGRAPHY_VALIDASI.md]
>
> Van den Oord, A., Li, Y., & Vinyals, O. (2018). Representation
> learning with contrastive predictive coding. *arXiv preprint
> arXiv:1807.03748*.
>
> Wang, S., Wu, H., Shi, X., Hu, T., Luo, H., Zhang, J. Y., & Zhou, J.
> (2024). TimeMixer: Decomposable multiscale mixing for time series
> forecasting. *arXiv preprint arXiv:2405.14616*.
>
> Woo, G., Liu, C., Sahoo, D., Kumar, A., & Hoi, S. (2022). CoST:
> Contrastive learning of disentangled seasonal-trend representations
> for time series forecasting. *International Conference on Learning
> Representations (ICLR 2022)*.
>
> Yue, Z., Wang, Y., Duan, J., Yang, T., Huang, C., Tong, Y., & Xu, B.
> (2022). TS2Vec: Towards universal representation of time series.
> *Proceedings of the AAAI Conference on Artificial Intelligence*,
> 36(8), 8980--8987. https://doi.org/10.1609/aaai.v36i8.20881
>
> Zhang, W., Yang, L., Geng, S., & Hong, S. (2023). Self-supervised
> time series representation learning via cross reconstruction
> transformer. *IEEE Transactions on Neural Networks and Learning
> Systems*, 35, 16129--16138.
> https://doi.org/10.1109/tnnls.2023.3292066
>
> Zhang, Y., Ma, L., Pal, S., Zhang, Y., & Coates, M. (2024).
> Multi-resolution time-series transformer for long-term forecasting.
> *Proceedings of the 27th International Conference on Artificial
> Intelligence and Statistics (AISTATS)*, 238, 4222--4230.
> arXiv:2311.04147.
