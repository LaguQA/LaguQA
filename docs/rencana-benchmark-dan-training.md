# Rencana tahap berikutnya: dari dataset ke benchmark dan model

Transkripsi 107 lagu sudah selesai. Berkas ini merencanakan tiga hal
sesudahnya: menutup lubang data yang tersisa, menyusun soal benchmark, dan
melatih serta menguji model.

---

## 1. Keadaan dataset sekarang

`dataset_clean.csv`, 107 baris, 13 kolom, tidak ada sel kosong selain
`composer` dan `origin` yang memang saling melengkapi menurut jenis lagu.

| kolom | terisi | catatan |
|---|---|---|
| `title`, `song_type`, `image_filename` | 107 | dari pendataan buku |
| `key_signature` | 107 | dari `Do =` atau `1 =` yang tercetak |
| `tempo` | 107 | dari keterangan tempo yang tercetak |
| `lyrics` | 107 | disusun ulang dari baris `w:`, rata-rata 312 karakter |
| `abc_notation` | 107 | rata-rata 1124 karakter |
| `time_signature` | 107 | **hanya 57 terbaca**, 50 disimpulkan |
| `composer` | 55 | seluruhnya lagu nasional, 3 lagu daerah punya keduanya |
| `origin` | 55 | seluruhnya lagu daerah |
| `abc_status` | 107 | 79 terverifikasi, 28 mentah |

Mutu notasi: 2691 dari 2709 bar lulus konservasi ketukan (99,3 persen), 834
dari 878 baris lirik selaras dengan notnya, nol masalah sintaks.

### Yang membatasi pemakaian

Tiga hal harus dibawa ke bab metodologi, bukan disembunyikan:

**Birama 50 lagu adalah kesimpulan, bukan bacaan.** Buku memakai dua gaya
kepala halaman. Halaman bergaya `Do = C` mencantumkan biramanya; halaman
bergaya `1 = C` hanya mencantumkan tempo. Gaya kedua dipakai hampir di seluruh
lagu nasional. Untuk lagu-lagu itu birama pada `M:` dihitung model dari jumlah
ketukan per bar. Kolom `time_signature_source` mencatat mana yang mana.

Akibatnya jelas: **soal tentang birama tidak boleh dibuat dari 50 lagu itu**,
karena kunci jawabannya dihasilkan model sejenis yang justru sedang diuji.

**28 lagu masih mentah.** Sisa temuannya 18 pelanggaran ketukan dan 44 baris
lirik yang jumlah suku katanya tidak sama dengan jumlah not. Lagu-lagu itu
belum layak jadi kunci jawaban soal melodi, meski metadatanya tetap terpakai.

**Dua pasang judul tidak terbedakan.** Ada dua *Desaku* (id 35 daerah, id 89
nasional) berjudul persis sama. *Tanah Airku 1* dan *2* dibedakan angka saja.
Soal yang menyebut judul saja akan ambigu untuk pasangan pertama.

### Lagu yang layak jadi kunci jawaban, per jenis soal

| jenis soal | lagu layak |
|---|---|
| nada dasar, tempo, lirik | 107 |
| melodi (butuh `abc_status` terverifikasi) | 79 |
| birama (butuh tercetak di buku) | 57 |
| pencipta | 55 |
| asal daerah | 55 |
| melodi **dan** birama sekaligus | 43 |

---

## 2. Menutup lubang data

Dikerjakan lebih dulu karena semua tahap sesudahnya bergantung padanya.

**2.1 Selesaikan 28 berkas mentah.** Prioritas menurut besar masalahnya:

- `021_cing_cangkeling` (6 bar, 3 lirik) — pindaiannya tembus-tinta parah.
  Buku menulis lima lambang per bar, Gemini menulis tiga not. Dari hitungan
  lirik dapat dipastikan lambang ketiga itu not, bukan tanda istirahat. Paling
  murah ditranskripsi ulang dengan gambar yang dipotong per sistem.
- `103_juwita_malam` (9 lirik) — baris `w:` tidak menutupi seluruh not pada
  bagian intro. Perlu ditambah `*` sebanyak not yang tak berlirik.
- `105_sapu_tangan_dari_bandung_selatan` (5 bar) — nada dan liriknya benar;
  buku menaruh garis birama tidak konsisten dan memakai legatura menyeberang
  bar. Cukup dicatat sebagai keterbatasan, tidak perlu diubah.
- `015_jali_jali`, `014_kicir_kicir`, `099_jembatan_merah` (3–5 lirik) dan 22
  berkas sisanya, umumnya meleset satu suku kata di satu baris.

**2.2 Periksa empat berkas berperingatan register.** `021`, `038_naluya`,
`089_desaku`, `103_juwita_malam` bernada tengah E5 ke atas. Bandingkan titik
oktaf pada halamannya. Perlu diingat pemeriksaan ini lemah: `061` sempat
tertandai dan ternyata benar.

**2.3 Bedakan dua *Desaku*.** Tambahkan keterangan pembeda pada kolom `title`
atau kolom baru, supaya soal tidak ambigu.

**2.4 Bekukan versi dataset.** Setelah 2.1–2.3 selesai, beri tanda versi
(`dataset_v1.csv` beserta ringkasan statistiknya). Seluruh angka di skripsi
mengacu ke versi itu, sehingga hasil evaluasi bisa diulang orang lain.

---

## 3. Menyusun benchmark LaguQA

### 3.1 Pembagian data lebih dulu, sebelum soal dibuat

Bagi **per lagu**, bukan per soal. Kalau lagu yang sama muncul di latih dan
uji, model bisa menjawab soal uji dari hafalan lagu itu, bukan dari
kemampuannya.

Usulan: 70 lagu latih, 37 lagu uji, diseimbangkan menurut `song_type` dan
`abc_status` supaya kedua sisi punya campuran yang sama. Simpan daftar
pembagiannya sebagai berkas tersendiri, jangan diacak ulang tiap kali.

### 3.2 Jenis soal

Disusun bertingkat, dari mengingat sampai menalar. Tingkat ketiga yang membuat
ini benchmark pemahaman, bukan kuis hafalan.

**Tingkat 1 — pengetahuan faktual.** Pencipta, asal daerah, nada dasar,
tempo, birama. Mudah dibuat dan mudah dinilai, tetapi paling rawan dijawab
dari ingatan model tentang lagu terkenal, bukan dari pemahaman.

**Tingkat 2 — lirik.** Melanjutkan potongan lirik, menyebut judul dari
potongan lirik, mengurutkan bait. Lirik daerah (Aceh, Batak, Sunda, Maluku)
menarik karena kecil kemungkinannya ada di data latih model umum.

**Tingkat 3 — penalaran notasi.** Ini yang membedakan LaguQA dari kuis biasa,
dan hanya bisa dibuat dari 79 lagu terverifikasi:

- Berapa bar lagu ini? Nada apa yang tertinggi? Berapa ambitusnya?
- Transposisikan delapan bar pertama ke nada dasar lain.
- Diberikan potongan ABC, lagu apa ini?
- Apakah dua potongan ini melodi yang sama pada nada dasar berbeda?
- Suku kata mana yang jatuh pada nada tertinggi?

Jawaban tingkat 3 bisa **dihitung dari berkas ABC**, bukan ditulis tangan.
Artinya kunci jawabannya bisa diverifikasi ulang secara otomatis, dan
jumlahnya bisa diperbanyak tanpa biaya tambahan. Ini kekuatan utama pendekatan
ABC dan layak ditonjolkan di skripsi.

**Tingkat 4 — pemahaman budaya.** Lagu ini dinyanyikan pada kesempatan apa?
Termasuk daerah mana dan apa cirinya? Perlu kehati-hatian: jawabannya sering
tidak tunggal, jadi nilai baiknya dengan rubrik, bukan pencocokan tepat.

### 3.3 Perkiraan jumlah

Dengan satu soal per jenis per lagu yang layak, tanpa memaksakan angka bulat:

| tingkat | perkiraan |
|---|---|
| 1 faktual | 381 (107 nada dasar + 107 tempo + 57 birama + 55 pencipta + 55 asal) |
| 2 lirik | ~214 (dua bentuk soal × 107) |
| 3 notasi | ~395 (lima bentuk soal × 79) |
| 4 budaya | ~107 |
| **jumlah** | **~1100** |

Lebih dari cukup untuk skripsi. Kalau waktunya sempit, tingkat 3 yang
dipertahankan karena paling orisinal, dan tingkat 1 dikurangi.

### 3.4 Sudah dibangun

Tiga alat sudah jadi dan hasilnya sudah ditulis:

```
tools/laguqa_split.py     70 latih / 37 uji, berlapis, dibekukan ke laguqa_split.json
tools/abc_to_jianpu.py    ABC -> notasi angka, 107 berkas tanpa kegagalan
tools/laguqa_gen.py       laguqa_train.jsonl (10.000) + laguqa_test.jsonl (677)
```

Komposisi latih 76 persen hafalan, 24 persen penalaran. Sasaran 90:10 tidak
tercapai karena stok hafalan yang benar-benar berbeda mentok di 7578 contoh
dari 70 lagu; sisanya sampai 10.000 terisi soal penalaran. Kalau rasio lebih
penting daripada jumlah, turunkan `--target` ke sekitar 8400.

Soal penalaran sengaja disisakan sedikit, bukan dihapus. Kategori itu
satu-satunya yang jawabannya berada di dalam pertanyaan, sehingga hanya di
situ model bisa menjawab benar untuk 37 lagu yang tidak pernah dilihatnya.
Tanpa itu kolom "belum pernah dilihat" pasti nol dan percobaan tidak bisa
menunjukkan generalisasi sama sekali.

Soal hitung not dan ambitus dibuang: keduanya aritmetika notasi yang tidak ada
Indonesianya, dan benchmark ini tentang lagu, bukan tentang berhitung.

Pemeriksaan yang sudah lulus: tidak ada lagu di kedua sisi, seluruh kunci sama
persis dengan `dataset_clean.csv`, tidak ada soal birama dari 50 lagu yang
biramanya disimpulkan, tidak ada soal notasi dari ABC mentah, tidak ada
pertanyaan kembar maupun pertanyaan yang sama di kedua sisi, dan 68 contoh
abstain berjawab jujur.

Ukurannya sekitar 730 ribu token per epoch, jadi lima epoch ~3,6 juta token,
kira-kira dua dolar sekali latih pada A100 40GB.

### 3.5 Bentuk berkas

Simpan sebagai JSONL, satu soal satu baris, memuat `id_lagu`, `tingkat`,
`jenis`, `pertanyaan`, `jawaban`, `pembagian` (latih/uji), dan `sumber_kunci`
(`tercetak`, `dihitung`, atau `ditulis_tangan`). Kolom terakhir itu yang
membuat batasan pada bagian 1 terbawa sampai ke hasil evaluasi.

---

## 4. Evaluasi awal sebelum melatih apa pun

Jangan langsung melatih. Ukur dulu model yang ada, karena angka inilah
pembanding yang membuat hasil pelatihan berarti.

Model yang diuji sebaiknya mencakup tiga kelompok: model tertutup berbahasa
umum (Gemini, GPT, Claude), model terbuka berukuran sedang (Qwen, Llama), dan
model yang dilatih khusus untuk bahasa Asia Tenggara (SEA-LION, Sailor,
Merak). Kelompok ketiga penting supaya skripsi bisa menjawab apakah pelatihan
khusus bahasa Indonesia benar-benar membantu.

Yang diukur: ketepatan per tingkat soal, bukan satu angka gabungan. Dugaan
yang layak diuji adalah model kuat di tingkat 1 untuk lagu nasional terkenal,
lalu jatuh tajam di tingkat 3, karena penalaran notasi tidak bisa dijawab dari
hafalan.

Untuk tingkat 1 dan 2, tambahkan pemeriksaan kebocoran: tanyakan hal yang sama
tanpa memberikan notasi maupun lirik. Kalau model tetap benar, berarti ia
menjawab dari ingatan, dan soal itu tidak mengukur pemahaman.

### 4.1 Lantai skor: berapa nilai orang yang tidak tahu apa-apa

Dijalankan `scripts/17_controls.py`, yang mengarang tiga berkas prediksi palsu
lalu menilainya dengan penilai yang sama persis seperti model. Tanpa angka ini,
akurasi 40 persen tidak bisa dibedakan antara "belajar banyak" dan "menebak
dengan baik".

Angka di bawah diukur ulang pada benchmark v1.3, dengan anggaran token 1024
supaya jawaban model dasar tidak terpotong (lihat 4.3). Kedua regime punya
lantainya sendiri dan **tidak boleh dipertukarkan**: soalnya berbeda, jadi
lantainya berbeda.

| kontrol | isi jawaban | split37 tepat | split37 toleran | uji penuh tepat | uji penuh toleran |
|---|---|---|---|---|---|
| konstan | jawaban tersering di tiap kategori | 15,4% | **26,9%** | 17,0% | **33,6%** |
| acak | jawaban acak dari kategori yang sama | 11,5% | 24,3% | 11,7% | 28,6% |
| kosong | tidak menjawab apa pun | 0,0% | 0,0% | 0,0% | 0,0% |
| **gemma4-e2b tanpa dilatih** | model sungguhan, belum dilatih | **0,0%** | **19,6%** | **0,0%** | (diukur) |

Kontrol konstan itu penebak sekuat mungkin: jawaban tersering dihitung dari
kunci sisi uji, jadi ia sudah "melihat" jawabannya. Itu disengaja — lantai yang
dihitung dari sisi latih akan terlalu rendah dan membuat setiap hasil tampak
lebih baik daripada sebenarnya.

Kontrol kosong turun dari 0,1% menjadi **0,0% di kedua regime** setelah soal
tempo berkunci `-` diperbaiki. Nol itu bukan hiasan: selama ia di atas nol,
ada aturan penilaian yang memberi nilai kepada jawaban hampa.

Temuan yang harus masuk ke pembahasan: **model yang belum dilatih (19,6%) ada di
bawah penebak konstan (26,9%)**. Menjawab "4/4" untuk semua soal birama dan
"Do = C" untuk semua soal nada dasar mengalahkan Gemma-4-E2B yang belum dilatih,
karena buku ini didominasi 4/4 (63%) dan Do = C (70%) sementara modelnya
memang tidak mengenal lagu-lagu ini. Perbandingan yang bermakna karena itu
bukan lawan nol, melainkan lawan 26,9% di split37 dan **33,6% di uji penuh**.

Ketepatan 0,0% pada model dasar bukan galat: model instruksi menjawab dengan
kalimat penuh, tidak pernah persis sama dengan kunci. Penilaian toleran ada
justru untuk itu.

### 4.2 Empat cacat yang ditemukan kontrol, sebelum ada angka yang terlanjur dipakai

Kontrol ini dijalankan sebelum hasil pelatihan mana pun dibaca, dan keempatnya
akan menghasilkan angka yang salah tanpa memunculkan galat apa pun.

**Berkas kunci tertukar.** `evaluate.py` selalu membaca `laguqa_test.jsonl`,
padahal prediksi regime `split70` berasal dari `laguqa_test_split37.jsonl`.
Keduanya memakai kunci `(id_lagu, kategori)` yang sama tetapi pertanyaannya
berbeda, sehingga jawaban dipasangkan dengan soal yang tidak pernah ditanyakan.
Sekarang berkas kuncinya ditebak dari nama berkas prediksi, dan ada uji yang
memastikan kedua berkas itu memang tidak bisa saling menggantikan.

**Soal tempo dengan kunci `-`.** Dua lagu (id 24 dan 87) tidak mencantumkan
tempo di halamannya, dan kolomnya berisi `-`. Soal tetap dibangkitkan, kuncinya
menjadi karakter `-`, yang setelah dinormalkan menjadi string kosong — sehingga
**model yang diam dinilai benar**. Kontrol "kosong" yang menemukannya. Sekarang
keduanya menjadi `tempo_abstain`, mengikuti pola `pencipta_abstain` yang sudah
ada.

**Penilai abstain tidak memeriksa separuh aturannya sendiri.** Komentarnya
berbunyi "dan tidak menyebut nama siapa pun", tetapi kodenya hanya mencari
frasa penolakan. Jawaban "Buku tidak mencantumkan penciptanya, tetapi
kemungkinan besar ciptaan Ismail Marzuki" mendapat nilai penuh — padahal
menolak-lalu-mengarang persis perilaku yang hendak dijaring soal-soal itu.

**Anggaran token memotong model dasar, tidak memotong model terlatih.** Pada
batas 96 token, 56 persen jawaban model dasar masih di tengah kalimat, sementara
model terlatih menjawab tiga kata dan tidak pernah kena batas. Batas yang
memotong satu pihak saja bukan mengukur pengetahuan, melainkan mengukur
kebanyakan bicara, dan itu akan melebarkan setiap selisih di skripsi. Batasnya
dinaikkan ke 256 dan tingkat pemotongannya sekarang dilaporkan tiap kali.

Cacat pertama dan keempat sama-sama tidak akan pernah memunculkan pesan galat.
Keduanya hanya menghasilkan angka.

### 4.3 Cacat keempat belum selesai: 256 pun masih memotong

Menaikkan batas ke 256 memperbaiki sisi yang diperiksa, bukan cacatnya. Sisi
yang diperiksa waktu itu model terlatih, dan model terlatih memang tidak pernah
kena batas berapa pun. Pada himpunan uji penuh (1002 soal) model dasar masih
kehilangan **22,1 persen** jawabannya di batas 256.

Sebarannya menunjukkan pemotongan itu tidak merata, dan justru itu masalahnya:

| kategori | jawaban dasar > 700 karakter | toleran pada batas 256 |
|---|---|---|
| `tempo` | 100% | 0,0% |
| `verifikasi_nada_dasar` | 100% | 52,0% |
| `notasi_ke_judul` | 84% | 0,0% |
| `judul_ke_baris` | 0% | 0,0% |

Kategori yang paling banyak dipotong bukan kategori acak. Model dasar menulis
paling panjang justru ketika ia tidak tahu jawabannya dan mulai menimbang
kemungkinan, sehingga batas token menghukum ketidaktahuan dua kali: sekali
karena memang tidak tahu, sekali lagi karena kalimatnya dipenggal sebelum
sempat menyebut apa pun.

Karena penilaian toleran bekerja dengan pencarian kata di dalam jawaban,
menambah token **hanya bisa menaikkan** skor model dasar, tidak pernah
menurunkannya. Artinya angka 19,1 persen pada batas 256 adalah **batas bawah**,
bukan hasil, dan setiap selisih yang dihitung terhadapnya adalah selisih
maksimum yang mungkin — bukan selisih yang sebenarnya.

Bawaannya sekarang 1024 token. Yang berubah untuk model terlatih: tidak ada.
Model terlatih berhenti di EOS jauh sebelum batas, jadi ruang tambahan itu
tidak dipakai dan tidak dibayar. Ini bukan kelonggaran yang diberikan merata,
melainkan kelonggaran yang hanya terpakai oleh pihak yang dirugikan batas.

Setelah diukur ulang, pemotongan turun dari 22,1 persen menjadi **0,1 persen**
(1 dari 1002). Dan hasilnya justru pelajaran kedua:

| | batas 256 | batas 1024 |
|---|---|---|
| **JUMLAH** | 19,1% | **19,8%** |
| `hitung_bar` | 66,0% | **82,0%** |
| `tempo` | 0,0% | 0,0% |
| `verifikasi_nada_dasar` | 52,0% | 52,0% |

Totalnya nyaris tidak bergerak — 0,7 poin — sehingga siapa pun yang hanya
melihat angka utama akan menyimpulkan batas token tidak berpengaruh dan
menutup perkaranya. Padahal satu kategori bergeser **16 poin**. Perbaikannya
terkonsentrasi di kategori tempat model dasar memang menghitung dengan benar
tetapi kehabisan token sebelum menyebut hasilnya; di kategori yang ia memang
tidak tahu, ruang tambahan tidak menolong apa-apa, dan itu memang seharusnya.

Angka utama tidak sanggup membedakan "tidak berpengaruh" dari "berpengaruh
besar di satu tempat dan nol di tempat lain". Itu alasan setiap hasil di
skripsi ini dilaporkan terurai per kategori, bukan sebagai satu bilangan.

**Pelajarannya, dan ini yang layak masuk skripsi:** memperbaiki cacat
pengukuran lalu memeriksanya pada pihak yang tidak terdampak sama saja dengan
tidak memeriksa. Verifikasi harus dijalankan pada kasus yang paling mungkin
gagal, bukan pada kasus yang paling mudah dijangkau.

### 4.4 Cacat kelima: urutan baris prediksi

Penilai memasangkan prediksi dengan kunci menurut **urutan berkas**, karena
berkas prediksi tidak menyimpan id soal — hanya `id_lagu` dan `kategori`. Di
himpunan uji penuh ada **82 slot** yang berisi lebih dari satu soal (di
split37 ada 181). Berkas prediksi yang urutannya berubah karena itu tetap
lengkap, tetap sah, setiap idnya tetap benar, dan dinilai dengan kunci milik
soal saudaranya — tanpa satu pun galat.

Ini ditemukan bukan karena terjadi, melainkan karena hampir dibuat: sempat
ditulis pengelompokan batch menurut kategori untuk menghemat biaya, sebelum
ketahuan bahwa `generate.py` sudah menulis berkas uji dalam blok kategori yang
berurutan, sehingga pengelompokan itu tidak menghemat apa pun dan hanya
menambah satu cara baru untuk salah. Pengelompokannya dibatalkan, dan
`check_alignment` ditambahkan supaya kalau suatu saat ada yang mencobanya lagi,
angkanya berteriak alih-alih diam.

Batasnya disebutkan terus terang di kodenya: pemeriksaan ini tidak bisa melihat
dua soal yang tertukar **di dalam satu slot yang sama**, karena bagi berkas
kunci urutan `(lagu, kategori)`-nya tetap identik. Menutup celah itu menuntut
id soal masuk ke skema prediksi, sedangkan skema itu sudah dibekukan dan
di-hash.

### 4.5 Cacat keenam, dan yang paling menentukan: komposisi data latih

Temuan ini datang dari pertanyaan pembimbing skripsi, bukan dari kode: kenapa
`asal` dan `pencipta` masih di bawah 10 persen, padahal seharusnya bisa di atas
80 persen. Jawabannya tidak ada di hyperparameter. Ada di berapa kali fakta itu
pernah ditunjukkan.

| kategori | contoh latih (v1.2) | per lagu | soal di ujian |
|---|---|---|---|
| `notasi_ke_judul` | 4.123 | 52 | 50 |
| `hitung_bar` | 2.247 | 28 | 50 |
| `judul_ke_lirik` | 107 | 1 | 50 |
| `asal` | **55** | **1** | 50 |
| `pencipta` | **55** | **1** | 50 |

Ujian memberi bobot sama rata, 50 soal tiap kategori. Data latih memberi bobot
**75 banding 1**. Model dilatih pada satu sebaran lalu dinilai pada sebaran
lain, dan kategori yang paling ingin dijawab justru yang paling jarang
ditunjukkan: tiga kali sepanjang tiga epoch, selalu dengan kalimat yang sama,
lalu diuji dengan kalimat berbeda.

**Kenapa timpangnya bukan kelalaian.** Isinya yang menentukan. Satu melodi bisa
diiris menjadi lima puluh potongan; satu pencipta tidak bisa diiris menjadi apa
pun. Anggaran "90 persen hafalan" terpenuhi di atas kertas, tetapi jatah
hafalan itu habis dimakan kategori yang paling mudah diperbanyak. Pengambilan
acak dari kolam yang timpang menghasilkan hasil yang timpang — dan itu terlihat
seimbang karena rasionya benar.

Yang membuatnya luput selama ini: `pick_test()` sudah menyeimbangkan rata per
kategori sejak awal, dan letaknya cuma dua puluh baris di bawah
`sample_to_target()` yang tidak. Dua fungsi bertetangga, memakai kata
"kategori" dengan arti berbeda, tanpa satu pun galat.

**Dua perbaikan.**

*Parafrase untuk fakta yang langka.* Satu fakta ditanyakan delapan cara. Yang
tetap sama di antara kedelapannya cuma jawabannya, jadi yang bisa dipelajari
cuma faktanya — sementara mengulang satu kalimat delapan kali hanya mengajarkan
kalimat itu. Susunan kalimat sisi uji sengaja tidak pernah dipakai di sisi
latih, dan pemeriksaannya dijalankan atas templatnya, bukan atas hasil
sampelnya, supaya tabrakan yang kebetulan tidak terambil tetap ketahuan.

*Anggaran per kategori, bukan per tingkat.* `sample_to_target()` sekarang
memakai aturan yang sama dengan `pick_test()`: jatah dibagi rata ke semua
kategori, secara water-filling — kategori termiskin dilayani lebih dulu dengan
seluruh isinya, sisanya dibagikan ke yang masih punya. Rasio hafalan/penalaran
tidak lagi ditetapkan; angkanya jatuh sendiri di 86/14 karena 19 dari 21
kategori memang hafalan, dan dicetak tiap kali supaya pergeserannya terlihat.

Hasilnya, dengan jumlah total yang sama persis, 15.000 contoh:

| kategori | v1.2 | v1.3 | |
|---|---|---|---|
| `pencipta` | 55 | **440** | 8,0× |
| `asal` | 55 | **440** | 8,0× |
| `judul_ke_lirik` | 107 | **856** | 8,0× |
| `verifikasi_pencipta` | 110 | **495** | 4,5× |
| `hitung_bar` | 2.247 | 1.054 | 0,5× |
| `notasi_ke_judul` | 4.123 | 1.054 | 0,3× |

Ditambah satu perubahan yang bukan sekadar volume: tiap fakta kini disodori
**empat klaim keliru** untuk ditolak, bukan satu. Model yang sekali diberi
"bukan Ismail Marzuki" bisa menghafal satu pasangan itu; diberi empat pencipta
keliru untuk lagu yang sama, satu-satunya yang tersisa untuk dipelajari adalah
yang benar. Perbanyakan ini **hanya di sisi latih** — di sisi uji perbandingan
benar/salah dijaga tetap 53:47, sebab ujian dengan empat klaim salah per satu
klaim benar akan memberi nilai tinggi kepada model yang menolak segalanya.

**Berkas ujinya tidak berubah satu byte pun.** `laguqa_test.jsonl` tetap
`e4553426…` dan `laguqa_test_split37.jsonl` tetap `2dffda6d…`. Itu bukan
kebetulan melainkan syarat: karena soalnya identik, hasil sebelum dan sesudah
penyeimbangan bisa dibandingkan langsung sebagai ablasi satu variabel — yang
berubah hanya komposisi data latih.

### 4.6 Cacat ketujuh: ruang jawaban `hitung_bar` terlalu sempit

Ditemukan 2 September 2026 sewaktu membangun kontrol untuk jalur pilihan
ganda, dan berlaku untuk **kedua** jalur.

Potongan notasi selalu dipotong sepanjang `EXCERPT_BARS = (2, 4, 8)`. Akibatnya
jawaban `hitung_bar` tidak pernah bernilai lain:

| berkas | 2 | 4 | 8 | nilai lain |
|---|---|---|---|---|
| `laguqa_test.jsonl` (50 soal) | 46,0% | 26,0% | 28,0% | tidak ada |
| `laguqa_train.jsonl` (1054 soal) | 34,2% | 37,0% | 28,8% | tidak ada |

Pada jalur pilihan ganda ini fatal dan sudah diperbaiki: pengecohnya dulu
`n-1, n+1, n+2, n*2`, yaitu angka seperti 7, 9, 10, dan 16 yang tidak pernah
menjadi jawaban di mana pun, sehingga cukup memilih angka bulat yang tersedia.
`kontrol-prior` menjawab benar **152 dari 152 soal tanpa membaca satu not pun**.
Setelah panjang potongan disebar 3–12 bar lewat `COUNT_BARS`, angka itu turun
ke 31,9%.

Pada jalur teks bebas cacatnya lebih halus tetapi masih ada, dan **belum
diperbaiki**. Karena hanya ada tiga jawaban yang mungkin, dan ketiganya muncul
di data latih, angka `hitung_bar` yang tinggi tidak berarti model bisa
menghitung bar. Yang terukur adalah kemampuan membedakan 2, 4, dan 8 — tanpa
pernah dituntut menggeneralisasi ke 5 atau 7. Karena itu klaim "Penalaran 100%"
pada arm terlatih harus dinyatakan apa adanya: **membedakan tiga panjang yang
sudah pernah dilihat**, bukan menghitung bar secara umum.

Perbaikannya sama seperti di pilihan ganda, yaitu memberi soal hitung bar
rentang panjangnya sendiri. Yang menahannya bukan kesulitan teknis melainkan
biaya: `EXCERPT_BARS` dipakai di empat tempat pada `generate.py`, sehingga
mengubahnya menghasilkan benchmark v1.4, membatalkan `e4553426…`, dan menuntut
seluruh pelatihan serta prediksi diulang. Ablasi komposisi data yang sedang
berjalan tidak terganggu cacat ini karena kedua armnya terkena sama persis,
jadi urutan yang masuk akal adalah menyelesaikan ablasi dulu di v1.3, lalu
membangun v1.4 untuk angka akhir yang masuk skripsi.

---

## 5. Pelatihan di modal.com

### 5.1 Model dasar yang dipilih

Tiga model, dipilih supaya perbandingannya menjawab pertanyaan, bukan
mengurutkan merek. Seluruh angka di tabel ini hasil pengukuran `probe()` pada
GPU L4, bukan kutipan dari kartu model.

| nama | model_id | parameter | lapis teks | modul LoRA | parameter dilatih | lisensi |
|---|---|---|---|---|---|---|
| gemma4-e2b | `google/gemma-4-E2B-it` | 5.104.297.504 | 15 + 20 | 205 | 24.158.208 (0,47%) | Apache-2.0 |
| sealion-e2b | `aisingapore/Gemma-SEA-LION-v4.5-E2B-IT` | 5.104.297.504 | 15 + 20 | 205 | 24.158.208 (0,47%) | MIT |
| gemma4-e4b | `google/gemma-4-E4B-it` | 7.941.100.832 | 24 + 18 | 258 | 34.881.536 (0,44%) | Apache-2.0 |

Baris pertama dan kedua sama persis pada setiap angka struktural. SEA-LION
v4.5 E2B memang mencantumkan `google/gemma-4-E2B-it` sebagai `base_model`
miliknya; yang membedakan hanya pralatih lanjutan bahasa Asia Tenggara, dan
bahasa Indonesia termasuk di dalamnya. Jadi:

- **gemma4-e2b lawan sealion-e2b** menyendirikan satu peubah, yaitu bahasa.
  Selisih hasilnya tidak bisa dijelaskan oleh ukuran maupun arsitektur.
- **gemma4-e2b lawan gemma4-e4b** menyendirikan peubah yang lain, yaitu
  ukuran.

Ketiganya berlisensi terbuka dan tidak digerbangi, sehingga penguji dapat
mengunduhnya tanpa token dan tanpa menyetujui perjanjian apa pun.

**Pembanding tanpa pelatihan.** `ornith-ai/Ornith-1.5-9B` dan
`aisingapore/Qwen-SEA-LION-v4-4B-VL` diukur zero-shot saja. Angkanya
menjawab seberapa banyak model kuat sudah tahu tentang lagu-lagu ini sebelum
dilatih apa pun; tanpa itu kenaikan hasil pelatihan tidak punya pembanding.
Ornith sengaja tidak dijadikan model latih meski nilai benchmark umumnya
tinggi: ia model penalaran yang membuka jawaban dengan blok `<think>`,
sedangkan seluruh jawaban di data latih berupa jawaban telanjang. Melatihnya
dengan 10.000 contoh semacam itu justru menghapus kebiasaan yang membuatnya
kuat, dan mengotori perbandingan karena ketiga model Gemma tidak akan rusak
dengan cara yang sama.

**Cara melatih.** LoRA, rank 16, alpha 32, dropout 0,05, pada seluruh proyeksi
atensi dan MLP di menara teks. Bukan hanya lapisan atensi seperti rencana
awal: proyeksi MLP ikut disertakan karena bagian terbesar data latih adalah
hafalan fakta, dan itu tersimpan di MLP.

**Dua hal yang ditemukan `probe()` dan mengubah rencana.**

Pertama, lapisannya tidak seragam. Pada E2B, 15 lapis pertama membawa q, k, v,
dan o sendiri, sedangkan 20 lapis sisanya hanya membawa q dan o karena memakai
ulang kunci dan nilai dari lapis sebelumnya. Versi pertama penjaga menuntut
tujuh proyeksi per lapis, menghitung 245, lalu menolak jalan. Arsitekturnya
yang benar, asumsinya yang salah.

Kedua, menara penglihatan dan pendengaran ternyata **tidak** memakai nama
`q_proj` dan kawan-kawannya, sehingga pencocokan bawaan PEFT — yang mencari
`.k_proj` lengkap dengan titiknya — sudah meleset dari keduanya. Kekhawatiran
awal tidak terbukti untuk ketiga model ini. Namun kalau titiknya dihilangkan,
seperti pada regex tulisan tangan atau PEFT versi lama, `relative_k_proj` di
menara pendengaran ikut tertangkap: 12 modul, satu per lapis audio, yang tidak
akan pernah menerima gradien tetapi tetap terhitung sebagai parameter terlatih.
`probe()` melaporkan kedua angka supaya jaraknya terukur, bukan diperdebatkan.

**Bentuk data latih.** Percakapan dipecah menjadi `prompt` dan `completion`,
sehingga kerugian hanya dihitung pada giliran asisten. Ini syarat yang
menopang seluruh rancangan: fakta yang ingin dihafal berada di jawaban, jadi
jawaban itulah yang harus dilihat kerugian. Kalau pertanyaannya ikut dinilai,
model menghabiskan kapasitasnya menghafal pertanyaan yang toh akan selalu
diberikan. `completion_only_loss=True` ditulis terang-terangan, bukan
dibiarkan mengikuti nilai bawaan, karena kalau setelan itu berubah diam-diam
tidak ada yang tampak rusak.

### 5.2 Keadaan `modal_train.py` sekarang

Sudah jadi dan sudah diuji di Modal. Enam perintah, empat di antaranya murah
dan dipakai untuk memastikan sesuatu sebelum mengeluarkan biaya besar.

| perintah | GPU | biaya | gunanya |
|---|---|---|---|
| `upload` | — | nol | menaruh empat berkas benchmark di volume `laguqa-data`, sekalian mencatat SHA-256-nya |
| `inspect_api` | — | ~nol | menanyakan setelan apa saja yang diterima `SFTConfig` di image ini |
| `probe` | L40S | ~$0,07 | memuat model, melaporkan isinya, memastikan LoRA menempel di tempat yang benar |
| `smoke` | L40S | ~$0,03 | 30 langkah pelatihan untuk memastikan kerugiannya bergerak |
| `bench` | beberapa | ~$0,15 | mengukur detik per langkah di beberapa kartu, lalu menghitung biaya percobaan penuh |
| `run` | L40S | $2,34 (terukur) | satu percobaan sungguhan |
| `predict` | L40S | — | menjawab soal uji, hasilnya dinilai `11_evaluate.py` di komputer sendiri |
| `fetch` | — | nol | menarik manifest dan prediksi turun untuk dinilai |

**Pemilihan GPU diukur, bukan ditebak.** Tiga kartu menjalankan 30 langkah yang
sama persis pada gemma4-e2b, ketiganya memuncak di 12,5 GB:

| kartu | detik/langkah | percobaan penuh | biaya/percobaan |
|---|---|---|---|
| L4 | 3,82 | 1,99 jam | $1,59 |
| A100 | 2,77 | 1,44 jam | $3,03 |
| **L40S** | **1,38** | **0,72 jam** | **$1,40** |

L40S paling cepat sekaligus paling murah — 2,8 kali lebih cepat dari L4 dengan
tarif 2,4 kali lipat. A100 kalah di kedua kolom. Kesimpulan ini tidak bisa
dibaca dari daftar harga, dan biaya mengukurnya lima belas sen.

**Kartu tercepat belum tentu boleh dipakai.** Penelitian ini didanai dua akun
Modal, dan keduanya tidak ditawari perangkat keras yang sama. Workspace yang
belum mendaftarkan metode pembayaran ditolak untuk L40S, A100, dan H100 —
penolakannya muncul saat app dibuat, jadi kartunya bukan pilihan melainkan
prasyarat. Diperiksa satu per satu, karena satu berkas yang mendeklarasikan
enam kartu ditolak gara-gara yang termahal dan tidak melaporkan apa pun tentang
lima sisanya:

| kartu | akun berkredit tanpa kartu | memori |
|---|---|---|
| T4 | bisa | 15 GB |
| L4 | bisa | 23 GB |
| A10G | bisa | 23 GB |
| L40S, A100-40GB, H100 | ditolak | — |

Ini menentukan, bukan sekadar merepotkan: pelatihan gemma4-e2b memuncak di
12,5 GB, jadi **muat di L4**, dan pada $0,80 per jam biayanya $2,27 per
percobaan — sedikit lebih murah daripada L40S, hanya 2,8 kali lebih lama. Akun
kedua karena itu bisa menjalankan percobaan sungguhan, bukan sekadar uji coba.
Yang belum pasti gemma4-e4b: 7,9 miliar parameter, 14,9 GB sekadar untuk
dimuat, sehingga puncak pelatihannya mungkin melewati 23 GB. Itu harus diuji,
bukan diperkirakan.

Jenis kartunya diatur lewat `LAGUQA_GPU`, dan manifes mencatat kartu yang
benar-benar diberikan driver — bukan yang diminta — sehingga mengganti
setelannya tidak bisa diam-diam salah melabeli hasil.

**Angka di tabel itu meleset 1,7 kali, dan itu perlu dicatat.** Percobaan
sungguhan yang pertama berjalan **2,29 detik/langkah**, bukan 1,38: 1.875
langkah dalam 71,5 menit seharga **$2,34**, bukan $1,40. Sebabnya `bench`
mengukur potongan 256 contoh, sedangkan urutan pada data penuh lebih panjang,
sehingga tiap langkah memproses lebih banyak token. Tolok ukur yang memakai
sampel kecil memang mengukur sampel kecil itu — bukan pekerjaan sebenarnya.

Perkiraan yang dipakai sekarang, dari percobaan yang benar-benar berjalan:

| pos | jumlah | biaya |
|---|---|---|
| 6 percobaan E2B (2 model × 3 seed) | @ $2,34 | $14,0 |
| 3 percobaan E4B (model lebih besar) | @ ~$3,60 | $10,8 |
| prediksi 9 adapter (jawaban pendek, cepat) | @ ~$0,50 | $4,5 |
| prediksi 3 model dasar (bertele-tele, lambat) | @ ~$1,60 | $4,8 |
| 2 pembanding (Ornith, Qwen-SEA-LION) | @ ~$2,00 | $4,0 |
| **jumlah** | | **~$38 dari $60** |

**Yang sudah terbukti pada `smoke`:** kerugian turun dari 4,69 ke 1,93 dalam 30
langkah, adapter tersimpan, manifest tertulis. Resep LoRA-nya jalan di
arsitektur tiga-menara ini, yang sebelumnya risiko terbuka.

**Dua kegagalan yang tercatat, karena keduanya menghemat uang:**

`check_targets` versi pertama menuntut tujuh proyeksi per lapis dan menolak
jalan. Ternyata arsitekturnya berbagi kunci dan nilai, jadi asumsinya yang
salah. Kalau penjaga itu tidak ada, LoRA tetap menempel dengan benar tetapi
tidak ada yang memberi tahu bahwa lapisannya tidak seragam.

`SFTConfig` di trl 1.12 tidak lagi menerima `warmup_ratio`, hanya
`warmup_steps`. Satu percobaan gagal karenanya, lalu `inspect_api` dibuat
supaya pertanyaan semacam itu dijawab di CPU seharga nol, bukan di GPU.

**Versi pustaka dipatok pada yang benar-benar terpakai:** torch 2.13.0,
transformers 5.16.1, peft 0.20.0, trl 1.12.0, datasets 5.0.1, accelerate
1.14.0. Patokan ini menggambarkan percobaan yang sudah berjalan, bukan versi
yang diharapkan berjalan.

### 5.3 Yang diharapkan — dan kenapa dugaannya terbalik

Dugaan yang ditulis di sini sebelum percobaan pertama berjalan:

> Yang masuk akal diharapkan: kenaikan pada soal faktual dan lirik, karena
> model menghafal isi 70 lagu latih. Yang **tidak** masuk akal diharapkan:
> kenaikan besar pada penalaran notasi. Tujuh ratus contoh tidak cukup untuk
> mengajari model menghitung ketukan. Kalau ternyata naik, curigai kebocoran
> lebih dulu sebelum menyimpulkan.

**Yang terjadi persis kebalikannya**, dan dugaan itu sengaja tidak dihapus,
karena dugaan yang meleset lebih berguna daripada dugaan yang dirapikan
belakangan supaya terlihat benar.

### 5.4 Percobaan pertama: gemma4-e2b, split70, seed 1 — ANGKANYA SUDAH TIDAK BERLAKU

> **Peringatan.** Seluruh angka di bagian ini diukur pada benchmark v1.1 dengan
> batas 256 token dan aturan `nada_dasar` yang lama. Ketiganya sejak itu
> diperbaiki, dan **tidak satu pun angka di sini boleh dikutip berdampingan
> dengan hasil v1.3.** Tiga alasan yang berdiri sendiri-sendiri:
>
> - v1.1 memuat soal yang membocorkan kuncinya sendiri (§4.5 dan SOURCE.md),
>   yang menaikkan sisi model dasar tanpa pengetahuan apa pun
> - batas 256 token memotong 22,1 persen jawaban model dasar (§4.3)
> - `nada_dasar` waktu itu menilai konvensi penulisan `Do = C`, bukan nadanya
>
> Bagian ini tetap ditulis karena **caranya**, bukan hasilnya. Penguraian yang
> membongkar `nada_dasar` 70,3 persen sebagai nol pengetahuan tetap berlaku
> sebagai metode, dan itulah yang dipakai ulang pada hasil v1.3. Berkas
> prediksinya disimpan di `hasil/arsip/`.

Loss 7,82 → 0,167 dalam 1.875 langkah, 71,5 menit, $2,34, puncak memori
12,5 GB. Tidak ada jawaban yang terpotong di batas 256 token.

| | tepat | toleran |
|---|---|---|
| gemma4-e2b sebelum dilatih | 0,0% | 16,2% |
| tebakan konstan (lantai) | 16,2% | 27,5% |
| **gemma4-e2b sesudah dilatih** | **26,9%** | **36,6%** |

Angka 36,6% itu **tidak boleh dilaporkan sendirian**. Diurai per kategori,
kenaikannya seluruhnya berkumpul di satu tempat:

| kategori | dasar | konstan | dilatih | selisih thd konstan |
|---|---|---|---|---|
| `nada_tertinggi` | 12,0% | 26,0% | **98,0%** | **+72,0** |
| `hitung_bar` | 14,0% | 38,0% | **100,0%** | **+62,0** |
| `verifikasi_nada_dasar` | 54,0% | 44,0% | 70,0% | +26,0 |
| `verifikasi_asal` | 28,9% | 50,0% | 65,8% | +15,8 |
| `rumpang` | 0,0% | 2,0% | 14,0% | +12,0 |
| `nada_dasar` | 0,0% | 70,3% | 70,3% | **+0,0** |
| `birama` | 10,5% | 63,2% | 57,9% | −5,3 |
| `tempo` | 0,0% | 18,9% | 2,7% | −16,2 |

**`nada_dasar` 70,3% adalah nol pengetahuan.** Modelnya mengeluarkan `Do = C`
untuk 37 dari 37 soal — satu jawaban unik untuk kunci yang punya lima nilai.
Skornya sama persis dengan penebak konstan karena ia memang penebak konstan.
Hal yang sama pada birama: 18 dari 19 jawaban `4/4`, dan hasilnya justru di
bawah menebak. Tanpa kontrol konstan, dua baris ini akan terbaca sebagai
"model belajar nada dasar dan birama".

**`hitung_bar` dan `nada_tertinggi` sebaliknya nyata.** Sebaran jawabannya
mengikuti sebaran kunci — `2`:19, `4`:17, `8`:14, cocok satu-satu — bukan satu
tebakan yang diulang, dan diukur pada 37 lagu yang tidak pernah dilatih.

Kesimpulan yang ditopang datanya:

> Fine-tuning memindahkan **keterampilan membaca notasi angka** ke lagu yang
> belum pernah dilihat, tetapi tidak memindahkan **fakta tentang lagu asing** —
> dan memang tidak bisa, karena 37 lagu itu ditahan. Yang mengisi tempat
> pengetahuan adalah prior mayoritas.

Regime `split70` secara rancangan tidak sanggup menjawab pertanyaan injeksi
pengetahuan. Itu tugas regime `full`, yang melatih seluruh 107 lagu dan menahan
**bentuk soalnya**, bukan lagunya.

Satu cacat perilaku yang perlu masuk pembahasan: kategori `jenis` hanya punya
dua jawaban sah, tetapi model menghasilkan 13 label berbeda termasuk
"Keroncong". Toleran 54,1%, tepat hanya 10,8%. Model tidak menghormati
himpunan jawaban tertutup.

---

## 6. Urutan pengerjaan yang disarankan

| tahap | isi | keluaran |
|---|---|---|
| A | tutup lubang data (bagian 2) | `dataset_v1.csv` beku |
| B | pembagian latih/uji | `pembagian.json` |
| C | pembuat soal tingkat 3 | skrip + soal terverifikasi otomatis |
| D | soal tingkat 1, 2, 4 | `laguqa.jsonl` lengkap |
| E | evaluasi model yang ada | tabel dasar per tingkat |
| F | pelatihan di modal | adapter + tabel hasil |
| G | analisis dan penulisan | bab hasil |

Tahap C didahulukan dari D karena paling orisinal dan paling bisa
diotomatiskan; kalau waktu habis, benchmark tetap punya isi yang bernilai.

Tahap E bisa dikerjakan bersamaan dengan D untuk soal yang sudah jadi.

### 5.5 Rancangan ujicoba hiperparameter

Disusun 2 September 2026, ketika sisa saldo akun latih $15,15 dan akun kedua
sekitar $28. Satu run penuh 3 epoch di L40S memakan 107 menit dan $3,50, jadi
anggarannya menentukan rancangannya, bukan sebaliknya.

**Perubahan yang membuat sapuan epoch nyaris gratis.** `save_strategy` diubah
dari `"no"` menjadi `"epoch"`. Satu run 3 epoch kini menghasilkan tiga adapter,
sehingga pertanyaan "apakah epoch ketiga sepadan" dijawab dengan tiga kali
generasi (~$0,45), bukan tiga kali latihan ($10,50). Adapter LoRA hanya
berukuran puluhan MB, jadi menyimpan ketiganya praktis tanpa biaya dibanding
jam GPU yang menghasilkannya.

Pertanyaan itu memang layak diajukan. Pada arm "sebelum", eval loss bergerak
1,012 → 0,532 → 0,496; epoch ketiga menelan sepertiga biaya run dan menurunkan
eval loss hanya 0,036, sementara loss latih terus turun ke 0,218. Jaraknya
melebar, yang berarti epoch ketiga lebih banyak menghafal daripada belajar.
Apakah hafalan itu justru menaikkan akurasi adalah pertanyaan empiris — untuk
benchmark yang memang menguji hafalan, jawabannya tidak jelas di muka, dan
hanya bisa diputuskan dengan menilai tiap checkpoint.

**Satu jebakan yang ditutup di muka.** `lora_alpha` kini mengikuti rank
(`alpha = 2r`) kecuali diisi eksplisit. Skala yang diterapkan LoRA adalah
`alpha/r`, sehingga menaikkan rank sambil menahan alpha tetap 32 justru
memperkecil pembaruan bobot. Hasilnya akan terbaca "rank tidak membantu",
padahal yang terukur konstantanya.

**Arm-nya**, satu variabel per arm dari satu dasar (r=16, α=32, lr=2e-4):

| arm | epoch | biaya | yang diuji |
|---|---|---|---|
| dasar | 3 | $3,50 | sekaligus memberi epoch 1, 2, 3 |
| `r=8` | 2 | $2,35 | kapasitas adapter lebih kecil |
| `r=32` | 2 | $2,35 | kapasitas lebih besar |
| `lr=1e-4` | 2 | $2,35 | langkah lebih kecil |
| `lr=4e-4` | 2 | $2,35 | langkah lebih besar |

Total latih $12,90 untuk tujuh titik ukur, ditambah sekitar $4 untuk prediksi
di L4. Arm sapuan dijalankan di akun kedua yang kartunya L4: sekitar 2,5 kali
lebih lambat per langkah, biaya per run hampir sama, tetapi saldonya lebih
besar dan waktu tunggu bukan kendala.

**Dua batasan yang wajib ditulis di laporan.** Arm sapuan memakai 2 epoch
sedangkan dasarnya 3, jadi perbandingan antar-arm hanya sah dibaca pada epoch
yang sama. Dan seluruhnya seed tunggal: pada kategori berisi 50 soal, satu
soal bernilai 2 poin persen, sehingga selisih di bawah sekitar 14 poin tidak
dapat dibedakan dari kebetulan. `scripts/20_compare.py` menandai baris seperti
itu dengan `kecil`, dan tanda itu tidak boleh dihilangkan dari tabel.

**Waktu dan biaya dicatat otomatis.** `table_runs()` menulis `latihan.csv`
berisi `detik_latih`, `detik_total`, `detik_per_langkah`, `menit_latih`,
`memori_gb`, `usd`, beserta `lora_r`, `lora_alpha`, dan `learning_rate` supaya
tiap arm bisa dibedakan, ditutup satu baris `TOTAL` untuk keseluruhan
percobaan. Angka itu tidak boleh dijumlahkan tangan; baris TOTAL-nya yang
dipakai.

**Urutannya.** Sapuan dijalankan setelah benchmark v1.4 terbangun, bukan
sebelumnya. Menjalankannya di v1.3 berarti membuang hasilnya begitu lirik dan
`hitung_bar` diperbaiki, yaitu membayar dua kali untuk jawaban yang sama.

### 5.6 Hasil dasar v1.4 dan sapuan epoch

Dijalankan 2 September 2026. Run dasar `penuh`: gemma4-e2b, seluruh kolam
20.970 contoh latih (1.103 disisihkan untuk validasi), 3 epoch, r=16, α=32,
lr=2e-4, seed 1. Di L40S memakan 3.933 langkah, 91 menit, **$2,99** —
sepertiga lebih murah dari perkiraan $3,50 karena kecepatannya 1,40 detik per
langkah, bukan 2,4 seperti pada run sebelumnya.

**Yang paling ditunggu: `notasi_ke_judul` pulih.** Memakai seluruh kolam
mengembalikannya ke **90,0%**, melewati 82% yang dicapai v1.2 tanpa
pemangkasan dan jauh di atas 6% ketika kolamnya dipangkas ke 15.000. Ini
mengonfirmasi diagnosis di §4.5: yang membunuh kategori itu bukan resepnya,
melainkan tiga perempat isinya yang tidak pernah ditunjukkan.

Teks bebas keseluruhan 75,6% tepat / 76,6% toleran atas 1.002 soal. Pilihan
ganda 57,8% atas 1.200 soal, dengan basis tanpa pelatihan 34,8% dan lantai
tebakan sadar-distribusi 29,8%.

**Sapuan epoch, dari tiga checkpoint satu run.** Dinilai di jalur pilihan
ganda:

| epoch | keseluruhan | `notasi_ke_judul` | eval loss | tak terbaca |
|---|---|---|---|---|
| 1 | 53,0% | 25,0% | 0,4949 | 14 |
| 2 | 57,2% | 32,4% | 0,2626 | 22 |
| 3 | 57,8% | 31,9% | 0,2500 | 33 |

**Epoch ketiga tidak sepadan.** Tambahannya 0,6 poin, yaitu tujuh soal dari
1.200, sementara batas keterbedaan pada seed tunggal jauh di atas itu.
`notasi_ke_judul` bahkan turun sedikit dari epoch 2. Sepertiga biaya run
dibelanjakan untuk selisih yang tidak dapat dibedakan dari kebetulan, persis
seperti yang diduga dari eval loss yang hanya turun 0,013 sementara loss latih
terus menurun. Rekomendasi untuk percobaan lanjutan: 2 epoch.

Satu pola sampingan yang layak dicatat. Jawaban yang tidak terbaca hurufnya
naik berurutan 14 → 22 → 33. Model makin sering menjawab dengan isi, bukan
dengan huruf pilihan. Rinciannya pada epoch 3: `rumpang` 21, `birama` 7,
`notasi_ke_judul` 4, `nada_tertinggi` 1.

Ketiga puluh tiga itu tidak seragam sebabnya, dan membedakannya penting.
Pada `rumpang` model menyebut kata yang memang tidak ada di antara opsinya —
misalnya menjawab `sape` ketika opsinya `kayu`, `patah`, `pisangku`,
`Bengkok`, `ampar`. Itu jawaban salah biasa, bukan soal format. Pada `birama`
sebaliknya: model menjawab `4` sementara opsinya `4/4`, `2/2`, `5/4`, `6/4`,
`2/4`, sehingga cocok ke empat opsi sekaligus dan sengaja tidak dipaksakan ke
salah satunya. Yang paling aneh `notasi_ke_judul`, tempat model menjawab `G`
atau `F` padahal semua opsinya judul lagu — model tampak menjawab pertanyaan
nada dasar, bukan pertanyaan yang diajukan.

Semuanya dihitung salah, dan itu memang benar. Tetapi angka tak-terbaca wajib
dilaporkan berdampingan dengan akurasi, karena hanya sebagian dari 33 itu yang
berarti model tidak tahu jawabannya.

### 5.7 Pemilihan model pembanding

Penguji meminta sedikitnya sepuluh model. Setelah penambahan ini ada 13, dan
seluruhnya diperiksa langsung ke Hub pada 2 September 2026: bobot terbuka,
tidak *gated*, bersafetensors, dan punya templat percakapan sendiri.

**Syarat kebaruan.** Nama yang mudah diingat justru banyak yang usang. Merak
v4, Cendol, Sailor2, SeaLLMs v3, dan Llama 3.2 semuanya rilis 2023–2024.
Membandingkan Gemma 4 dengan model dua tahun lebih tua mengukur kemajuan
umum bidang ini, bukan pengetahuan tentang buku lagu, dan selisihnya akan
terbaca sebagai keunggulan yang tidak pernah diuji. Semuanya dikeluarkan.

**Dua yang dikeluarkan karena alasan lain.** Cendol tidak menyertakan templat
percakapan; menuliskannya sendiri berarti satu model memakai format karangan
peneliti sementara dua belas lainnya memakai format resminya, yaitu bentuk
ketimpangan pengukuran yang sama seperti batas 256 token. Llama 3.2 dan
CohereLabs tiny-aya terkunci di balik persetujuan manual, sehingga penguji
tidak bisa mengulang runnya.

**Satu pengecualian yang disengaja.** SahabatAI v1 rilis 2024–2025, tetapi
tetap dimasukkan karena tidak ada penggantinya: Sahabat-AI v2 hanya terbit
pada ukuran 70B, sedangkan Nusantara dan Komodo berhenti di 2024. Mengeluarkan
SahabatAI berarti menghapus seluruh model khusus Indonesia dari perbandingan.
Bahwa model khusus Indonesia berukuran kecil tidak lagi diperbarui sejak 2025
adalah temuan tersendiri yang layak ditulis, bukan kekurangan yang disembunyikan.

**Ukurannya dibatasi kartu.** Sapuan pembanding berjalan di L4 24 GB, sehingga
model di atas sekitar 10 miliar parameter pada bf16 tidak muat. Itu menutup
`google/gemma-4-12B-it`, `Qwen-SEA-LION-v4-32B-IT`, dan `Llama-Sahabat-AI-v2-70B-IT`.
Batasan ini wajib ditulis di laporan: papan peringkat ini membandingkan model
kelas 2–10 miliar parameter, bukan model terbaik yang ada.

| kunci | model | kelompok | rilis |
|---|---|---|---|
| `qwen35-4b` | Qwen/Qwen3.5-4B | multibahasa | 2026-02 |
| `qwen35-9b` | Qwen/Qwen3.5-9B | multibahasa | 2026-02 |
| `granite42-8b` | ibm-granite/granite-4.2-8b | multibahasa | 2026-08 |
| `lfm25-2b` | LiquidAI/LFM2.5-2.6B | multibahasa | 2026-07 |
| `smollm3-3b` | HuggingFaceTB/SmolLM3-3B | multibahasa | 2025-07 |
| `apertus-sealion-8b` | aisingapore/Apertus-SEA-LION-v4-8B-IT | Asia Tenggara | 2026-02 |
| `sealion-v35-8b` | aisingapore/Llama-SEA-LION-v3.5-8B-R | Asia Tenggara | 2025-04 |
| `sahabatai-9b` | Sahabat-AI/gemma2-9b-cpt-sahabatai-v1-instruct | Indonesia | 2025-05 |

**Jalurnya pilihan ganda saja.** Teks bebas menilai apakah model mengeluarkan
untai jawaban yang persis, dan model yang tidak pernah dilatih pada format ini
dihukum karena gayanya, bukan karena pengetahuannya. Buktinya sudah ada:
qwen-sealion-4b mencetak 0,0% tepat di seluruh kategori padahal 18,5% toleran.
Pilihan ganda menyamakan formatnya — semua model memilih satu huruf dari lima —
sehingga yang tersisa untuk dibandingkan tinggal pengetahuannya.

### 5.8 Parameter inferensi dan pelatihan yang wajib dilaporkan

Angka akurasi tanpa setelan yang menghasilkannya tidak dapat diulang. Seluruh
setelan di bawah kini ikut tertulis pada baris pertama setiap berkas prediksi,
sehingga tidak bisa terpisah dari jawabannya.

**Penyandian (*decoding*).** Serakah (*greedy*), `do_sample=False`. Tidak ada
suhu, `top_p`, `top_k`, maupun *beam search* — ketiganya bernilai kosong dan
`num_beams` = 1. Alasannya penilaian membandingkan jawaban dengan satu kunci,
sehingga pengambilan sampel hanya menambah ragam yang tidak ada kaitannya
dengan pengetahuan model. Dua kali menjalankan run yang sama menghasilkan
berkas yang identik bita per bita, dan itu memang tujuannya.

**Presisi dan kuantisasi.** Seluruh model dimuat pada `bfloat16` tanpa
kuantisasi apa pun. Tidak ada 8-bit, 4-bit, GPTQ, AWQ, maupun NVFP4. Ini
disengaja: varian terkuantisasi dari model yang sama bisa berselisih beberapa
poin, dan membandingkan model bf16 dengan model 4-bit akan mengukur
kuantisasinya, bukan modelnya. Konsekuensinya kartu 24 GB membatasi ukuran
model yang bisa diuji, dan batas itu sudah ditulis di §5.7.

**Setelan berpikir (*thinking*).** Tidak ada `enable_thinking` yang dipaksakan.
Setiap model memakai bawaan templat percakapannya sendiri. Memaksa mati akan
melumpuhkan model penalaran pada tugas yang memang dirancang untuk mereka;
memaksa hidup akan membebani model yang tidak punya mode itu. Bawaan
masing-masing adalah satu-satunya setelan yang adil ketika papan peringkatnya
mencampur keduanya.

Blok `<think>` dibuang sebelum penilaian, di kedua jalur. Kalau tidak, model
yang menimbang "bisa A atau B, tetapi jawabannya C" akan terbaca menyebut tiga
huruf sekaligus lalu dinilai tidak terbaca, sehingga yang terukur kebiasaan
menampilkan penalarannya, bukan pengetahuannya. Jejaknya **tidak dibuang**:
berkas `*.audit.csv` memuat tiga kolom terpisah — `prediksi` berisi jawaban
mentah utuh, `penalaran` berisi isi blok berpikirnya, dan `jawaban` berisi teks
yang benar-benar dibaca penilai.

**Anggaran token.** 1.024 token baru sebagai bawaan, dinaikkan ke 2.048 untuk
model penalaran. Angka ini pernah menjadi cacat pengukuran dua kali, jadi
tingkat pemotongan dilaporkan per run dan wajib dibaca berdampingan dengan
akurasinya. Pemotongan yang tidak nol berarti anggarannya masih ikut terukur.

**Ukuran batch inferensi** 16, hanya memengaruhi kecepatan, bukan hasil, karena
penyandiannya serakah.

**Pelatihan.** LoRA `r`=16, `alpha`=32, `dropout`=0,05, laju belajar 2e-4,
batch 4 dengan akumulasi gradien 4 (batch efektif 16), 3 epoch, pemanasan 117
langkah, seed 1. `alpha` mengikuti `2r` kecuali diisi eksplisit; lihat §5.5
untuk alasannya. Yang dilatih 24.158.208 parameter dari 5.128.455.712, yaitu
0,47 persen.

### 5.9 Audit menyeluruh benchmark, 2 September 2026

Dijalankan `scripts/25_audit_benchmark.py`, empat belas pemeriksaan otomatis
atas 1.200 soal pilihan ganda. Skrip ini dijalankan ulang setiap kali soal
dibangun; cacat benchmark tidak memberi tanda apa pun, skornya tetap terlihat
wajar dan hanya salah, sehingga pemeriksaannya harus otomatis.

**Temuan yang mengubah angka.**

*Judul bocor di dalam liriknya sendiri.* 26 dari 126 soal `lirik_ke_judul`
mengutip baris yang memuat judul lagunya, misalnya "Oh Kopral Jono gadis mana
yang tak kenal" untuk lagu *Kopral Jono*. Model tanpa pelatihan menjawab benar
**26 dari 26** soal semacam itu, seratus persen, sementara pada 100 soal
sisanya hanya 48%. Model terlatih juga 26 dari 26. Seluruh angka
`lirik_ke_judul` yang pernah dilaporkan karena itu terlalu tinggi sekitar
sepuluh poin, untuk setiap model.

Penyebabnya bukan kelalaian merancang melainkan perbaikan yang hanya sampai
separuh: penjaga `gives_away()` sudah ada di `generate.py` sejak v1.1, dan
tidak pernah disalin ke `multichoice.py`. Dua pembangkit soal, satu penjaga.
Setelah penjaga yang sama dipakai di keduanya, temuannya nol.

*Satu orang muncul sebagai dua pilihan.* Kolom pencipta memuat beberapa ejaan
untuk orang yang sama, dan kolom itu juga sumber pengecoh, sehingga satu soal
bisa menawarkan "Ismail Marzuki", "Imail Marzuki", dan "Ismail, MZ" sebagai
tiga pilihan berbeda dengan kunci pada yang ketiga. Empat soal seperti itu, dan
model yang menjawab dengan ejaan baku dinilai salah karena benar. Diperbaiki di
`laguqa.dataset.composers`; ejaan asli buku disimpan di kolom
`composer_printed` sehingga keterlacakannya tidak hilang.

**Temuan yang diperiksa lalu dinyatakan bukan masalah.** Ketiganya dicatat
karena menolak sesuatu tanpa alasan sama tidak jujurnya dengan menerimanya.

*Kata jawaban terulang di kutipan `rumpang`*, 21 soal. Diukur, bukan diduga:
model tanpa pelatihan mendapat 40,0% pada soal itu dan 40,4% pada sisanya.
Tidak ada yang bisa dieksploitasi, karena pengecohnya memang diambil dari
kosakata lagu yang sama sehingga terulangnya sebuah kata tidak menunjuk ke mana
pun.

*Kunci bisa ditebak dari panjang opsi.* Strategi "pilih yang terpendek"
mendapat 21,1% dan "pilih yang terpanjang" 20,5%, terhadap peluang acak 20%.
Tiga kategori sedikit lebih condong — `asal` 36,4%, `birama` 33,3%,
`nada_tertinggi` 35,4% — dan kecondongan itu melekat pada datanya, karena nama
daerah yang benar memang cenderung lebih pendek daripada pengecohnya. Karena
seluruhnya masih di bawah lantai kontrol sadar-distribusi yang 29,8%,
melaporkannya cukup; menambalnya akan mengubah isi soal demi angka.

*Huruf kapital.* Kunci berhuruf besar pada 62,5% soal, pengecoh 63,0%. Selisih
0,5 poin, tidak ada petunjuk.

**Yang lulus tanpa catatan:** jumlah opsi selalu lima, tidak ada opsi kosong,
kunci selalu ada di antara opsinya, tidak ada opsi kembar persis, seluruh kunci
fakta cocok dengan CSV, tidak ada soal birama yang dibuat dari 50 lagu berbirama
disimpulkan, tidak ada opsi memakai judul ganda *Desaku*, sebaran huruf kunci
19,1–21,2% terhadap harapan 20%, dan tidak ada satu pun lagu yang muncul di
kedua sisi berkas split.

**Akibatnya pada versi.** Benchmark teks bebas naik ke v1.5 dan pilihan ganda
ke v1.3. Seluruh prediksi yang dibuat atas versi sebelumnya menjadi tidak
berlaku dan ditolak penilai, bukan dinilai diam-diam.

**Satu keterbatasan yang wajib ditulis.** Arm sapuan hiperparameter dilatih
pada data latih v1.4, yang masih memuat ejaan lama, lalu dinilai dengan kunci
v1.5. Sembilan lagu penciptanya berubah, menyentuh 14 dari 1.002 soal teks
bebas (1,4%) dan 12 dari 1.200 soal pilihan ganda (1,0%). Selisih sekecil itu
berada di bawah ambang keterbedaan seed tunggal yang sudah dijelaskan di §5.5,
sehingga arm-arm itu tetap sah dibandingkan satu sama lain. Model unggulan
dilatih ulang pada v1.5 dengan tag `penuh15`, dan itulah yang diterbitkan.

### 5.10 Lantai kontrol pada v1.5 / MC v1.3

Diukur ulang setelah benchmark dibangun ulang. Seluruh angka model wajib dibaca
terhadap baris ini, bukan terhadap nol.

| kontrol | pilihan ganda v1.3 | teks bebas v1.5 (toleran) |
|---|---|---|
| tebakan sadar-distribusi | **32,1%** | — |
| tebakan tersering | 21,2% | 32,8% |
| tebakan acak | 17,2% | 26,5% |
| tidak menjawab | 0,0% | 0,0% |

`kontrol-kosong` mencetak 0,0% di kedua jalur. Itu pemeriksaan terhadap penilai
sendiri: kalau jawaban kosong mendapat angka di atas nol, ada aturan penilaian
yang bocor. Nol berarti tidak ada.

Lantai pilihan ganda naik dari 29,8% ke 32,1% setelah nama pencipta dibakukan.
Sebabnya masuk akal dan bukan tanda bahaya: menggabungkan tiga ejaan Ismail
Marzuki menjadikannya pencipta 10 lagu, bukan 7, sehingga penebak yang hafal
sebaran jawaban punya tebakan yang lebih sering benar. Lantai yang naik membuat
klaim tentang model jadi lebih sulit, bukan lebih mudah.

**Satu jebakan operasional yang hampir merusak hasil.** Kedua akun Modal punya
volume `laguqa-data` yang terpisah, dan `upload` hanya menulis ke volume profil
yang sedang dipakai. Soal v1.5 sempat terunggah ke akun utama saja, sehingga
prediksi di akun kedua berjalan dua jam penuh menjawab soal versi lama.
Ketahuan hanya karena tajuk versi pada berkas prediksi memuat sha berkas
soalnya; tanpa itu angkanya akan terlihat wajar dan masuk laporan. Sesudah
membangun ulang benchmark, `upload` wajib dijalankan di kedua profil sebelum
prediksi apa pun.

### 5.11 Hasil sapuan hiperparameter

Angka di bawah dinilai dengan cara ketiga (peluang teks jawaban, §5.12) atas
1.200 soal pilihan ganda v1.3. Versi bagian ini yang lebih awal memakai cara
kedua dan menyimpulkan sebaliknya; itu tercatat di §5.12 karena pembalikannya
sendiri adalah temuan.

| arm | epoch | akurasi | vs dasar | Fakta | Notasi | Penalaran | biaya |
|---|---|---|---|---|---|---|---|
| **lr = 4e-4** | 2 | **61,0%** | **+9,6** | 71,2% | 46,1% | 89,7% | $2,19 |
| dasar (r=16, lr=2e-4) | 3 | 51,4% | — | 51,4% | 16,6% | 97,6% | $2,99 |
| dasar | 2 | 50,3% | −1,1 | 51,9% | 15,0% | 95,5% | — |
| r = 32 | 2 | 50,0% | −1,4 | 52,2% | 19,2% | 89,3% | $1,58 |
| dasar | 1 | 46,1% | −5,3 | 46,5% | 17,1% | 87,2% | — |
| lr = 1e-4 | 2 | 46,0% | −5,4 | 47,3% | 17,6% | 87,9% | $2,08 |
| r = 8 | 2 | 45,7% | −5,7 | 45,4% | 15,0% | 87,2% | $2,13 |

**Laju belajar berpengaruh, rank tidak.** lr=4e-4 unggul 9,6 poin atas dasar,
z=4,74, bertahan setelah koreksi Bonferroni untuk lima perbandingan. Sebaliknya
r=32 tidak berbeda dari dasar (z=−0,69) dan r=8 justru lebih buruk (z=−2,79).
Menurunkan laju belajar ke 1e-4 merugikan sebesar menurunkan rank ke 8.

Perbedaannya paling besar justru di kategori tersulit. Pada `notasi_ke_judul`,
satu-satunya kategori yang menuntut pembacaan notasi untuk mengenali lagu,
lr=4e-4 mencapai 46,1% sementara seluruh arm lain berada di 15–19%. Dasar yang
dilatih tiga epoch penuh hanya 16,6%, jadi yang membatasi bukan lamanya
pelatihan melainkan besarnya langkah.

**Epoch ketiga tetap tidak sepadan.** Dasar bergerak 46,1% → 50,3% → 51,4%.
Epoch ketiga menambah 1,1 poin dengan biaya sepertiga run, dan selisih itu
tidak dapat dibedakan dari kebetulan.

**Satu hasil yang belum terjelaskan.** `penuh15`, yang dilatih pada data v1.5
yang sudah dibakukan nama penciptanya dan mencatat eval loss lebih baik
(0,2131 berbanding 0,2500), justru mencetak 46,2% terhadap 51,4% milik `penuh`
(z=−2,55). Keduanya run terpisah ber-seed tunggal, sehingga selisih ini bisa
saja ragam antar-run; tetapi arahnya berlawanan dengan eval loss-nya dan itu
belum terjelaskan. Ditulis apa adanya. Model yang diterbitkan karena itu
dilatih ulang dengan tag `final`: data v1.5 dan lr=4e-4 sekaligus.

**Batasannya.** Benih tunggal, satu model dasar, dan arm sapuan dilatih pada
data v1.4 sedangkan dasar `penuh` juga v1.4, sehingga keduanya sebanding.
Untuk membedakan selisih di bawah sekitar 5 poin pada n=1.200 diperlukan
beberapa seed per arm, dan anggarannya tidak ada.

### 5.12 Cara menilai pilihan ganda, dan dua kali salah sebelum benar

Bagian ini mencatat tiga cara penilaian yang dicoba berurutan pada 2 September
2026. Ditulis lengkap dengan kesalahannya karena urutan itu sendiri yang
menjadi buktinya: tiap cara terlihat benar sampai diukur.

**Cara pertama, membaca jawaban dari teks yang dibangkitkan.** Cara ini
menghukum model yang bertele-tele. `LFM2.5-2.6B` dan `granite-4.2-8b`
menuliskan penalarannya terbuka tanpa tag `<think>` — "The user is asking about
the highest note..." — lalu menembus anggaran 1.024 token dalam keadaan masih
berpikir. Jawabannya terpotong di tengah kalimat tanpa pernah menyebut satu
huruf pun. Dinilai begini keduanya mencetak nol dan akan ditulis sebagai model
yang tidak mengenal lagu Indonesia. Menaikkan anggaran token tidak
menyelesaikannya, hanya memindahkan ambangnya, dan run yang sempat berjalan
menuju sekitar tiga jam per model.

**Cara kedua, peluang model atas kelima huruf pilihan.** Satu forward pass,
tanpa membangkitkan apa pun, jadi tidak ada yang bisa terpotong. Waktunya turun
dari tiga jam menjadi beberapa menit. Cara ini yang dipakai benchmark seperti
MMLU, dan sempat terlihat sebagai jawabannya.

Ternyata bukan. Cara ini menukar satu ketimpangan dengan ketimpangan lain: ia
menghukum model yang menjawab dengan isi pilihan, bukan dengan hurufnya.
Terbaca jelas pada arm sapuan `lr4e4`, yang **terbaik** dengan cara pertama
(55,4%) tetapi hampir **terburuk** dengan cara kedua (35,4%). Arm itu juga yang
angka "terbaca"-nya paling rendah pada cara pertama, 93,7%, yaitu paling sering
menjawab dengan isi. Pembalikan urutan itu yang menjadi tanda bahaya.

**Cara ketiga, peluang teks jawabannya.** Untuk tiap pilihan dihitung rerata
log-peluang token-tokennya bila dilanjutkan dari pertanyaan; yang tertinggi
dipilih. Cara ini tidak menuntut model mengenal konvensi A sampai E sama
sekali, sehingga netral terhadap kedua ketimpangan di atas. Rerata, bukan
jumlah, supaya pilihan yang panjang tidak dihukum karena panjangnya.

Selisihnya besar. Pada 320 soal yang sama, `lr4e4` mendapat **63,4%** dengan
cara ketiga dan 39,4% dengan cara kedua. Dua puluh empat poin itu pengetahuan
yang sebelumnya tertutup keengganan model mengeluarkan huruf.

**Pemeriksaan yang membuktikan cara ketiga tidak sekadar longgar.** Diuji lebih
dulu pada `LFM2.5-2.6B`, model yang tidak tahu apa-apa: cara ketiga memberinya
16,9% dan cara kedua 20,6%, keduanya di peluang acak atau di bawahnya. Cara
ketiga tidak menaikkan angka model yang memang tidak tahu; ia hanya berhenti
menghukum model yang tahu tetapi menjawab dengan cara lain.

**Dua hal teknis yang menentukan benar-tidaknya.** Penilaian huruf memerlukan
padding rata kiri agar posisi terakhir benar-benar akhir prompt, dan
`logits_to_keep=1` agar tidak memateralisasi tensor [16 x 768 x kosakata] yang
mencapai 6,3 GB dan mematikan model 9B di kartu 24 GB. Penilaian teks jawaban
memerlukan padding rata kanan, supaya prompt menempati posisi yang sama di
kelima barisnya.

Kelima pilihan diproses sekaligus, bukan satu per satu. Versi cepat diuji
terhadap versi lambat pada 320 soal dan menghasilkan **320 dari 320 jawaban
identik**, sehingga percepatannya tidak menyentuh angkanya.

**Yang dipakai di laporan.** Jalur pilihan ganda memakai cara ketiga untuk
seluruh baris, karena satu kolom hanya bermakna kalau seluruh isinya diukur
sama. Jalur teks bebas tetap memakai generasi, karena di sana jawabannya memang
teks. Angka dari cara pertama dan kedua disimpan sebagai pemeriksa silang, dan
nama berkasnya dibedakan dengan akhiran metode supaya tidak pernah tercampur
dalam satu kolom.

**Yang wajib ditulis sebagai keterbatasan.** Urutan antar-arm sapuan berbeda
menurut cara penilaian. Kesimpulan utama — model tanpa pelatihan tidak mengenal
buku ini, dan pelatihan mengajarkannya — kokoh pada ketiga cara. Yang tidak
kokoh adalah peringkat antar-arm, dan itu tidak boleh dilaporkan seolah-olah
kokoh.
