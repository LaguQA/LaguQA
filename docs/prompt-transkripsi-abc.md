# Alur transkripsi notasi angka ke ABC

Prompt yang dipakai ada di **[skema/prompt.txt](../skema/prompt.txt)**. Berkas
ini hanya menjelaskan alasan di balik isinya dan cara menjalankannya, supaya
promptnya sendiri tetap satu dan tidak bercabang.

Versi lama diarsipkan di `skema/prompt-v1-lama.txt`.

---

## Mengapa prompt lama direvisi

Prompt versi pertama sudah rinci, tetapi mengandung satu kesalahan yang justru
menyebabkan kesalahan terbesar pada keluaran. Pada bagian 4.3 tertulis:

> Uppercase C–B = the octave below middle C region (C3–B3); lowercase c–b =
> middle octave (C4–B4) ... so that 1 ⇒ the tonic letter in the middle register
> (lowercase ...)

Di ABC 2.1, huruf besar `C` justru *middle C*, dan huruf kecil `c` satu oktaf di
atasnya. Karena promptnya menyuruh memetakan angka 1 ke huruf kecil, seluruh
lagu naik satu oktaf. Inilah sumber kesalahan oktaf pada Timang-Timang dan
Injit-Injit Semut, dan modelnya sebenarnya hanya menuruti perintah.

Kesalahan ini juga menerangkan mengapa gerbang konservasi ketukan tidak
menangkapnya: menggeser semua nada satu oktaf tidak mengubah durasi sama sekali.

Empat perubahan lain:

1. **Keluaran hanya ABC.** Prompt lama meminta *verification dossier* di luar
   blok kode. Sekarang seluruh jawaban wajib satu blok berpagar, dan semua
   catatan keraguan masuk sebagai komentar `%` di dalamnya.
2. **Kontrak kepala berkas.** Urutan field ditetapkan dan wajib lengkap, dengan
   `-` untuk yang kosong, ditambah baris `% laguqa-do` dan `% laguqa-halaman`.
   Ini yang membuat audit bisa dikerjakan mesin, bukan dibaca satu per satu.

   Nomor lagu dan nama gambar sengaja **tidak** diminta dari model. Model tidak
   melihat nama berkas gambar yang diunggah, jadi jawabannya pasti karangan.
   Percobaan pertama membuktikannya: Bungong Jeumpa yang bernomor 0 diberi
   `id 11` dan berkas `11_bungong_jeumpa_p1.jpg` yang tidak ada, diambil dari
   nomor halaman. Keduanya kini diisi `abc_ingest.py` dari nama berkas masukan.
3. **Lima kesalahan nyata ditaruh di depan**, lengkap dengan contoh bar
   terpandu, menggantikan sepuluh "hukum" abstrak yang tidak mencegah apa pun.
4. **Bagian 4.11 yang lama dihapus.** Isinya berdebat dengan dirinya sendiri
   ("prefer explicit ... NO: prefer ..."), jadi tidak mungkin dipatuhi.

---

## Menjalankan

**1. Kirim gambar.** Ambil dari `scans_prep/`, bukan `clean database buku/`.
Berkas di sana sudah diratakan pencahayaannya, garis balok dan titik oktaf
terbaca jelas, dan ukurannya enam kali lebih kecil. Lagu berhalaman dua
(`_p1` dan `_p2`) dikirim sekaligus dalam satu percakapan.

**2. Tempel isi `skema/prompt.txt`.** Satu lagu satu percakapan.

**3. Simpan jawabannya apa adanya** ke berkas yang sudah disiapkan di
`abc/gemini/`. Seluruh 107 nama dibuat sekaligus di muka:

```
tools/.venv/bin/python tools/abc_stub.py --apply
```

Berkas kosong berisi tiga baris komentar: nomor lagu, judulnya, dan nama gambar
yang harus diunggah. Ganti seluruh isinya dengan jawaban model. Tidak perlu
dibersihkan; pagar markdown dan basa-basi akan dibuang otomatis.

Nama berkas dibuat di muka karena namanya itulah satu-satunya sumber nomor
lagu, dan mengetiknya 107 kali membuka peluang salah ketik yang baru ketahuan
berbulan-bulan kemudian. Menjalankan `abc_stub.py` tanpa `--apply` menampilkan
kemajuan pengisian. Sebagai jaring pengaman kedua, `abc_ingest.py`
membandingkan `T:` pada keluaran dengan judul menurut dataset, sehingga
keluaran lagu lain yang tersimpan di nomor keliru langsung ditolak.

**4. Terima dan periksa:**

```
tools/.venv/bin/python tools/abc_ingest.py abc/gemini/*.abc          # periksa saja
tools/.venv/bin/python tools/abc_ingest.py abc/gemini/*.abc --apply  # simpan ke abc/
```

Berkas yang masih kosong dilewati, jadi glob di atas boleh dijalankan sejak
lagu pertama selesai.

Skrip ini membuka pagar markdown, memeriksa kontrak kepala berkas, memastikan
nama gambarnya benar-benar ada dan berawalan id yang sama, mendeteksi keluaran
yang terpotong, menjalankan validator, lalu memberi nama berkas dari
`% laguqa-id`. Berkas yang melanggar kontrak tidak disimpan.

**5. Satukan ke dataset:**

```
tools/.venv/bin/python tools/abc_meta.py --apply   # composer, origin, do, birama, tempo, lirik
tools/.venv/bin/python tools/abc_sync.py --apply   # abc_notation dan abc_status
```

Kedua skrip memisahkan diri karena sumbernya berbeda. `abc_sync.py` menyalin
notasi dari `abc/`, yaitu berkas yang sudah lulus kontrak, sedangkan
`abc_meta.py` membaca `abc/gemini/` apa adanya: metadata tidak bergantung pada
benar tidaknya nada, jadi tidak perlu menunggu berkas lulus validator.

Enam kolom itu tidak diketik ulang ke spreadsheet karena model sudah
membacanya sewaktu mentranskripsi, dan kepala berkas ABC dijadikan satu-satunya
tempat metadata ditulis. Kolom yang sudah terisi tidak ditimpa kecuali diminta
`--timpa`.

---

## Menyusun ulang lirik

Baris `w:` bukan kalimat. Isinya suku kata beserta tanda penyelarasan dengan
not, dan tiga di antaranya mengubah bacaan:

| tanda | arti |
|---|---|
| `-` | suku kata masih satu kata dengan berikutnya |
| `_` | not tambahan untuk suku kata sebelumnya |
| `*` | not tanpa suku kata |
| `~` | dua kata pada satu not |
| `\-` | tanda hubung yang memang tercetak |

Dua jebakan membuat penggabungan naif menghasilkan teks yang salah.

**Bait ditumpuk secara menegak.** Bait kedua tersebar sebagai baris `w:` kedua
di bawah setiap baris musik, bukan sebagai kumpulan baris di bagian bawah
berkas. Membaca berkas dari atas ke bawah menganyam semua baitnya jadi satu.
Syukur yang berbait tiga terbaca "Dari yakinku teguh hati / ikhlasDari yakinku
teguh cinta / ikhlasDari yakinku teguh bakti". Dua puluh dua dari delapan puluh
enam lagu berbentuk begini, jadi seperempat dataset akan salah kalau baris `w:`
tidak dikelompokkan lebih dahulu menurut baris musik yang diikutinya.

**Pemenggalan baris musik tidak selalu jatuh di batas kata.** "Garuda Pancasila
A-" berlanjut ke "ku-lah" pada baris berikutnya, dan menyusun tiap baris
sendiri-sendiri membelah "Akulah" jadi dua.

Sisanya tidak bisa dibereskan skrip. Model kerap membubuhkan `-` di akhir baris
`w:` padahal katanya sudah selesai, sehingga dua kata menyatu: `ikhlasDari`,
`berseruIndonesia`, `kulitDi`. Ini tidak terbedakan dari sambungan yang benar
(`tariKan`) maupun dari kata yang memang ditulis berkapital (`karuniaMu`) tanpa
kamus, jadi `abc_meta.py` melaporkannya sebagai "kata menyatu?" dan tidak
mengubah apa pun. Perbaikannya di berkas ABC-nya, bukan di dataset.

## Memperbaiki bar yang gagal

Validator menyebut nomor baris dan isi barnya. Potong bagian halaman itu lalu
perbesar:

```
tools/.venv/bin/python tools/crop_scan.py scans_prep/<berkas> --box 0.05,0.14,0.55,0.21
```

Kotak potong dinyatakan sebagai proporsi `kiri,atas,kanan,bawah` terhadap
halaman. Pada perbesaran ini garis balok terbaca jelas, dan garis balok itulah
yang menentukan nilai nada.

Peringatan nada tidak menghentikan proses, tetapi wajib dilihat manual. Ambitus
di atas 17 semiton hampir selalu berarti ada kelompok nada yang salah oktaf.

Ambitus saja tidak cukup. Menggeser seluruh lagu satu oktaf tidak mengubah
ambitusnya sedikit pun, sehingga kesalahan oktaf menyeluruh lolos begitu saja.
Enam dari tujuh transkripsi hasil prompt lama ternyata naik satu oktaf penuh
dan tetap dinyatakan bersih, karena bar, lirik, dan ambitusnya memang benar
semua. Validator karena itu memeriksa juga nada tengah lagu, yang untuk
nyanyian bersama tidak wajar berada di atas C5.

Ambangnya kemudian dikalibrasi ulang setelah seluruh 107 selesai, dan ternyata
angka pertama itu memang terlalu ketat. C5 diambil dari tiga belas berkas awal
yang hampir semuanya lagu daerah; lagu nasional beregister lebih tinggi,
kuartil-3 nada tengahnya C5 lawan A4 pada lagu daerah. Ambang lama menandai 13
berkas, dan Berkibarlah Benderaku sudah dipastikan ke halaman bukunya **benar**:
buku memang memberi titik oktaf naik pada `3 1 2 3` di sana, dan Gemini
menyalinnya dengan tepat.

Ambang sekarang D5, menyisakan empat berkas. Isyarat ini tetap lemah dan tidak
bisa dipertajam dari data yang ada: sebaran nada tengah menyambung dari C4
sampai G5 tanpa celah, sehingga ambang mana pun memotong di tengah sebaran.
Perlakukan hasilnya sebagai bahan periksa manual, bukan vonis.

## Buku menggambar garis birama tiap setengah bar

Pada sebagian lagu, buku mencetak birama 4/4 di kepala halaman tetapi menggambar
garis birama tiap dua ketuk. Transkripsi yang setia pada halaman lalu berisi bar
separuh kuota semua, dan Manuk Dadali gagal di 58 bar sekaligus karena ini —
padahal nada dan liriknya benar seluruhnya.

Yang diperbaiki notasinya, bukan `M:`-nya. Menurunkan `M:` menjadi 2/4 memang
membuat validator lulus, tetapi kolom `time_signature` lalu bertentangan dengan
angka yang tercetak di buku, dan itulah yang ditanyakan soal. Jadi tiap dua bar
tercetak digabung menjadi satu bar 4/4, tanpa memindahkan satu not pun sehingga
penyelarasan lirik tidak tersentuh.

Bar hasil gabungan tidak selalu berhenti di ujung baris. Karena itu validator
kini membawa bar yang belum tertutup ke baris berikutnya, sebagaimana ABC 2.1
memang mengizinkan: baris yang tidak diakhiri `|` bersambung. Sebelumnya bar
begitu terhitung dua kali, keduanya kurang panjang.

Perlu diingat garis birama rangkap di pergantian baris — `... |` di akhir baris
lalu `| ...` di awal baris berikutnya. Keduanya menutup bar yang sama, jadi
kalau digabung keduanya harus ikut hilang; menyisakan salah satunya membuat bar
tetap tertutup di tempat semula.
