# Biaya cloud computing Modal

Catatan biaya seluruh pekerjaan LaguQA pada layanan Modal, disalin dari dasbor
penagihan kedua akun pada 3 September 2026. Kurs yang dipakai **Rp17.685,59**
per dolar Amerika Serikat.

Tangkapan layar aslinya disimpan di `docs/bukti/`. Berkas gambarnya perlu
disalin sendiri ke folder itu karena hanya dikirim lewat percakapan, bukan
sebagai berkas di cakram.

## Ringkasan

| | Akun pertama | Akun kedua | Jumlah |
|---|---|---|---|
| Nama akun | IRedDragonICY | vhpwdar | |
| Profil CLI | bawaan | `akun1` | |
| Terpakai menurut penghitung | $28,30 | $29,84 | **$58,14** |
| Rupiah | Rp500.502 | Rp527.738 | **Rp1.028.240** |

Rincian per aplikasi berjumlah $58,16 dan rincian per sumber daya berjumlah
$58,77. Selisihnya kecil dan berasal dari pembulatan, kecuali satu selisih
$0,60 pada akun kedua yang dijelaskan di bawah.

## Biaya menurut kegiatan

Sumber: panel *Ephemeral App Breakdown* pada kedua akun.

| Kegiatan | Akun pertama | Akun kedua | Jumlah (USD) | Jumlah (Rp) |
|---|---|---|---|---|
| `modal_train.run` — pelatihan | $21,24 | $18,51 | $39,75 | Rp703.002 |
| `modal_train.predict` — penilaian | $6,69 | $11,33 | $18,02 | Rp318.694 |
| `modal_train.smoke` — uji asap | $0,27 | — | $0,27 | Rp4.775 |
| `modal_train.probe` — pemeriksaan | $0,11 | — | $0,11 | Rp1.945 |
| `<image build>` | $0,01 | $0,00 | $0,01 | Rp177 |
| `modal_train.inspect_api` | $0,00 | — | $0,00 | Rp0 |
| **Jumlah** | **$28,32** | **$29,84** | **$58,16** | **Rp1.028.594** |

Pelatihan 68,3 persen, penilaian 31,0 persen, sisanya 0,7 persen.

## Biaya menurut sumber daya

Sumber: panel *Resource Breakdown* pada kedua akun.

| Sumber daya | Akun pertama | Akun kedua | Jumlah (USD) | Jumlah (Rp) |
|---|---|---|---|---|
| L40S | $27,10 | — | $27,10 | Rp479.279 |
| L4 | $0,21 | $28,23 | $28,44 | Rp502.978 |
| A100 40GB | $0,12 | — | $0,12 | Rp2.122 |
| CPU | $0,70 | $1,77 | $2,47 | Rp43.683 |
| Memori | $0,19 | $0,45 | $0,64 | Rp11.319 |
| **Jumlah** | **$28,32** | **$30,45** | **$58,77** | **Rp1.039.382** |

Kartu grafis menyerap $55,66 atau 94,7 persen, sedangkan prosesor dan memori
container hanya $3,11 atau 5,3 persen.

## Dua selisih yang perlu diketahui

**Selisih $0,60 pada akun kedua.** Rincian sumber daya menyebut $30,44
sedangkan rincian aplikasi dan penghitung terpakai sama-sama menyebut $29,84.
Selisihnya belum tertelusur ke aplikasi mana pun. Dugaan yang paling masuk akal
adalah pemakaian yang tidak dilekatkan ke nama aplikasi, misalnya penyimpanan
volume. Angka yang dipakai di naskah adalah $29,84 karena cocok dengan
penghitung terpakai.

**Selisih $7,56 antara manifes dan tagihan pelatihan.** Manifes tiga belas
percobaan pelatihan yang selesai mencatat $32,19 untuk 29,1 jam kartu grafis,
sedangkan `modal_train.run` ditagih $39,75. Dua sebabnya:

1. `estimated_cost_usd` pada manifes hanya menghitung waktu kartu grafis
   dikali tarif per jam, tidak memuat prosesor dan memori container.
2. Percobaan yang berhenti di tengah jalan tidak meninggalkan manifes tetapi
   tetap ditagih. Satu di antaranya pelatihan yang mati di langkah 580 dari
   2673 karena batas lima jam sesi.

## Biaya pelatihan per percobaan

Dari `hasil/*-manifest.json`, medan `estimated_cost_usd`.

| Run | Kartu | Biaya | Lama |
|---|---|---|---|
| gemma4-e2b-full-s1-penuh15 | L40S | $5,128 | 157,8 mnt |
| gemma4-e2b-full-s1 | L40S | $3,500 | 107,7 mnt |
| gemma4-e2b-full-s1-seimbang | L40S | $3,475 | 106,9 mnt |
| gemma4-e2b-full-s1-penuh | L40S | $2,992 | 92,1 mnt |
| gemma4-e2b-full-s1-lr4e4 | L4 | $2,193 | 164,5 mnt |
| gemma4-e2b-full-s3-final | L4 | $2,145 | 160,9 mnt |
| gemma4-e2b-full-s1-r8 | L4 | $2,129 | 159,6 mnt |
| gemma4-e2b-full-s1-final | L4 | $2,114 | 158,6 mnt |
| gemma4-e2b-full-s1-lr1e4 | L4 | $2,076 | 155,7 mnt |
| gemma4-e2b-full14-s2-lr4e4 | L4 | $1,688 | 126,6 mnt |
| gemma4-e2b-full-s2-final | L4 | $1,626 | 121,9 mnt |
| gemma4-e2b-full-s1-r32 | L4 | $1,576 | 118,2 mnt |
| gemma4-e2b-full14-s3-lr4e4 | L4 | $1,550 | 116,3 mnt |
| **Jumlah** | | **$32,19** | **29,1 jam** |

## Lalu lintas keluar

Tidak ditagih pada kedua akun. Akun pertama 3,92 GiB dan akun kedua 3,73 GiB,
keduanya jauh di bawah jatah 1,0 TiB per siklus dengan tarif $0,04 per GiB.
