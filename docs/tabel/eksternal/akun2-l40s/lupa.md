## IndoMMLU

Tebakan acak 23.4%, 2000 soal.

| model | prompt | akurasi | selisih dari base |
|---|---|---:|---:|
| base | lagu | 35.2 | — |
| penuh | lagu | 42.8 | +7.6 |
| penuh15 | lagu | 38.3 | +3.1 |
| base | netral | 34.4 | — |
| penuh | netral | 41.8 | +7.4 |
| penuh15 | netral | 37.9 | +3.5 |

Selisih per grup, prompt lagu.

| grup | n | base | penuh | penuh15 |
|---|---:|---:|---:|---:|
| Humanities | 308 | 38.0 | +9.1 | +4.5 |
| Indonesian language | 431 | 36.2 | +5.3 | +1.2 |
| Local languages and cultures | 492 | 32.3 | +6.3 | +2.6 |
| STEM | 399 | 31.1 | +7.5 | +2.8 |
| Social science | 370 | 40.0 | +10.5 | +5.1 |

Selisih per subjek, prompt lagu.

| subjek | n | base | penuh | penuh15 |
|---|---:|---:|---:|---:|
| Agama Hindu | 21 | 42.9 | +0.0 | -4.8 |
| Agama Islam | 94 | 39.4 | +13.8 | +9.6 |
| Agama Kristen | 27 | 40.7 | +3.7 | +0.0 |
| Bahasa Bali | 63 | 30.2 | +3.2 | +7.9 |
| Bahasa Banjar | 19 | 15.8 | +10.5 | +10.5 |
| Bahasa Dayak Ngaju | 14 | 21.4 | +21.4 | +14.3 |
| Bahasa Indonesia | 431 | 36.2 | +5.3 | +1.2 |
| Bahasa Jawa | 133 | 37.6 | +9.8 | +4.5 |
| Bahasa Lampung | 19 | 36.8 | -15.8 | -15.8 |
| Bahasa Madura | 38 | 31.6 | -2.6 | +0.0 |
| Bahasa Makassar | 25 | 32.0 | -4.0 | +0.0 |
| Bahasa Sunda | 155 | 31.6 | +9.0 | +1.3 |
| Biologi | 113 | 32.7 | +3.5 | +3.5 |
| Budaya Alam Minangkabau | 26 | 30.8 | +7.7 | -3.8 |
| Ekonomi | 66 | 33.3 | +13.6 | +4.5 |
| Fisika | 66 | 30.3 | +6.1 | +1.5 |
| Geografi | 65 | 32.3 | +9.2 | +6.2 |
| IPA | 129 | 35.7 | +10.9 | +0.0 |
| IPS | 80 | 48.8 | +8.7 | +5.0 |
| Kesenian | 81 | 35.8 | +12.3 | +8.6 |
| Kimia | 91 | 23.1 | +8.8 | +6.6 |
| PPKN | 93 | 44.1 | +10.8 | +6.5 |
| Penjaskes | 19 | 47.4 | +10.5 | +0.0 |
| Sejarah | 66 | 33.3 | +3.0 | -1.5 |
| Sosiologi | 66 | 37.9 | +10.6 | +3.0 |

Selisih per grup, prompt netral.

| grup | n | base | penuh | penuh15 |
|---|---:|---:|---:|---:|
| Humanities | 308 | 38.0 | +6.8 | +1.6 |
| Indonesian language | 431 | 32.5 | +8.6 | +3.0 |
| Local languages and cultures | 492 | 32.9 | +3.9 | +3.5 |
| STEM | 399 | 31.8 | +6.5 | +3.5 |
| Social science | 370 | 38.4 | +11.9 | +5.7 |

Selisih per subjek, prompt netral.

| subjek | n | base | penuh | penuh15 |
|---|---:|---:|---:|---:|
| Agama Hindu | 21 | 42.9 | -9.5 | -4.8 |
| Agama Islam | 94 | 38.3 | +11.7 | +6.4 |
| Agama Kristen | 27 | 37.0 | +7.4 | +3.7 |
| Bahasa Bali | 63 | 30.2 | +1.6 | +7.9 |
| Bahasa Banjar | 19 | 10.5 | +10.5 | +15.8 |
| Bahasa Dayak Ngaju | 14 | 14.3 | +7.1 | +28.6 |
| Bahasa Indonesia | 431 | 32.5 | +8.6 | +3.0 |
| Bahasa Jawa | 133 | 39.1 | +3.8 | +7.5 |
| Bahasa Lampung | 19 | 42.1 | -15.8 | -21.1 |
| Bahasa Madura | 38 | 31.6 | -2.6 | +0.0 |
| Bahasa Makassar | 25 | 32.0 | +0.0 | +0.0 |
| Bahasa Sunda | 155 | 32.9 | +7.7 | -0.6 |
| Biologi | 113 | 33.6 | +5.3 | +3.5 |
| Budaya Alam Minangkabau | 26 | 30.8 | +7.7 | +0.0 |
| Ekonomi | 66 | 33.3 | +15.2 | +3.0 |
| Fisika | 66 | 28.8 | +12.1 | +6.1 |
| Geografi | 65 | 29.2 | +9.2 | +7.7 |
| IPA | 129 | 38.0 | +1.6 | +0.0 |
| IPS | 80 | 47.5 | +11.2 | +6.2 |
| Kesenian | 81 | 42.0 | +3.7 | -1.2 |
| Kimia | 91 | 23.1 | +11.0 | +6.6 |
| PPKN | 93 | 41.9 | +11.8 | +5.4 |
| Penjaskes | 19 | 47.4 | +5.3 | +0.0 |
| Sejarah | 66 | 28.8 | +9.1 | +0.0 |
| Sosiologi | 66 | 36.4 | +12.1 | +6.1 |

## IndoCulture

Tebakan acak 33.3%, 2429 soal.

| model | prompt | akurasi | selisih dari base |
|---|---|---:|---:|
| base | lagu | 58.7 | — |
| penuh | lagu | 57.6 | -1.1 |
| penuh15 | lagu | 55.7 | -2.9 |
| base | netral | 59.0 | — |
| penuh | netral | 59.0 | -0.1 |
| penuh15 | netral | 55.8 | -3.2 |

Selisih per topik, prompt lagu.

| topik | n | base | penuh | penuh15 |
|---|---:|---:|---:|---:|
| Hari Besar Agama | 183 | 69.9 | -1.6 | -2.7 |
| Hubungan Keluarga | 208 | 56.7 | -7.7 | -9.1 |
| Ikan dan Ternak | 107 | 48.6 | +3.7 | -2.8 |
| Kehamilan, Bayi, Anak | 256 | 62.5 | -5.9 | -5.1 |
| Kematian | 159 | 68.6 | -3.8 | -3.8 |
| Makanan | 369 | 53.7 | +1.9 | +0.3 |
| Permainan | 96 | 56.2 | +1.0 | -1.0 |
| Pernikahan | 320 | 54.4 | -1.9 | -1.9 |
| Pertanian | 98 | 62.2 | +1.0 | -3.1 |
| Sehari-hari | 146 | 55.5 | +4.1 | -0.7 |
| Seni | 272 | 58.1 | +1.1 | +0.0 |
| Socio-religious | 215 | 61.4 | -1.4 | -7.0 |

Selisih per provinsi, prompt lagu.

| provinsi | n | base | penuh | penuh15 |
|---|---:|---:|---:|---:|
| Aceh | 246 | 63.0 | -2.4 | -2.4 |
| Bali | 241 | 75.5 | -2.9 | -3.7 |
| Jawa Barat | 231 | 66.7 | +1.3 | -5.2 |
| Jawa Tengah | 233 | 56.7 | -3.9 | -2.6 |
| Jawa Timur | 171 | 57.9 | +2.9 | +1.8 |
| Kalimantan Selatan | 233 | 51.1 | -0.4 | -1.3 |
| NTT | 103 | 59.2 | -6.8 | -8.7 |
| Papua | 253 | 64.8 | +1.6 | +0.0 |
| Sulawesi Selatan | 185 | 53.0 | -1.1 | -0.5 |
| Sumatera Barat | 299 | 46.5 | +0.7 | -2.7 |
| Sumatera Utara | 234 | 52.1 | -3.8 | -8.5 |

Selisih per cakupan, prompt lagu.

| cakupan | n | base | penuh | penuh15 |
|---|---:|---:|---:|---:|
| khas-provinsi | 1847 | 54.8 | -0.3 | -1.1 |
| umum | 582 | 70.8 | -3.6 | -8.6 |

Selisih per topik, prompt netral.

| topik | n | base | penuh | penuh15 |
|---|---:|---:|---:|---:|
| Hari Besar Agama | 183 | 69.9 | +1.1 | -1.1 |
| Hubungan Keluarga | 208 | 56.7 | -8.2 | -6.7 |
| Ikan dan Ternak | 107 | 53.3 | +3.7 | -6.5 |
| Kehamilan, Bayi, Anak | 256 | 61.7 | -4.7 | -2.3 |
| Kematian | 159 | 69.8 | -3.1 | -4.4 |
| Makanan | 369 | 53.9 | +2.4 | +0.8 |
| Permainan | 96 | 55.2 | +4.2 | -2.1 |
| Pernikahan | 320 | 52.5 | +0.9 | -2.5 |
| Pertanian | 98 | 67.3 | -6.1 | -7.1 |
| Sehari-hari | 146 | 56.8 | +2.1 | -1.4 |
| Seni | 272 | 58.5 | +1.5 | -2.6 |
| Socio-religious | 215 | 62.3 | +4.2 | -8.8 |

Selisih per provinsi, prompt netral.

| provinsi | n | base | penuh | penuh15 |
|---|---:|---:|---:|---:|
| Aceh | 246 | 63.8 | +0.0 | -0.4 |
| Bali | 241 | 76.8 | -4.6 | -5.4 |
| Jawa Barat | 231 | 64.1 | +6.1 | -2.2 |
| Jawa Tengah | 233 | 56.2 | +0.4 | -3.0 |
| Jawa Timur | 171 | 59.6 | +2.3 | -3.5 |
| Kalimantan Selatan | 233 | 52.4 | -0.4 | -2.1 |
| NTT | 103 | 59.2 | +0.0 | -3.9 |
| Papua | 253 | 64.8 | +0.8 | +1.2 |
| Sulawesi Selatan | 185 | 52.4 | -0.5 | -1.1 |
| Sumatera Barat | 299 | 48.8 | -1.3 | -7.4 |
| Sumatera Utara | 234 | 51.7 | -2.6 | -6.8 |

Selisih per cakupan, prompt netral.

| cakupan | n | base | penuh | penuh15 |
|---|---:|---:|---:|---:|
| khas-provinsi | 1847 | 55.4 | +0.2 | -1.9 |
| umum | 582 | 70.4 | -0.9 | -7.4 |

