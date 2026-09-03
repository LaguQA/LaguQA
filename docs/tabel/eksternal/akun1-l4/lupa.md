## IndoMMLU

Tebakan acak 23.4%, 2000 soal.

| model | prompt | akurasi | selisih dari base |
|---|---|---:|---:|
| base | lagu | 35.1 | — |
| final | lagu | 44.0 | +8.9 |
| lr4e4 | lagu | 45.5 | +10.4 |
| base | netral | 34.4 | — |
| final | netral | 43.6 | +9.2 |
| lr4e4 | netral | 47.0 | +12.6 |

Model pembanding, akurasi mutlak, prompt netral.

| model | akurasi |
|---|---:|
| sahabatai-9b | 48.4 |
| gemma4-e4b | 36.6 |
| qwen-sealion-4b | 35.6 |
| ornith-9b | 34.6 |
| sealion-e2b | 34.2 |
| apertus-sealion-8b | 33.9 |
| sealion-v35-8b | 32.4 |
| qwen35-9b | 31.6 |
| smollm3-3b | 30.3 |
| qwen35-4b | 29.6 |
| lfm25-2b | 28.6 |
| granite42-8b | 27.9 |

Selisih per grup, prompt lagu.

| grup | n | base | final | lr4e4 |
|---|---:|---:|---:|---:|
| Humanities | 308 | 38.3 | +9.4 | +12.7 |
| Indonesian language | 431 | 36.4 | +9.7 | +11.1 |
| Local languages and cultures | 492 | 32.1 | +5.5 | +3.9 |
| STEM | 399 | 30.6 | +9.3 | +13.0 |
| Social science | 370 | 39.7 | +11.4 | +13.5 |

Selisih per subjek, prompt lagu.

| subjek | n | base | final | lr4e4 |
|---|---:|---:|---:|---:|
| Agama Hindu | 21 | 42.9 | +0.0 | +0.0 |
| Agama Islam | 94 | 40.4 | +13.8 | +17.0 |
| Agama Kristen | 27 | 40.7 | +7.4 | +11.1 |
| Bahasa Bali | 63 | 31.7 | +4.8 | -6.3 |
| Bahasa Banjar | 19 | 15.8 | +10.5 | +15.8 |
| Bahasa Dayak Ngaju | 14 | 21.4 | +0.0 | +7.1 |
| Bahasa Indonesia | 431 | 36.4 | +9.7 | +11.1 |
| Bahasa Jawa | 133 | 37.6 | +4.5 | +7.5 |
| Bahasa Lampung | 19 | 36.8 | +0.0 | -5.3 |
| Bahasa Madura | 38 | 31.6 | +0.0 | -7.9 |
| Bahasa Makassar | 25 | 32.0 | +0.0 | -4.0 |
| Bahasa Sunda | 155 | 30.3 | +9.0 | +7.1 |
| Biologi | 113 | 32.7 | +7.1 | +13.3 |
| Budaya Alam Minangkabau | 26 | 30.8 | +7.7 | +11.5 |
| Ekonomi | 66 | 33.3 | +18.2 | +7.6 |
| Fisika | 66 | 28.8 | +16.7 | +12.1 |
| Geografi | 65 | 32.3 | +6.2 | +9.2 |
| IPA | 129 | 34.9 | +13.2 | +20.9 |
| IPS | 80 | 48.8 | +10.0 | +16.2 |
| Kesenian | 81 | 35.8 | +13.6 | +21.0 |
| Kimia | 91 | 23.1 | +1.1 | +2.2 |
| PPKN | 93 | 44.1 | +8.6 | +16.1 |
| Penjaskes | 19 | 47.4 | +5.3 | +5.3 |
| Sejarah | 66 | 33.3 | +3.0 | +3.0 |
| Sosiologi | 66 | 36.4 | +15.2 | +16.7 |

Selisih per grup, prompt netral.

| grup | n | base | final | lr4e4 |
|---|---:|---:|---:|---:|
| Humanities | 308 | 38.0 | +8.8 | +14.3 |
| Indonesian language | 431 | 32.5 | +12.5 | +16.2 |
| Local languages and cultures | 492 | 32.7 | +5.3 | +4.7 |
| STEM | 399 | 32.3 | +8.0 | +14.3 |
| Social science | 370 | 38.1 | +12.4 | +15.9 |

Selisih per subjek, prompt netral.

| subjek | n | base | final | lr4e4 |
|---|---:|---:|---:|---:|
| Agama Hindu | 21 | 42.9 | +0.0 | +4.8 |
| Agama Islam | 94 | 38.3 | +13.8 | +19.1 |
| Agama Kristen | 27 | 37.0 | +3.7 | +14.8 |
| Bahasa Bali | 63 | 31.7 | +4.8 | +0.0 |
| Bahasa Banjar | 19 | 10.5 | +15.8 | +21.1 |
| Bahasa Dayak Ngaju | 14 | 14.3 | +14.3 | +7.1 |
| Bahasa Indonesia | 431 | 32.5 | +12.5 | +16.2 |
| Bahasa Jawa | 133 | 38.3 | +3.8 | +4.5 |
| Bahasa Lampung | 19 | 36.8 | +0.0 | +5.3 |
| Bahasa Madura | 38 | 31.6 | -2.6 | -15.8 |
| Bahasa Makassar | 25 | 32.0 | -4.0 | +0.0 |
| Bahasa Sunda | 155 | 32.9 | +7.7 | +9.7 |
| Biologi | 113 | 34.5 | +4.4 | +15.0 |
| Budaya Alam Minangkabau | 26 | 30.8 | +11.5 | +7.7 |
| Ekonomi | 66 | 31.8 | +16.7 | +12.1 |
| Fisika | 66 | 30.3 | +16.7 | +15.2 |
| Geografi | 65 | 29.2 | +6.2 | +15.4 |
| IPA | 129 | 37.2 | +10.1 | +21.7 |
| IPS | 80 | 47.5 | +16.2 | +16.2 |
| Kesenian | 81 | 42.0 | +7.4 | +17.3 |
| Kimia | 91 | 24.2 | +3.3 | +2.2 |
| PPKN | 93 | 41.9 | +10.8 | +18.3 |
| Penjaskes | 19 | 47.4 | +0.0 | +15.8 |
| Sejarah | 66 | 28.8 | +10.6 | +6.1 |
| Sosiologi | 66 | 36.4 | +12.1 | +16.7 |

## IndoCulture

Tebakan acak 33.3%, 2429 soal.

| model | prompt | akurasi | selisih dari base |
|---|---|---:|---:|
| base | lagu | 58.3 | — |
| final | lagu | 53.7 | -4.6 |
| lr4e4 | lagu | 56.4 | -1.9 |
| base | netral | 58.7 | — |
| final | netral | 54.6 | -4.1 |
| lr4e4 | netral | 56.9 | -1.8 |

Model pembanding, akurasi mutlak, prompt netral.

| model | akurasi |
|---|---:|
| sahabatai-9b | 65.5 |
| gemma4-e4b | 61.4 |
| sealion-e2b | 59.0 |
| qwen-sealion-4b | 57.6 |
| ornith-9b | 56.7 |
| qwen35-9b | 56.4 |
| apertus-sealion-8b | 56.3 |
| sealion-v35-8b | 53.9 |
| qwen35-4b | 49.9 |
| smollm3-3b | 49.0 |
| granite42-8b | 47.2 |
| lfm25-2b | 46.7 |

Selisih per topik, prompt lagu.

| topik | n | base | final | lr4e4 |
|---|---:|---:|---:|---:|
| Hari Besar Agama | 183 | 69.9 | -8.2 | -2.2 |
| Hubungan Keluarga | 208 | 55.8 | -5.8 | -1.4 |
| Ikan dan Ternak | 107 | 47.7 | -5.6 | -6.5 |
| Kehamilan, Bayi, Anak | 256 | 60.9 | -3.1 | -3.5 |
| Kematian | 159 | 67.3 | -5.7 | -1.3 |
| Makanan | 369 | 53.9 | -5.1 | -2.2 |
| Permainan | 96 | 57.3 | -3.1 | -4.2 |
| Pernikahan | 320 | 54.7 | -6.9 | -2.5 |
| Pertanian | 98 | 62.2 | -3.1 | -3.1 |
| Sehari-hari | 146 | 56.2 | +0.7 | +3.4 |
| Seni | 272 | 57.0 | -1.8 | +3.7 |
| Socio-religious | 215 | 61.4 | -5.1 | -6.5 |

Selisih per provinsi, prompt lagu.

| provinsi | n | base | final | lr4e4 |
|---|---:|---:|---:|---:|
| Aceh | 246 | 62.2 | -5.7 | -3.3 |
| Bali | 241 | 75.9 | -5.4 | -2.5 |
| Jawa Barat | 231 | 66.2 | -3.5 | -4.3 |
| Jawa Tengah | 233 | 56.2 | -3.9 | -1.3 |
| Jawa Timur | 171 | 58.5 | -3.5 | -1.2 |
| Kalimantan Selatan | 233 | 49.8 | -3.4 | -0.9 |
| NTT | 103 | 58.3 | -10.7 | -1.0 |
| Papua | 253 | 64.8 | -2.4 | +5.1 |
| Sulawesi Selatan | 185 | 51.9 | -6.5 | -5.4 |
| Sumatera Barat | 299 | 46.2 | -2.0 | -1.3 |
| Sumatera Utara | 234 | 52.6 | -8.1 | -6.0 |

Selisih per cakupan, prompt lagu.

| cakupan | n | base | final | lr4e4 |
|---|---:|---:|---:|---:|
| khas-provinsi | 1847 | 54.7 | -4.0 | -1.5 |
| umum | 582 | 69.9 | -6.7 | -3.3 |

Selisih per topik, prompt netral.

| topik | n | base | final | lr4e4 |
|---|---:|---:|---:|---:|
| Hari Besar Agama | 183 | 71.6 | -8.7 | -2.7 |
| Hubungan Keluarga | 208 | 55.3 | -6.7 | -4.3 |
| Ikan dan Ternak | 107 | 51.4 | -2.8 | -7.5 |
| Kehamilan, Bayi, Anak | 256 | 61.7 | -4.3 | -5.1 |
| Kematian | 159 | 69.8 | -5.0 | -3.1 |
| Makanan | 369 | 53.9 | -3.5 | -2.2 |
| Permainan | 96 | 53.1 | +0.0 | +3.1 |
| Pernikahan | 320 | 52.5 | -3.8 | -0.3 |
| Pertanian | 98 | 66.3 | -7.1 | -4.1 |
| Sehari-hari | 146 | 55.5 | -0.7 | +3.4 |
| Seni | 272 | 58.8 | -2.6 | +0.0 |
| Socio-religious | 215 | 61.9 | -3.7 | +0.5 |

Selisih per provinsi, prompt netral.

| provinsi | n | base | final | lr4e4 |
|---|---:|---:|---:|---:|
| Aceh | 246 | 62.6 | -2.4 | -1.2 |
| Bali | 241 | 78.0 | -6.6 | -2.9 |
| Jawa Barat | 231 | 63.6 | -2.2 | -0.9 |
| Jawa Tengah | 233 | 55.8 | +0.4 | +2.6 |
| Jawa Timur | 171 | 59.1 | -5.3 | -2.3 |
| Kalimantan Selatan | 233 | 52.4 | -4.3 | -3.9 |
| NTT | 103 | 60.2 | -9.7 | +0.0 |
| Papua | 253 | 64.0 | -2.4 | +3.6 |
| Sulawesi Selatan | 185 | 52.4 | -7.0 | -7.6 |
| Sumatera Barat | 299 | 48.2 | -4.0 | -4.7 |
| Sumatera Utara | 234 | 51.3 | -6.0 | -2.6 |

Selisih per cakupan, prompt netral.

| cakupan | n | base | final | lr4e4 |
|---|---:|---:|---:|---:|
| khas-provinsi | 1847 | 55.2 | -3.6 | -1.5 |
| umum | 582 | 70.1 | -5.7 | -2.7 |

