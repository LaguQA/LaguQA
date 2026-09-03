Regime `full`, akurasi toleran (%). Dinilai `scripts/11_evaluate.py`.

| Model | Keseluruhan | Fakta | Abstain | Verifikasi | Lirik | Notasi | Penalaran |
|---|---|---|---|---|---|---|---|
| **Kontrol (tidak mengenal satu lagu pun)** | | | | | | | |
| Tebakan tersering | 33.6 | 38.3 | 100.0 | 49.3 | 2.8 | 4.0 | 35.0 |
| Tebakan acak | 28.6 | 25.7 | 100.0 | 52.7 | 1.6 | 4.0 | 21.0 |
| Tidak menjawab | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **Model tanpa pelatihan** | | | | | | | |
| gemma4-e2b | 20.0 ↓ | 14.0 | 36.3 | 48.7 | 0.4 | 0.0 | 47.0 |
| **Hasil fine-tuning LaguQA** | | | | | | | |
| gemma4-e2b (benih 1) | 53.5 | 57.7 | 88.2 | 66.7 | 11.2 | 45.0 | 100.0 |

↓ menandai skor di bawah lantai tebakan tersering (33.6%): model itu tahu lebih sedikit tentang buku ini daripada penebak yang tidak mengenal satu lagu pun.

Kolom **Abstain** harus dibaca berpasangan dengan **Fakta**, tidak sendirian. Soal abstain menguji apakah model menolak mengarang ketika bukunya memang tidak mencantumkan apa-apa, sehingga model yang menolak menjawab segalanya mendapat 100% di sana — dan kedua kontrol memang begitu. Angka abstain yang rendah berarti model mengarang; angka abstain tinggi berarti sesuatu hanya jika kolom Fakta juga tinggi.
