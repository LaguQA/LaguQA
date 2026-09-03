Jalur pilihan ganda, akurasi (%). Dinilai `scripts/22_evaluate_mc.py`.

| Model | Keseluruhan | Fakta | Tebak judul | Lirik | Notasi | Penalaran |
|---|---|---|---|---|---|---|
| **Kontrol (tidak mengenal satu lagu pun)** | | | | | | |
| Tebakan ikut sebaran kunci | 32.1 | 58.4 | 17.8 | 24.0 | 17.6 | 20.3 |
| Tebakan tersering | 21.2 | 20.9 | 15.9 | 24.8 | 19.7 | 21.4 |
| Tebakan acak | 17.2 | 18.5 | 16.8 | 15.7 | 17.1 | 16.9 |
| Tidak menjawab | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **Model tanpa pelatihan** | | | | | | |
| sahabatai-9b | 29.5 ↓ | 25.3 | 27.1 | 25.6 | 17.1 | 47.2 |
| smollm3-3b | 27.1 ↓ | 35.9 | 19.6 | 16.1 | 24.9 | 29.3 |
| qwen-sealion-4b | 25.2 ↓ | 21.7 | 19.6 | 20.7 | 19.2 | 39.3 |
| granite42-8b | 24.2 ↓ | 26.1 | 19.6 | 18.2 | 26.9 | 26.9 |
| qwen35-4b | 24.2 ↓ | 16.8 | 18.7 | 21.1 | 21.8 | 39.7 |
| gemma4-e2b | 24.1 ↓ | 25.5 | 21.5 | 18.2 | 17.1 | 32.8 |
| gemma4-e4b | 24.1 ↓ | 24.7 | 20.6 | 19.0 | 19.2 | 32.1 |
| sealion-e2b | 23.8 ↓ | 22.0 | 19.6 | 18.6 | 15.5 | 37.6 |
| sealion-v35-8b | 23.0 ↓ | 28.5 | 17.8 | 17.8 | 24.4 | 21.4 |
| apertus-sealion-8b | 22.6 ↓ | 24.5 | 22.4 | 12.8 | 21.8 | 29.0 |
| qwen35-9b | 21.8 ↓ | 25.8 | 22.4 | 16.1 | 21.8 | 21.4 |
| ornith-9b | 21.0 ↓ | 18.8 | 20.6 | 17.4 | 21.8 | 26.6 |
| lfm25-2b | 18.8 ↓ | 14.9 | 19.6 | 19.8 | 24.4 | 19.0 |
| **Hasil fine-tuning LaguQA** | | | | | | |
| gemma4-e2b [lr4e4] (seed 1) | 61.0 | 71.2 | 22.4 | 40.1 | 46.1 | 89.7 |
| gemma4-e2b [lr4e4] (seed 3) | 56.7 | 51.1 | 26.2 | 36.4 | 53.9 | 93.8 |
| gemma4-e2b [final] (seed 1) | 56.4 | 63.6 | 27.1 | 30.2 | 50.3 | 84.1 |
| gemma4-e2b [lr4e4] (seed 2) | 52.3 | 55.4 | 25.2 | 35.1 | 40.9 | 80.3 |
| gemma4-e2b [final] (seed 2) | 51.8 | 51.6 | 26.2 | 32.6 | 54.9 | 75.5 |
| gemma4-e2b [final] (seed 3) | 51.7 | 60.9 | 25.2 | 33.5 | 25.4 | 82.8 |
| gemma4-e2b [penuh] (seed 1) | 51.4 | 51.4 | 29.9 | 33.5 | 16.6 | 97.6 |
| gemma4-e2b [penuh-checkpoint-2622] (seed 1) | 50.3 | 51.9 | 26.2 | 32.6 | 15.0 | 95.5 |
| gemma4-e2b [r32] (seed 1) | 50.0 | 52.2 | 24.3 | 35.5 | 19.2 | 89.3 |
| gemma4-e2b [penuh15] (seed 1) | 46.2 | 42.1 | 18.7 | 26.9 | 18.7 | 96.2 |
| gemma4-e2b [penuh-checkpoint-1311] (seed 1) | 46.1 | 46.5 | 24.3 | 28.9 | 17.1 | 87.2 |
| gemma4-e2b [lr1e4] (seed 1) | 46.0 | 47.3 | 26.2 | 25.2 | 17.6 | 87.9 |
| gemma4-e2b [r8] (seed 1) | 45.7 | 45.4 | 29.9 | 27.7 | 15.0 | 87.2 |

↓ menandai skor di bawah batas bawah, yang dipegang tebakan ikut sebaran kunci pada 32,1%. Model bertanda itu tahu lebih sedikit tentang buku ini daripada penebak yang tidak mengenal satu lagu pun.

Batas bawahnya **penebak yang mengikuti sebaran kunci**, bukan tebakan acak. Kunci soal birama 70,2% bernilai 4/4 dan kunci soal nada dasar 70,9% bernilai Do = C, sehingga penebak yang hafal sebaran itu dan nol lagu sudah mendapat 32,1%. Tebakan huruf tersering hanya 21,2% karena opsinya diacak, dan memakainya sebagai batas bawah akan membuat setiap model tampak berpengetahuan.

Dibangun dari laguqa_mc.jsonl sha256 `96b2a22b3d0a08a7` pada 2026-09-03 11:00.
