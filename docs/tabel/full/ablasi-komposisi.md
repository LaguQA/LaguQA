Regime `full`, akurasi toleran (%) per kategori. Kolom Selisih membandingkan kolom terakhir dengan kolom pertama.

| Kategori | Soal | gemma4-e2b (benih 1) | gemma4-e2b [seimbang] (benih 1) | Selisih | Catatan |
|---|---|---|---|---|---|
| **KESELURUHAN** | 1002 | 53.5 | 71.8 | +18.3 (+183 soal) |  |
| judul_ke_lirik | 50 | 2.0 | 98.0 | +96.0 (+48 soal) |  |
| notasi_ke_judul | 50 | 82.0 | 6.0 | -76.0 (-38 soal) |  |
| tempo | 50 | 26.0 | 98.0 | +72.0 (+36 soal) |  |
| pencipta | 50 | 40.0 | 100.0 | +60.0 (+30 soal) |  |
| jenis | 50 | 52.0 | 100.0 | +48.0 (+24 soal) |  |
| lanjut_lirik | 50 | 12.0 | 48.0 | +36.0 (+18 soal) |  |
| birama | 50 | 72.0 | 100.0 | +28.0 (+14 soal) |  |
| judul_ke_baris | 50 | 0.0 | 26.0 | +26.0 (+13 soal) |  |
| asal | 50 | 78.0 | 100.0 | +22.0 (+11 soal) |  |
| nada_dasar | 50 | 78.0 | 100.0 | +22.0 (+11 soal) |  |
| asal_abstain | 50 | 82.0 | 100.0 | +18.0 (+9 soal) |  |
| verifikasi_asal | 50 | 68.0 | 80.0 | +12.0 (+6 soal) | kecil |
| verifikasi_pencipta | 50 | 60.0 | 54.0 | -6.0 (-3 soal) | kecil |
| judul_ke_notasi | 50 | 8.0 | 12.0 | +4.0 (+2 soal) | kecil |
| tempo_abstain | 2 | 0.0 | 100.0 | +100.0 (+2 soal) | kecil |
| verifikasi_nada_dasar | 50 | 72.0 | 76.0 | +4.0 (+2 soal) | kecil |
| lirik_ke_judul | 50 | 22.0 | 20.0 | -2.0 (-1 soal) | kecil |
| nada_tertinggi | 50 | 100.0 | 98.0 | -2.0 (-1 soal) | kecil |
| pencipta_abstain | 50 | 98.0 | 100.0 | +2.0 (+1 soal) | kecil |
| rumpang | 50 | 20.0 | 18.0 | -2.0 (-1 soal) | kecil |
| hitung_bar | 50 | 100.0 | 100.0 | 0.0 (0 soal) | tetap |

**kecil** menandai selisih di bawah 7 soal. Pada kategori berisi 50 soal, satu soal bernilai 2 poin persen, sehingga selisih sekecil itu tidak dapat dibedakan dari kebetulan dengan satu benih saja. Tabel ini bukan uji signifikansi; untuk itu diperlukan beberapa benih, bukan perhitungan tambahan atas satu benih.

Berkas yang dibandingkan:
- gemma4-e2b (benih 1) — `gemma4-e2b-full-s1--full.jsonl`
- gemma4-e2b [seimbang] (benih 1) — `gemma4-e2b-full-s1-seimbang--full.jsonl`
