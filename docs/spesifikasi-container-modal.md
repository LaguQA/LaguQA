# Spesifikasi container Modal, hasil pengukuran

Keluaran `modal_train.py::mesin` pada profil `akun1` (workspace `vhpwdar`),
3 September 2026. Dijalankan dengan `LAGUQA_GPU=L4` sehingga container yang
diperiksa sama jenisnya dengan yang dipakai melatih.

```json
{
  "gpu": "NVIDIA L4",
  "gpu_total_gb": 22.0,
  "gpu_capability": "8.9",
  "gpu_multiprocessors": 58,
  "cpu_model": "unknown",
  "cpu_terlihat": 17,
  "cpu_terpakai": 17,
  "cpu_quota_cgroup": "",
  "ram_total_gb": 377.3,
  "ram_tersedia_gb": 374.3,
  "ram_batas_cgroup": "",
  "kartu_per_wadah": 1,
  "driver_cuda": "13.0",
  "versions": {
    "torch": "2.13.0",
    "transformers": "5.16.1",
    "peft": "0.20.0",
    "trl": "1.12.0",
    "datasets": "5.0.1",
    "accelerate": "1.14.0"
  }
}
```

## Cara membacanya

**Angka prosesor dan memori adalah batas atas, bukan jatah.** `cpu_quota_cgroup`
dan `ram_batas_cgroup` kosong, artinya tidak ada batas cgroup yang terpasang
pada container. Tidak ada satu pun `@app.function` di `modal_train.py` yang
memakai argumen `cpu=` atau `memory=`, jadi container berjalan pada bawaan
Modal. Nilai 17 inti dan 377,3 GB itu milik mesin induk yang terlihat lewat
`/proc`, bukan sumber daya yang dijaminkan.

`cpu_terpakai` diambil dari `os.sched_getaffinity(0)` dan bernilai sama dengan
`cpu_terlihat`, jadi topeng afinitasnya memang membolehkan seluruh 17 inti
logis. Yang tidak diketahui adalah berapa yang benar-benar tersedia saat mesin
induknya ramai.

**`cpu_model` kosong.** `/proc/cpuinfo` di container tidak memuat baris
`model name`, sehingga jenis prosesornya tidak dapat dilaporkan. Nama prosesor
karena itu tidak ditulis di naskah, bukan ditebak.

**Memori kartu 22,0 GiB, bukan 24 GB.** Angka 24 GB adalah kapasitas nominal
L4 menurut pabrikannya; `torch.cuda.get_device_properties` melaporkan 22,0 GiB
yang benar-benar dapat dialokasikan. Puncak pemakaian saat pelatihan 12,5 GB,
jadi masih lapang.

**Versi pustaka cocok dengan Tabel 3.7.** Keenam nomor versi yang dilaporkan
container sama persis dengan yang tertulis di naskah, sehingga tabel itu
tervalidasi terhadap keadaan yang sungguh terpasang.

Log mentahnya di `logs/mesin.log`.
