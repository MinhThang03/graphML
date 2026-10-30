# graphML — Tái lập LGEA và MAKAR (EMNLP 2025)

Mã nguồn tái lập thực nghiệm của nhóm cho môn **Học máy với dữ liệu đồ thị** (HP3-K35).

| Thành viên | MSHV |
|---|---|
| Châu Ngọc Huy | 25C11008 |
| Hoàng Minh Thắng | 25C11020 |

README chỉ mô tả **cách chạy lại** thực nghiệm. Kết quả, phân tích và mô tả phương pháp nằm trong báo cáo: [`report/25C11008-25C11020.pdf`](report/25C11008-25C11020.pdf).

## Hai bài báo

| Thư mục | Bài báo | Mã gốc | Hướng dẫn chạy |
|---|---|---|---|
| [`LGEA/`](LGEA/) | Lu et al., *Breaking the Noise Barrier: LLM-Guided Semantic Filtering and Enhancement for Multi-Modal Entity Alignment*, EMNLP 2025 | [alusang/LGEA-framework](https://github.com/alusang/LGEA-framework) | [`LGEA/README.md`](LGEA/README.md) |
| [`MAKAR/`](MAKAR/) | Lin et al., *MAKAR: a Multi-Agent framework based Knowledge-Augmented Reasoning for Grounded Multimodal Named Entity Recognition*, EMNLP 2025 | [Nikol-coder/MAKAR](https://github.com/Nikol-coder/MAKAR) | [`MAKAR/README.md`](MAKAR/README.md) |

## Cấu trúc

```
graphML/
├── README.md
├── report/
│   └── 25C11008-25C11020.pdf   báo cáo của nhóm
├── LGEA/
│   ├── README.md          môi trường, cấu hình, cách chạy, khắc phục sự cố
│   ├── notebooks/         3 notebook Kaggle, lưu kèm output của lần chạy
│   └── paper/             PDF bài báo gốc
├── MAKAR/
│   ├── README.md
│   ├── notebooks/         notebook Kaggle chạy toàn bộ pipeline
│   ├── logs/              log kernel Kaggle của lần chạy GMNER
│   └── paper/
└── tools/
    ├── wandb_import_lgea.py    đưa log từ output notebook LGEA lên W&B
    ├── wandb_import_makar.py   đưa log từ output notebook / log kernel MAKAR lên W&B
    └── requirements.txt
```

## Phần cứng chung

Mọi thực nghiệm chạy trên **Kaggle Notebooks, GPU T4 ×2 (15 GB mỗi card), bật Internet**. Mỗi notebook
tự cài thư viện, tự tải dữ liệu và mô hình; không cần chuẩn bị gì trên máy cá nhân. Chi tiết phiên bản
thư viện và thời lượng từng phần nằm trong README của từng thư mục.

## Nhật ký thực nghiệm (Weights & Biases)

| Bài báo | Project công khai |
|---|---|
| LGEA | https://forge.coreweave.com/wandb/chaungochuy0709-university-of-science-vietnam-national-/graphml-lgea |
| MAKAR | https://forge.coreweave.com/wandb/chaungochuy0709-university-of-science-vietnam-national-/graphml-makar |

Các run trên W&B được **nhập lại từ output của notebook** sau khi chạy trên Kaggle, không chạy lại mô
hình. Mỗi run ghi tên notebook nguồn trong phần ghi chú và cấu hình. Để tạo lại:

```bash
python3 -m venv ~/venvs/graphml
~/venvs/graphml/bin/pip install -r tools/requirements.txt
~/venvs/graphml/bin/wandb login
~/venvs/graphml/bin/python tools/wandb_import_lgea.py --dry-run    # xem các run đọc được
~/venvs/graphml/bin/python tools/wandb_import_lgea.py              # đẩy lên W&B
~/venvs/graphml/bin/python tools/wandb_import_makar.py
```

Đổi `ENTITY` và `PROJECT` ở đầu mỗi script nếu đẩy sang tài khoản khác.
