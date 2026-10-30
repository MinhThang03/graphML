# MAKAR — hướng dẫn chạy lại

Bài báo: X. Lin et al., *MAKAR: a Multi-Agent framework based Knowledge-Augmented Reasoning for Grounded
Multimodal Named Entity Recognition*, EMNLP 2025 ([PDF](paper/2025.emnlp-main.311.pdf),
[ACL Anthology](https://aclanthology.org/2025.emnlp-main.311/)). Mã gốc:
[Nikol-coder/MAKAR](https://github.com/Nikol-coder/MAKAR).

Thư mục này chỉ mô tả **cách chạy lại**. Kết quả và phân tích nằm trong báo cáo.

## 1. Phần cứng và môi trường

| | |
|---|---|
| Nền tảng | Kaggle Notebooks |
| Accelerator | **GPU T4 ×2** (mỗi card 15 GB, không hỗ trợ bf16 nên suy luận chạy fp16) |
| Internet | **On** (tải mô hình, dữ liệu, ảnh và tìm kiếm web cho ECA) |
| Thời lượng | khoảng 3–5 giờ cho một bộ dữ liệu; notebook tự dừng gọn trước giới hạn 12 giờ |
| Python | ảnh Kaggle mặc định (3.13 lúc chạy); dùng torch và transformers có sẵn, cài thêm `modelscope`, `gdown`, `ddgs` |

vLLM không cài được trên ảnh Kaggle lúc chạy, nên `BACKEND = "auto"` rơi về transformers và chia
đều truy vấn cho hai GPU.

## 2. Notebook

`notebooks/makar-reproduce-kaggle.ipynb` chạy toàn bộ pipeline cho một bộ dữ liệu. Bản trong repo
được lưu kèm output của lần chạy **FMNERG** trên Kaggle để làm bằng chứng.

Biến cấu hình chính ở cell đầu:

| Biến | Ý nghĩa |
|---|---|
| `DATASET` | `"GMNER"` (4 loại thô) hoặc `"FMNERG"` (51 loại mịn) |
| `QUICK_TEST` | `True` để chạy thử nhanh (1 epoch, tập con) trước lần chạy đầy đủ |
| `TIME_BUDGET_H`, `NER_MAX_H`, `ECA_MAX_H`, `RAW_MAX_H` | trần thời gian cho cả phiên và từng bước |
| `NER_ENCODER`, `NER_EPOCHS`, `NER_LR` | bộ mã hóa XLM-RoBERTa-large và siêu tham số huấn luyện NER |
| `GMNER_TRAIN_MODE` | `"gmner7000"`: đúng 7 000 câu train; `"full"`: toàn bộ tệp train của tác giả |
| `RUN_NO_KEA` | huấn luyện thêm bản không có tri thức bổ trợ trên GPU thứ hai |
| `RUN_ECA`, `ECA_MODEL`, `TAU`, `ECA_NUM_PAGES` | bật bước sửa thực thể, MLLM dùng để sửa, ngưỡng tin cậy, số trang web |
| `MAKAR_CKPT` | checkpoint ERGA của tác giả trên ModelScope (`soliton110/MAKAR-3B` hoặc `-7B`) |
| `RAW_BASELINE`, `RAW_MAX_SENT` | mốc Qwen2.5-VL chưa huấn luyện và cỡ tập con dùng để so |

Notebook dùng checkpoint ERGA đã huấn luyện của tác giả; bước SFT và GRPO không chạy lại được trên
hai GPU T4.

## 3. Chạy trên Kaggle

1. Tải notebook lên Kaggle (*Create → Import Notebook*).
2. *Settings*: Accelerator **GPU T4 ×2**, Internet **On**.
3. Đặt `DATASET` trong cell cấu hình.
4. *Save Version → Save & Run All (Commit)*.
5. Nếu notebook in "⏭ Bỏ qua: hết ngân sách thời gian": vào *Add Input → Notebook Output*, chọn
   version vừa chạy, rồi *Save & Run All* lần nữa. Notebook tự chạy tiếp từ bước dang dở.

Kết quả nằm trong `/kaggle/working/`:

- `results_<DATASET>.json`: các độ đo của từng hệ thống;
- `comparison.csv`: bảng đối chiếu với số của bài báo;
- thư mục `<DATASET>/`: log huấn luyện và output trung gian của từng bước.

## 4. Nguồn dữ liệu

| Thành phần | Nguồn |
|---|---|
| Dữ liệu NER kèm tri thức bổ trợ | kho [Nikol-coder/MAKAR](https://github.com/Nikol-coder/MAKAR) |
| Nhãn vàng và hộp bao | [NUSTM/GMNER](https://github.com/NUSTM/GMNER), [NUSTM/FMNERG](https://github.com/NUSTM/FMNERG) |
| Ảnh Twitter-2015/2017 (1 GB) | Google Drive, notebook tự tải bằng `gdown` |
| Checkpoint ERGA | ModelScope `soliton110/MAKAR-3B` |
| MLLM cho ECA và mốc Raw | Hugging Face `Qwen/Qwen2.5-VL-3B-Instruct` |

## 5. Nhật ký W&B

Project công khai: [https://forge.coreweave.com/wandb/chaungochuy0709-university-of-science-vietnam-national-/graphml-makar](https://forge.coreweave.com/wandb/chaungochuy0709-university-of-science-vietnam-national-/graphml-makar).
Các run được nhập bằng script `../tools/wandb_import_makar.py` (xem `../README.md`), không chạy lại mô hình:
FMNERG từ output của notebook, GMNER từ log kernel `logs/makar-gmner-reproduce.log`
(cùng notebook, `DATASET = "GMNER"`).

## 6. Khắc phục sự cố

**Định vị chạy rất chậm (khoảng 80 giây một truy vấn).** Checkpoint MAKAR lưu kèm
`"use_cache": false`, nên `generate` tính lại toàn bộ chuỗi cho mỗi token mới. Notebook đã đặt lại
`model.config.use_cache = True` khi nạp mô hình; nếu chạy ngoài notebook, cần làm tương tự.

**Đầu ra MLLM lỗi (chuỗi "!!!!" hoặc sai định dạng).** Đặt `MLLM_DTYPE = "float32"`; chỉ vừa bộ nhớ với
mô hình 3B.

**Tìm kiếm web trả về rỗng hoặc lỗi kết nối.** Bing Search API đã ngừng nên ECA dùng DuckDuckGo
(`ddgs`); vài truy vấn có thể không có kết quả hoặc gặp lỗi kết nối tạm thời. Đây là giới hạn của nguồn tìm kiếm, notebook vẫn chạy tiếp các bước sau.
