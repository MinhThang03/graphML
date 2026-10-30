# LGEA — hướng dẫn chạy lại

Bài báo: C. Lu et al., *Breaking the Noise Barrier: LLM-Guided Semantic Filtering and Enhancement for
Multi-Modal Entity Alignment*, EMNLP 2025 ([PDF](paper/2025.emnlp-main.1684.pdf),
[ACL Anthology](https://aclanthology.org/2025.emnlp-main.1684/)). Mã gốc:
[alusang/LGEA-framework](https://github.com/alusang/LGEA-framework).

Thư mục này chỉ mô tả **cách chạy lại**. Kết quả và phân tích nằm trong báo cáo.

## 1. Phần cứng và môi trường

| | |
|---|---|
| Nền tảng | Kaggle Notebooks |
| Accelerator | **GPU T4 ×2** — bắt buộc. Llama-3-8B ở fp16 cần ~16 GB nên một card 15 GB không đủ; P100 và T4 đều không hỗ trợ bf16 |
| Internet | **ON** (tải trọng số mô hình và dữ liệu) |
| Thời lượng | ~2,1 h mỗi split DBP15K; ~2,2 h mỗi cấu hình FBDB15K khi bật huấn luyện lặp |
| Python | 3.11 (ảnh Kaggle mặc định) |

Phiên bản thư viện được ghim trong notebook và **không được đổi**:

```
transformers==4.44.2    peft==0.12.0    llm2vec==0.2.3
```

`llm2vec` yêu cầu `transformers>=4.43.1,<=4.44.2`. Ràng buộc này quyết định giá trị
`CODE_REV` trong notebook; đổi một trong hai sẽ làm hỏng bước nhúng văn bản.

## 2. Chuẩn bị trước khi chạy

1. **Trọng số Llama-3-8B** là mô hình bị giới hạn truy cập. Chọn một trong hai:
   - Chấp nhận điều khoản tại
     [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct),
     rồi thêm token vào Kaggle: *Add-ons → Secrets*, tên `HF_TOKEN`.
   - Hoặc đổi `BASE_MODEL` trong cell cấu hình sang
     `NousResearch/Meta-Llama-3-8B-Instruct` (bản sao không giới hạn, cùng trọng số).
2. Không cần tải dữ liệu thủ công — notebook tự tải.

## 3. Các notebook

| Notebook | Phần thực nghiệm | Cấu hình đã chạy |
|---|---|---|
| `notebooks/reproduce-lgea-emnlp-2025-dbp15k-zh-en.ipynb` | Bảng 1, DBP15K ZH-EN | `SPLITS = ['zh_en']` |
| `notebooks/reproduce-lgea-emnlp-2025-dbp15k-fr-en.ipynb` | Bảng 1, DBP15K JA-EN và FR-EN | `SPLITS = ['ja_en', 'fr_en']` |
| `notebooks/reproduce-lgea-table-2-table-3-fbdb15k-fbyg.ipynb` | Bảng 2 và Bảng 3, FBDB15K | `DATASETS = ['FBDB15K']`, bảy cấu hình trong `RUNS` |

Hai notebook DBP15K có cùng mã, chỉ khác danh sách `SPLITS`. Các notebook được lưu kèm output của
lần chạy trên Kaggle để làm bằng chứng; mở trên Kaggle sẽ chạy lại từ đầu.

Biến cấu hình chính ở cell đầu mỗi notebook:

| Biến | Ý nghĩa |
|---|---|
| `SPLITS` / `DATASETS` | bộ dữ liệu cần chạy |
| `RUNS` | (FBDB15K) các cấu hình của Bảng 3; `'LGEA'` là dòng Bảng 2 |
| `DATA_RATE`, `EPOCHS`, `SEED`, `USE_IL` | tỉ lệ cặp hạt giống, số epoch, seed, bật huấn luyện lặp |
| `BASE_MODEL`, `CODE_REV` | trọng số Llama-3-8B và revision mã LLM2Vec đã ghim |
| `BUNDLE` | đường dẫn Kaggle Dataset chứa dữ liệu dựng sẵn; `None` thì notebook tự tải và tự dựng tệp thuộc tính |
| `EMB_INPUT` | (FBDB15K) dùng lại embedding thuộc tính của phiên trước, bỏ qua bước LLM |
| `FORCE_DATA`, `FORCE_EMBED`, `FORCE_TRAIN` | (FBDB15K) buộc làm lại một bước dù đã có đánh dấu hoàn thành |

## 4. Chạy trên Kaggle

1. Tải notebook lên Kaggle (*Create → Import Notebook*).
2. *Settings*: Accelerator **GPU T4 ×2**, Internet **On**.
3. Thêm secret `HF_TOKEN` (*Add-ons → Secrets*) nếu dùng `meta-llama/Meta-Llama-3-8B-Instruct`.
4. Chỉnh cell cấu hình nếu cần, rồi *Save Version → Save & Run All (Commit)*.
5. Lấy output trong tab *Output* của version đã chạy.

Notebook tự tải dữ liệu, tự dựng ba tệp mà mã gốc cần nhưng không phát hành (`attribute1.txt`,
`attribute2.txt`, `ent2id`), tự vá mã cho T4 và tự in bảng so sánh ở cell cuối.

Notebook FBDB15K đánh dấu hoàn thành cho từng cặp `(bộ dữ liệu, cấu hình)`. Nếu phiên 12 giờ bị cắt,
gắn Output của version trước làm Input rồi chạy lại: các cấu hình đã xong được bỏ qua.

**Phạm vi chạy lại được.** Mô-đun lọc ảnh bằng LLM cần tệp mặt nạ mà tác giả không phát hành; sinh
lại nó cần tập ảnh gốc khoảng 50 GB chỉ có trên Baidu Cloud. Khi không có đường dẫn mặt nạ, mã gốc
dùng vector toàn 1, nên mọi lần chạy đều ở cấu hình `w/o mask`.

## 5. Nguồn dữ liệu

| Thành phần | Nguồn |
|---|---|
| DBP15K / FBDB15K / FBYG15K (văn bản, đặc trưng ảnh) | gói dữ liệu của [MEAformer](https://github.com/zjukg/MEAformer) |
| Giá trị thuộc tính DBP15K | [HackCz/DBP15K_raw](https://huggingface.co/datasets/HackCz/DBP15K_raw) |
| Giá trị thuộc tính FB15K / DB15K / YAGO15K | [mmkb](https://github.com/mniepert/mmkb) |
| Ảnh gốc (cho mô-đun lọc thị giác) | chỉ phát hành qua Baidu Cloud — **không sử dụng được** |

## 6. Nhật ký W&B

Project công khai: [https://forge.coreweave.com/wandb/chaungochuy0709-university-of-science-vietnam-national-/graphml-lgea](https://forge.coreweave.com/wandb/chaungochuy0709-university-of-science-vietnam-national-/graphml-lgea).
Các run được nhập từ output của các notebook trên bằng script `../tools/wandb_import_lgea.py`
(xem `../README.md`), không chạy lại mô hình.

## 7. Khắc phục sự cố

Sáu chỗ dưới đây đã được vá sẵn trong notebook, liệt kê theo thông báo lỗi để tra nhanh nếu chạy
ngoài Kaggle.

Mã gốc công bố tháng 12/2025 không còn chạy được với thư viện hiện hành, và giả định một GPU
lớn. Sáu chỗ dưới đây đã được vá sẵn trong notebook — liệt kê theo **thông báo lỗi** để tra
nhanh nếu bạn chạy ngoài Kaggle.

**`ModuleNotFoundError: No module named 'transformers.modeling_layers'`**
Kho LLM2Vec của McGill đã viết lại `modeling_llama_encoder.py` và nay đòi `transformers≥4.56.2`,
trong khi `llm2vec` ghim `≤4.44.2`. Nâng cấp không giải quyết được: bản mã cũ cần
`LlamaFlashAttention2`, đã bị xóa ở 4.48 — tức **4.48–4.51 là vùng không chạy được**.
Cách xử lý: ghim `CODE_REV = '1af5f3140c42'` và truyền qua tham số **`code_revision=`**, không
phải `revision=` — vì `config.auto_map` là một tham chiếu trỏ chéo sang kho khác.

**`AttributeError: Can't get local object 'add_hook_to_module.<locals>.new_forward'`**
`llm2vec.py:357` tách tiến trình theo từng GPU rồi pickle mô hình — một giả định dữ liệu song
song, không dùng được khi mô hình đã chia tầng qua nhiều card. Notebook thay bằng hàm
`encode_sharded()` riêng.

**`RuntimeError: ... mat2 is on cuda:0, different from other tensors on cuda:1`**
Adapter PEFT thứ hai được nạp sau khi `merge_and_unload()` đã gỡ hook phân bổ thiết bị.
Notebook căn lại bằng chính hàm nội bộ của peft trước khi mã hóa.

**`RuntimeError: ... found at least two devices` tại `residual + hidden_states`**
`accelerate` khớp `_no_split_modules=["LlamaDecoderLayer"]` **theo tên lớp**, nhưng LLM2Vec
dựng lớp tên `ModifiedLlamaDecoderLayer` — nên một tầng bị cắt đôi giữa hai card. Notebook tự
dựng `device_map` tường minh thay vì để `accelerate` tự chia.

**`torch.OutOfMemoryError: Tried to allocate 5.66 GiB` tại `layers.py:26`**
Đạo hàm của phép nhân ma trận thưa dựng nguyên một ma trận dày `N×N` rồi chỉ lấy ra số phần
tử bằng số cạnh. Trên A100 40 GB không ai để ý; trên T4 15 GB thì hỏng. Notebook viết lại theo
từng cạnh — **O(N²) → O(E·d)**, bộ nhớ còn 0,08 GiB. Đây là **sửa cách dùng bộ nhớ, không đổi
mô hình**: hai công thức đã được kiểm chứng cho ra kết quả trùng khớp trước khi áp dụng.
Notebook cũng đặt `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**Cell kết quả in ra `n/a` ở Hits@1 và Hits@10**
Bộ phân tích log cũ tìm chuỗi `hits@1`, nhưng log thật ghi dạng
`acc of top [1, 10, 50] = [...], mr = ..., mrr = ...`. Đã sửa biểu thức chính quy.
