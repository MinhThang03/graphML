# Tái lập LGEA (EMNLP 2025)

Mã nguồn tái lập thực nghiệm cho bài báo *Breaking the Noise Barrier: LLM-Guided Semantic
Filtering and Enhancement for Multi-Modal Entity Alignment*
([paper](https://aclanthology.org/2025.emnlp-main.1684.pdf) ·
[mã gốc](https://github.com/alusang/LGEA-framework)).

Kho này chỉ mô tả **cách chạy lại**. Kết quả và phân tích nằm trong báo cáo.

## 0. Kho này gồm gì

Mã gốc của tác giả chỉ có phần mô hình — **không kèm dữ liệu và không kèm sản phẩm tiền xử
lý nào**. Kho này bổ sung những thứ còn thiếu để chạy được từ đầu:

- hai script dựng lại tệp thuộc tính (`attribute1.txt`, `attribute2.txt`, `ent2id`) từ dữ
  liệu thô, cho cả DBP15K lẫn FBDB15K/FBYG15K;
- hai notebook Kaggle chạy một lượt từ tải dữ liệu đến in kết quả, kèm các bản vá cần thiết
  để mã chạy được trên thư viện hiện hành và trên 2 GPU 15 GB;
- script đóng gói dữ liệu, trích số liệu và đẩy nhật ký lên W&B.

**Phạm vi chạy lại được.** Mô-đun lọc ảnh bằng LLM cần tệp mặt nạ, mà tệp này không được
phát hành; sinh lại nó đòi hỏi **tập ảnh gốc ~50 GB** chỉ có trên Baidu Cloud. Khi không có
đường dẫn mặt nạ, mã gốc trả về vector toàn 1 — tức notebook chạy ở cấu hình `w/o mask`.
Nhánh thị giác vẫn hoạt động bình thường vì nó dùng đặc trưng ảnh đã trích sẵn, không đọc
file ảnh.

---

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

## 3. Dựng lại tệp thuộc tính (notebook tự gọi — chỉ cần khi chạy ngoài Kaggle)

```bash
# DBP15K
python scripts/build_attributes.py --raw <thư-mục-DBP15K_raw> \
    --ent-ids-dir data/DBP15K/zh_en --out data/DBP15K/zh_en

# FBDB15K / FBYG15K
python scripts/build_attributes_fb.py --mmkb ./mmkb --dataset FBDB15K \
    --out data/mmkg/FBDB15K/norm
```

Hai script này có chốt chặn, cố ý dừng hẳn thay vì ghi ra tệp sai:

- `build_attributes.py` lấy id từ **chính các tệp mà file `.pkl` đặc trưng ảnh được đánh chỉ
  số theo**, và dừng nếu hai nguồn id bất đồng.
- `build_attributes_fb.py` tự nhận ra đồ thị nào thuộc phía nào dựa trên **nội dung** tệp
  `ent_ids`, và yêu cầu độ phủ phép nối ≥30%. Dưới ngưỡng này, script in ra mẫu khóa của cả
  hai bên rồi dừng — nếu không, định dạng khóa lệch nhau sẽ cho ra tệp rỗng một cách âm thầm.

## 4. Chạy

### 4.1 Bảng 1 — DBP15K

Mở `notebooks/LGEA_reproduce_kaggle.ipynb`, chỉnh cell cấu hình:

```python
SPLITS = ['zh_en']        # hoặc ['zh_en', 'ja_en', 'fr_en']
USE_IL = True             # huấn luyện lặp, theo mô tả trong bài báo
```

Chạy bằng *Save Version → Save & Run All (Commit)*.

### 4.2 Bảng 2 và Bảng 3 — FBDB15K / FBYG15K

Mở `notebooks/LGEA_reproduce_fb.ipynb`:

```python
DATASETS = ['FBDB15K']    # thêm 'FBYG15K' nếu cần
USE_IL   = True
RUNS = ['LGEA', 'w/o img', 'w/o rel', 'w/o structure', 'w/o att', 'w/o llm', 'w/o it']
```

> Chạy đủ 7 cấu hình trên cả hai bộ dữ liệu tốn khoảng 23 giờ GPU, vượt giới hạn 12 giờ của
> một phiên Kaggle. Mỗi cặp `(bộ dữ liệu, cấu hình)` có đánh dấu hoàn thành riêng nên có thể
> chia thành nhiều phiên.

## 5. Tiết kiệm thời gian khi chạy lại

Bước nhúng thuộc tính bằng LLM chiếm phần lớn thời gian và chỉ phụ thuộc vào bộ dữ liệu, không
phụ thuộc cấu hình cắt bỏ. Lưu kết quả của một phiên thành Kaggle Dataset rồi trỏ tới nó:

```python
EMB_INPUT = '/kaggle/input/<slug-dataset>'
```

Notebook đối chiếu số thực thể trong tệp với bộ dữ liệu đang chạy trước khi dùng, nên không
thể nạp nhầm embedding của bộ khác.

Tương tự, đặt `BUNDLE` để bỏ qua bước tải 1,4 GB (xem `scripts/make_kaggle_bundle.py` để
dựng sẵn dataset này), và dùng `CKPT_INPUT` + `ONLY_TEST = True` để nạp lại trọng số đã
huấn luyện mà không huấn luyện lại.

## 6. Ghi nhận thực nghiệm lên Weights & Biases

```bash
pip install wandb && wandb login
python scripts/extract_results.py          # đọc notebook đã chạy -> results.json
python scripts/wandb_replay.py --entity <entity> --project lgea-reproduction
```

Sau đó đặt project ở chế độ công khai: *Settings → Privacy → Open to the public*.

## 7. Cấu trúc kho mã

```
notebooks/
  LGEA_reproduce_kaggle.ipynb     Bảng 1 (DBP15K)
  LGEA_reproduce_fb.ipynb         Bảng 2 và Bảng 3 (FBDB15K / FBYG15K)
scripts/
  make_notebook.py                sinh notebook DBP15K
  make_notebook_fb.py             sinh notebook FB
  build_attributes.py             dựng tệp thuộc tính cho DBP15K
  build_attributes_fb.py          dựng tệp thuộc tính cho FBDB15K / FBYG15K
  make_kaggle_bundle.py           đóng gói dữ liệu để upload thành Kaggle Dataset
  extract_results.py              trích số liệu từ notebook đã chạy
  wandb_replay.py                 đẩy nhật ký huấn luyện lên W&B
report/
  main.tex                        mã nguồn báo cáo
```

Hai notebook được **sinh ra** bởi các script trong `scripts/`. Muốn sửa nội dung notebook,
sửa script rồi chạy lại nó thay vì sửa trực tiếp tệp `.ipynb`.

## 8. Nguồn dữ liệu

| Thành phần | Nguồn |
|---|---|
| DBP15K / FBDB15K / FBYG15K (văn bản, đặc trưng ảnh) | gói dữ liệu của [MEAformer](https://github.com/zjukg/MEAformer) |
| Giá trị thuộc tính DBP15K | [HackCz/DBP15K_raw](https://huggingface.co/datasets/HackCz/DBP15K_raw) |
| Giá trị thuộc tính FB15K / DB15K / YAGO15K | [mmkb](https://github.com/mniepert/mmkb) |
| Ảnh gốc (cho mô-đun lọc thị giác) | chỉ phát hành qua Baidu Cloud — **không sử dụng được** |

## 9. Khắc phục sự cố

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

## 10. Ghi chú về tính tái lập

- Hai notebook được **sinh ra** bởi script trong `scripts/`, không chỉnh tay. Mọi thay đổi
  nội dung notebook đều phải sửa script rồi chạy lại.
- Mọi con số trong báo cáo đều do `extract_results.py` trích thẳng từ notebook đã chạy, rồi
  `make_report.py` thay vào bản mẫu LaTeX bằng token — không con số nào được chép tay.
- Mỗi cặp `(bộ dữ liệu, cấu hình)` có đánh dấu hoàn thành riêng, nên một lần chạy bị cắt giữa
  chừng có thể tiếp tục mà không tính lại từ đầu.
