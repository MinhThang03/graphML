# Tổng kết công việc — Tái lập LGEA (EMNLP 2025)

**Nhóm:** Hoàng Minh Thắng (25C11020) · Châu Ngọc Huy (25C11008)
**Môn học:** Học máy với dữ liệu đồ thị — HP3-K35
**Bài báo:** C. Lu et al., *Breaking the Noise Barrier: LLM-Guided Semantic Filtering and
Enhancement for Multi-Modal Entity Alignment*, EMNLP 2025, pp. 33153–33167.

> Tài liệu nội bộ của nhóm — tổng hợp toàn bộ công việc và kết quả.
> **Không đưa file này lên GitHub**: quy định môn học yêu cầu README của kho mã chỉ mô tả
> cách chạy lại, không chứa kết quả. Nội dung ở đây thuộc về báo cáo PDF.

---

## 1. Phạm vi đã đặt ra

| Mục tiêu | Trạng thái |
|---|---|
| Bảng 1 — DBP15K ZH-EN | ✅ chạy xong |
| Bảng 1 — DBP15K JA-EN / FR-EN | ⚪ có hạ tầng, không chạy (quota GPU) |
| Bảng 2 — FBDB15K | ⚠️ chạy được dòng đối chứng `w/o mask`, không chạy được dòng có mặt nạ |
| Bảng 2 — FBYG15K | ⚪ có hạ tầng, không chạy (quota GPU) |
| Bảng 3 — ablation 7 cấu hình trên FBDB15K | ✅ chạy đủ cả 7 |
| Mô-đun lọc ảnh bằng LLM (đóng góp ①) | ❌ không tái lập được — thiếu tập ảnh gốc |

---

## 2. Vấn đề lớn nhất: kho mã gốc không chạy được như công bố

Kho [alusang/LGEA-framework](https://github.com/alusang/LGEA-framework) **chỉ có mã, không
có dữ liệu và không có sản phẩm tiền xử lý nào**. Cụ thể là thiếu:

- tệp thuộc tính `attribute1.txt` / `attribute2.txt` / `ent2id` cho mọi bộ dữ liệu
- tệp mặt nạ thị giác (đầu ra của BLIP + GPT-3.5)
- tệp caption trung gian

Nhóm phải dựng lại toàn bộ đường ống dữ liệu từ **ba nguồn độc lập**:

| Thành phần | Nguồn |
|---|---|
| Cấu trúc đồ thị, đặc trưng ảnh đã trích sẵn | gói dữ liệu của MEAformer |
| Giá trị thuộc tính DBP15K | `HackCz/DBP15K_raw` (N-Triples thô) |
| Giá trị thuộc tính FB15K / DB15K / YAGO15K | kho `mniepert/mmkb` |
| Ảnh gốc (~50 GB) | chỉ phát hành qua Baidu Cloud — **không tiếp cận được** |

### Công cụ nhóm tự viết cho khâu này

**`build_attributes.py`** — dựng tệp thuộc tính DBP15K từ N-Triples thô.
Điểm quan trọng: tham số `--ent-ids-dir` buộc lấy id từ **chính các tệp MEAformer** mà file
`.pkl` đặc trưng ảnh được đánh chỉ số theo. Có chốt chặn dừng hẳn nếu hai nguồn id bất đồng.
Kết quả: **38.960 dòng, 0 lỗi khi đánh giá**.

**`build_attributes_fb.py`** — tương tự cho FBDB15K/FBYG15K, nhưng khó hơn vì ba định dạng
khác nhau: FB15K dùng tab với giá trị trần, DB15K và YAGO15K dùng N-Triples. Hàm `detect_kg`
tự nhận ra đồ thị nào thuộc phía nào **dựa trên nội dung** chứ không giả định theo vị trí, và
bắt buộc độ phủ phép nối ≥30% — nếu khóa không khớp định dạng thì dừng kèm chẩn đoán, thay vì
âm thầm ghi ra tệp rỗng và cho ra một lần chạy vô nghĩa.

---

## 3. Sáu chỗ phải vá để mã 10 tháng tuổi chạy được

| # | Triệu chứng | Nguyên nhân thật | Cách xử lý |
|---|---|---|---|
| 1 | `ModuleNotFoundError: transformers.modeling_layers` | Kho McGill viết lại `modeling_llama_encoder.py` ngày 2025-12-02, đòi `transformers≥4.56.2`; nhưng `llm2vec` ghim `≤4.44.2`. Bản cũ lại cần `LlamaFlashAttention2`, đã bị xóa ở 4.48 → **4.48–4.51 là vùng chết** | Ghim `CODE_REV='1af5f3140c42'` và truyền `code_revision=` (không phải `revision=`, vì `config.auto_map` trỏ chéo kho) |
| 2 | `AttributeError: Can't get local object 'add_hook_to_module.<locals>.new_forward'` | `llm2vec.py:357` tách tiến trình theo từng GPU rồi pickle mô hình — giả định dữ liệu song song, không dùng được với mô hình đã chia tầng | Viết `encode_sharded()` riêng; đã kiểm chứng tính bất biến sắp xếp/khôi phục trên 4 ca, kể cả ca có giá trị bằng nhau và ca n=1 |
| 3 | `mat2 is on cuda:0, different from other tensors on cuda:1` | Adapter PEFT thứ hai nạp sau khi `merge_and_unload()` đã gỡ hook | Căn lại bằng chính hàm nội bộ của peft. Kết quả trên Kaggle: *"realigned 224 adapter layers, 0 still mismatched"* |
| 4 | `found at least two devices` tại `residual + hidden_states` | `accelerate` khớp `_no_split_modules=["LlamaDecoderLayer"]` **theo tên lớp**, nhưng LLM2Vec dựng `ModifiedLlamaDecoderLayer` → một tầng bị cắt đôi giữa hai card | Tự viết `device_map` tường minh. Cân bằng đo được: cuda:0 = 7,59 GB · cuda:1 = 7,42 GB |
| 5 | `OutOfMemoryError: Tried to allocate 5.66 GiB` tại `layers.py:26` | Đạo hàm nhân ma trận thưa dựng **nguyên ma trận dày N×N** rồi chỉ lấy ra số phần tử bằng số cạnh | Viết lại theo từng cạnh: **O(N²) → O(E·d)**, bộ nhớ còn 0,08 GiB. Đã chứng minh hai công thức tương đương bằng Python thuần **trước khi** áp dụng |
| 6 | Kết quả in ra `n/a` ở Hits@1 / Hits@10 | Bộ phân tích log tìm chuỗi `hits@1`, nhưng log thật ghi `acc of top [1, 10, 50] = [...]` | Viết lại biểu thức chính quy, đọc ngược từ `'Test result'` |

**Không chỗ nào trong sáu chỗ này làm thay đổi mô hình.** Năm chỗ là tương thích thư viện và
phân bổ GPU; chỗ thứ sáu là bộ nhớ, và đã được kiểm chứng cho ra số trùng khớp.

---

## 4. Thực nghiệm đã chạy

| Hạng mục | Chi tiết |
|---|---|
| Nền tảng | Kaggle Notebooks, GPU **T4 ×2** (15 GB/card) |
| So với bài báo | Bài báo dùng **A100 40 GB** |
| Thời gian | ≈2,1 h cho DBP15K ZH-EN + ≈11,5 h cho bảy cấu hình FBDB15K ⇒ **≈14 giờ GPU** |
| Ràng buộc | 12 h mỗi phiên · 30 h mỗi tuần · 20 GB `/kaggle/working` · T4 không hỗ trợ bf16 |
| Số lần chạy ghi nhận | 8 lần chạy, 3.263 điểm dữ liệu theo vòng |

---

## 5. Kết quả

### 5.1 DBP15K ZH-EN (30% cặp hạt giống)

| | Hits@1 | Hits@10 | MRR |
|---|---|---|---|
| Bài báo | 0.970 | 0.998 | 0.982 |
| **Nhóm** | **0.9578** | **0.9974** | **0.975** |
| Chênh lệch | −1.2 điểm | −0.1 | −0.7 |

Đạt được **dù thiếu hoàn toàn mô-đun lọc ảnh**. Với một bài tái lập, mức này coi như tái lập
được. Đường huấn luyện hội tụ trong khoảng **50 vòng đầu** rồi đi ngang.

### 5.2 FBDB15K — dòng đối chứng (20% cặp hạt giống)

| Cấu hình | Bài báo H@1 | Nhóm H@1 |
|---|---|---|
| LGEA đầy đủ (có mặt nạ) | 0.801 | *không tái lập được* |
| **`w/o mask`** | **0.714** | **0.7322 (+1.8 điểm)** |

**Vì sao `w/o mask` là dòng hợp lệ:** đọc mã nguồn ([`model/NewEA.py:268`](https://github.com/alusang/LGEA-framework/blob/main/model/NewEA.py)),
nếu không cung cấp đường dẫn mặt nạ thì hàm trả về **vector toàn 1** — đúng bằng định nghĩa
`w/o mask` mà chính bài báo báo cáo ở Bảng 3. Nhóm không chạy một cấu hình sai, mà chạy một
cấu hình **có đối chứng trực tiếp trong bài báo**.

### 5.3 Ablation đầy đủ — FBDB15K, bảy cấu hình

| Cấu hình | Bài báo H@1 | Nhóm H@1 | Nhóm H@10 | Nhóm MRR | Sụt vs nhóm | Nhóm, chưa bật lặp |
|---|---|---|---|---|---|---|
| **LGEA (= w/o mask)** | 0.714 | **0.7322** | 0.8798 | 0.786 | — | 0.5919 |
| w/o structure | 0.440 | 0.4030 | 0.5764 | 0.466 | **−32.9** | 0.3478 |
| w/o att | 0.655 | 0.5402 | 0.7646 | 0.621 | −19.2 | 0.3688 |
| w/o llm | 0.682 | 0.5570 | 0.7834 | 0.637 | −17.5 | 0.3797 |
| w/o it | 0.684 | 0.5987 | 0.8192 | 0.676 | −13.3 | — |
| w/o rel | 0.737 | 0.6688 | 0.8455 | 0.732 | −6.3 | 0.5203 |
| w/o img | 0.740 | 0.7147 | 0.8640 | 0.767 | −1.7 | 0.6277 |

**Thứ tự tầm quan trọng trùng khớp với bài báo ở cả 6/6 vị trí:**
`structure > att > llm > it > rel > img`

Mức sụt của bài báo (đo so với LGEA đầy đủ của họ, 0.801): cấu trúc −36.1 · thuộc tính −14.6 ·
LLM −11.9 · lặp −11.7 · quan hệ −6.4 · ảnh −6.1. Mức sụt của nhóm đo so với 0.7322. **Mỗi bên
so với cấu hình đầy đủ của chính mình** — nên thứ tự so được, giá trị tuyệt đối thì không.

**Một phép kiểm chứng chéo đáng tin:** cột cuối là hiệu năng tại vòng 500, ngay trước khi
huấn luyện lặp bật. Giá trị của LGEA ở cột đó là **0.5919**, còn lần chạy `w/o it` độc lập
(tắt hẳn huấn luyện lặp) cho **0.5987**. Hai cách đo khác nhau, chênh 0,7 điểm — hai con số
xác nhận lẫn nhau.

---

## 6. Ba phát hiện của nhóm

### ① Huấn luyện lặp đáng 13.3 điểm nhưng chỉ được nhắc một dòng

0.7322 so với 0.5987 — nhiều hơn ảnh (1.7) và quan hệ (6.3) **cộng lại**. Bước nhảy rõ rệt
ở vòng 500 trùng đúng với `il_start = 500`. Trong bài báo, cơ chế này chỉ xuất hiện ở §5.1
phần thiết lập thực nghiệm, không nằm trong Figure 1, không có trong mô tả phương pháp.

### ② Tuyên bố "không dùng surface form" không được bảo đảm

Bài báo khẳng định ở **ba chỗ** rằng tên thực thể không được sử dụng. Nhóm đã tuân thủ: loại
tên khỏi phần định danh mỗi bộ ba. Nhưng khi đo gióng hàng **chỉ bằng vector thuộc tính**,
không huấn luyện, không cấu trúc, không ảnh → **Hits@1 = 0.872** trên DBP15K ZH-EN, gần bằng
LoginMEA (0.873), phương pháp mạnh nhất trước đó trong bảng so sánh của chính bài báo.

Nguyên nhân: trong DBP15K, **tên thực thể tồn tại dưới dạng một thuộc tính** — các bộ ba có
vị từ `name` mang giá trị đúng bằng tên hiển thị. Loại tên khỏi *định danh* không loại được
tên khỏi *giá trị*.

Nhóm **không** cáo buộc tác giả cố ý. Luận điểm là: loại bỏ định danh thực thể không tự động
bảo đảm "không dùng surface form" — còn phụ thuộc bộ dữ liệu có chứa tên ở nơi khác hay không.
(Các bộ FB có thuộc tính thuần số nên là phép thử sạch hơn.)

### ③ Mốc 0.714 nhạy hơn bài báo trình bày

Bài báo quy **8.7 điểm** cho bộ lọc ảnh (0.714 → 0.801). Baseline **không hề có bộ lọc** của
nhóm đã đạt 0.732 — vượt chính cái mốc đó. Nếu trần 0.801 giữ nguyên, phần công cho bộ lọc
chỉ còn ≈**6.9 điểm**.

*Giới hạn phải nói kèm:* đây là so giữa hai lần hiện thực khác nhau, nên **không bác bỏ**
đóng góp của bài báo. Nó chỉ cho thấy con số 8.7 nhạy với chất lượng baseline.

### Quan sát bổ sung từ mã nguồn lọc ảnh

Dù không chạy được mô-đun, nhóm đã đọc `gene_mask.py` và thấy hai điều bài báo không nêu:

- BLIP sinh **3 caption** mỗi ảnh, nhưng khi đem so GPT-3.5 chỉ nhận **1 caption bốc ngẫu
  nhiên** (`random.choice`). Mặt nạ vì vậy **không tất định** — chạy lại có thể ra kết quả
  khác, và bài báo không báo cáo độ dao động này.
- Prompt yêu cầu GPT tóm tắt chủ đề chung và kết quả được ghi vào trường `context`, nhưng
  hàm đọc mặt nạ chỉ lấy trường `mask`. **Trường `context` không bao giờ đi vào mô hình.**

---

## 7. Những gì KHÔNG làm được, và vì sao

| Hạng mục | Lý do |
|---|---|
| Mô-đun lọc ảnh (đóng góp ①) | Tác giả không phát hành tệp mặt nạ. Muốn tự sinh phải chạy BLIP trên **ảnh gốc** — BLIP cần pixel, không dùng được vector đặc trưng đã nén. Ảnh gốc 50 GB chỉ có trên Baidu Cloud |
| DBP15K JA-EN / FR-EN | Quota GPU. Hạ tầng đã sẵn sàng, chỉ cần đổi `SPLITS` |
| FBYG15K | Quota GPU. Nhóm ưu tiên chạy **đủ bảy dòng trên một bộ** thay vì một dòng trên hai bộ |

**Lưu ý quan trọng:** nhánh thị giác **vẫn hoạt động bình thường** — mô hình đọc vector ResNet
2048 chiều đã trích sẵn, không đọc file ảnh. Bằng chứng: `w/o img` (0.7147) *thấp hơn* cấu
hình đầy đủ (0.7322). Thứ thiếu chỉ là **công tắc bật/tắt từng ảnh**, không phải bản thân ảnh.

---

## 8. Sản phẩm bàn giao

| Tệp | Vai trò |
|---|---|
| `build_attributes.py` · `build_attributes_fb.py` | Dựng lại tệp thuộc tính từ nguồn thô |
| `make_notebook.py` → `LGEA_reproduce_kaggle.ipynb` | Notebook Bảng 1 (21 cell) |
| `make_notebook_fb.py` → `LGEA_reproduce_fb.ipynb` | Notebook Bảng 2 & 3 (20 cell) |
| `make_kaggle_bundle.py` | Đóng gói ~830 MB dữ liệu để upload thành Kaggle Dataset |
| `extract_results.py` → `results.json` | Trích **mọi con số được trích dẫn** thẳng từ notebook đã chạy |
| `make_report.py` + `template_main.tex` → `latex/main.tex` | Sinh báo cáo bằng thay thế token — **không con số nào chép tay** |
| `lint_tex.py` | Kiểm tra cấu trúc LaTeX (không có pdflatex ở máy) |
| `wandb_replay.py` | Phát lại nhật ký thật lên W&B — 8 lần chạy, 3.263 điểm |
| `script_thuc_nghiem.md` | Script nói cho slide 16–22 |
| `REPO_README.md` | README cho GitHub (chỉ cách chạy, theo quy định) |

Hai notebook được **sinh ra** bởi script, không sửa trực tiếp — nên mọi thay đổi đều tái lập được.

---

## 9. Việc còn lại

- [ ] Điền khối `\author` trong `latex/main.tex` (tên + MSHV)
- [ ] Châu Ngọc Huy viết phần bài báo thứ hai (đã có khung dàn ý trong `main.tex`)
- [ ] Tạo kho GitHub, thay `TODO-USERNAME/TODO-REPO` trong `main.tex`
- [ ] Chạy `wandb_replay.py`, đặt project công khai, thay `TODO-ENTITY/TODO-PROJECT`
- [ ] Biên dịch `main.tex` trên Overleaf — **chưa từng biên dịch**, máy không có LaTeX.
      Nếu `\usepackage[T5]{fontenc}` lỗi thì chuyển sang XeLaTeX + fontspec (ghi chú có sẵn trong file)
- [ ] Xuất PDF đúng tên `25C11020-25C11008.pdf`, kiểm tra <20 MB và ≤25 trang
