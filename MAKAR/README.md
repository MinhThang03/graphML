# Tái lập kết quả paper MAKAR (EMNLP 2025) trên Kaggle

> **MAKAR: a Multi-Agent framework based Knowledge-Augmented Reasoning for Grounded Multimodal Named Entity Recognition**
> Lin et al., EMNLP 2025, tr. 6110–6130 · [Paper](2025.emnlp-main.311.pdf) · Code gốc: <https://github.com/Nikol-coder/MAKAR>

Dự án này tái lập các kết quả chính của MAKAR trên Kaggle với 2 GPU T4 miễn phí, trong khi paper dùng 8×A800. Phần nào chạy được thì được làm lại thật. Phần không thể train lại trên Kaggle (SFT + GRPO cho Qwen2.5-VL) thì dùng checkpoint chính thức của tác giả.

## 1. Bài toán và phương pháp

**GMNER:** cho một tweet gồm văn bản và ảnh, trích xuất các bộ ba *(thực thể, loại, bounding box)*. Box là `None` nếu thực thể không xuất hiện trong ảnh. **FMNERG** là cùng bài toán nhưng với 51 loại thực thể mịn thay vì 4 loại.

MAKAR gồm 3 agent:

| Agent | Vai trò |
|---|---|
| **KEA** – Knowledge Enhancement | MLLM sinh *auxiliary knowledge* cho mỗi tweet, ghép vào câu, rồi XLM-RoBERTa + CRF gán nhãn thực thể |
| **ECA** – Entity Correction | Với thực thể có độ tin cậy < τ = 0.5: tìm web, tóm tắt, rồi để MLLM sửa biên và loại thực thể |
| **ERGA** – Entity Reasoning Grounding | Qwen2.5-VL suy luận `<think>…</think><answer>[x1,y1,x2,y2]</answer>`, huấn luyện SFT (3k mẫu dễ) rồi GRPO (3k mẫu khó) |

Chỉ số đánh giá (F1):
- **MNER:** đúng biên và đúng loại thực thể.
- **EEG:** đúng biên và đúng vùng ảnh (IoU > 0.5).
- **MNERG:** đúng cả bộ ba.

## 2. Các file

| File | Nội dung |
|---|---|
| `makar_reproduce_kaggle.ipynb` | Notebook sạch (chưa có output), dùng để upload lên Kaggle |
| `makar-reproduce-kaggle.ipynb` | Notebook đã chạy trên Kaggle (lần chạy **FMNERG** đầy đủ, có output) |
| `results/`, `results.zip` | Output của lần chạy **GMNER** đầu tiên: kết quả NER, ECA, 776/2879 truy vấn grounding |
| `2025.emnlp-main.311.pdf` | Paper gốc |

## 3. Notebook làm những gì

1. **Dữ liệu:**
   - Dữ liệu NER kèm auxiliary knowledge lấy từ repo MAKAR.
   - Nhãn vàng và bounding box lấy từ [NUSTM/GMNER](https://github.com/NUSTM/GMNER) và [NUSTM/FMNERG](https://github.com/NUSTM/FMNERG).
   - Ảnh Twitter-2015/2017 (1 GB) tải qua Google Drive.
2. **KEA:** train lại từ đầu XLM-RoBERTa-large + CRF, dùng siêu tham số theo config AdaSeq của tác giả. Song song train thêm bản không có auxiliary knowledge (ablation *w/o KEA*), mỗi bản một GPU.
3. **ECA:**
   - Lấy độ tin cậy của mỗi thực thể từ xác suất biên (marginal) của CRF.
   - Tìm web bằng DuckDuckGo, vì Bing Search API mà paper dùng đã ngừng.
   - Dùng Qwen2.5-VL-3B-Instruct để tóm tắt và sửa thực thể.
4. **ERGA:** dùng checkpoint chính thức `soliton110/MAKAR-3B` (ModelScope), với prompt giống hệt dữ liệu train của tác giả.
5. **Baseline Raw (Table 5):** Qwen2.5-VL-3B-Instruct chưa fine-tune, chạy cùng prompt trên 300 câu ngẫu nhiên.
6. **Minh họa GRPO:** sinh 5 câu trả lời cho mỗi input, tính reward (format + IoU) và advantage chuẩn hóa theo nhóm (Eq. 9–12).
7. **Case study:** vẽ box dự đoán và box vàng kèm đoạn suy luận `<think>` (giống Fig. 6).

Notebook có ngân sách thời gian 11 giờ, vì Kaggle dừng cứng sau 12 giờ. Nếu chưa xong, chỉ cần gắn Output của version trước làm Input rồi chạy lại, notebook sẽ tự chạy tiếp.

## 4. Kết quả

### Table 2 – Thống kê dữ liệu: khớp chính xác với paper

| Tập test | Entity | Grounded | Box |
|---|---|---|---|
| GMNER (paper / tái lập) | 2543 / 2543 | 1036 / 1036 | 1244 / 1244 |
| FMNERG (paper / tái lập) | 2543 / 2543 | 1046 / 1046 | 1254 / 1254 |

### FMNERG: chạy đầy đủ (MAKAR-3B, NER 8 epoch, khoảng 4 giờ trên 2×T4)

| Hệ thống | MNERG | MNER | EEG |
|---|---|---|---|
| **MAKAR tái lập** (KEA + ECA + ERGA) | **63.75** | **77.22** | **73.57** |
| w/o ECA | 63.51 | 76.93 | 73.57 |
| w/o KEA | 56.00 | 67.59 | 71.94 |
| *Paper – MAKAR-7B (Table 3)* | *60.54* | *71.24* | *75.66* |
| *Paper – w/o KEA / w/o ECA (Table 4)* | *57.50 / 58.57* | – | – |

Table 5, trên cùng tập con 300 câu:

| Hệ thống | EEG tái lập | EEG paper (FMNERG) |
|---|---|---|
| Qwen2.5-VL-3B-Instruct (Raw) | 23.54 | 38.38 |
| MAKAR-3B (SFT + GRPO) | 74.98 | 72.43 |

Phần grounding có hai điểm rõ ràng:
- **Huấn luyện SFT + GRPO tạo khác biệt lớn:** EEG tăng từ 23.5 lên 75.0. Model gốc thường không trả lời đúng định dạng.
- **EEG của MAKAR-3B tái lập được gần sát paper:** 73.57–74.98, so với 72.43 cho bản 3B và 75.66 cho bản 7B trong paper.

### GMNER: lần chạy đầu, chưa xong grounding

| Hệ thống | MNER F1 (tái lập) | Paper |
|---|---|---|
| KEA + ECA | 83.94 | 86.38 |
| KEA (w/o ECA) | 83.89 | – |
| w/o KEA | 82.54 | – |

Grounding chỉ xong 776/2879 truy vấn do lỗi tốc độ (xem mục 5). Để có MNERG/EEG cho GMNER, chạy lại notebook với `DATASET = "GMNER"` và gắn `results/` làm Input.

## 5. Các phát hiện khi tái lập

1. **Rò nhãn vàng trong auxiliary knowledge của FMNERG.** Trong file auxiliary knowledge tác giả công bố cho **tập test FMNERG**, **821/1500 mẫu (55%)** có những câu như *"identified as a park based on the context provided by the ground truth entities"*. Như vậy, nhiều khả năng MLLM đã được xem nhãn vàng khi sinh phần kiến thức bổ trợ này. Tập dev FMNERG và cả hai tập dev/test GMNER không có dấu hiệu này (0%). Hệ quả thấy được trong kết quả:
   - Trên FMNERG, F1 của KEA ở tập test (76.16) cao hơn ở tập dev (70.85).
   - Kết quả tái lập vượt paper ở MNER (77.22 so với 71.24) và MNERG (63.75 so với 60.54).
   - Khoảng cách giữa KEA và w/o KEA phình lên 9,6 điểm MNER.

   Vì vậy **không nên coi MNER/MNERG trên FMNERG là kết quả sạch.** Số EEG ít bị ảnh hưởng hơn, vì vùng ảnh do ERGA dự đoán, không dùng auxiliary knowledge.
2. **Rò rỉ giữa train và test của GMNER.** File train GMNER của tác giả có 28.070 câu (9.335 câu khác nhau): đó là 7.000 câu của GMNER, trộn thêm Twitter-2015/2017 và nhân bản với nhiều phiên bản auxiliary knowledge. Trong đó **43 câu trùng tập test**, và có lẫn nhãn `MISC` của Twitter-2015. Notebook mặc định chỉ dùng đúng 7.000 câu train của GMNER.
3. **Checkpoint MAKAR lưu kèm `"use_cache": false`** (còn sót lại từ lúc train). Khi chạy bằng HuggingFace `generate`, model phải tính lại cả chuỗi (kể cả ảnh) cho mỗi token mới, nên chậm khoảng 20 lần: ~80 giây cho một truy vấn. Sau khi bật lại cache, 3151 truy vấn chỉ mất 73 phút.
4. **ECA ít tác dụng với ngưỡng τ = 0.5.** CRF rất tự tin nên chỉ 11 thực thể (GMNER) và 91 thực thể (FMNERG) có độ tin cậy dưới ngưỡng. ECA chỉ cải thiện +0,05 (GMNER) và +0,29 (FMNERG) MNER. Dù vậy, nhóm tin cậy thấp thực sự hay sai: precision 19,8% so với 78,6% của nhóm tin cậy cao trên FMNERG, đúng với giả thuyết của paper.
5. **Minh họa GRPO:** phần lớn nhóm 5 câu trả lời có reward giống nhau, nên advantage A = 0 và không có tín hiệu để học. Đây chính là lý do paper dành GRPO cho các mẫu **khó** (Table 6).

## 6. Khác biệt so với paper

| Thành phần | Paper | Tái lập |
|---|---|---|
| Phần cứng | 8×A800 | 2×T4 (Kaggle) |
| NER | 25 epoch, train mở rộng (28k câu với GMNER) | 8 epoch, train 7k câu |
| Search engine của ECA | Bing + ReaderLM-v2 | DuckDuckGo (title + snippet) |
| MLLM của ECA | Qwen2.5-VL-7B | Qwen2.5-VL-3B (7B không vừa một T4) |
| ERGA | Train SFT + GRPO, kết quả chính dùng 7B | Checkpoint chính thức MAKAR-3B, không train lại |
| Suy luận | bf16 | fp16 (T4 không hỗ trợ bf16), dùng transformers vì vLLM không cài được trên Python 3.13 của Kaggle |

## 7. Cách chạy lại

1. Upload `makar_reproduce_kaggle.ipynb` lên Kaggle, chọn *Accelerator = GPU T4 ×2* và *Internet = On*.
2. Đặt `DATASET = "GMNER"` hoặc `"FMNERG"` trong cell CẤU HÌNH.
3. Bấm *Save Version → Save & Run All*. Mỗi dataset mất khoảng 4–5 giờ.
4. Nếu notebook báo "⏭ Bỏ qua: hết ngân sách thời gian": vào *Add Input → Notebook Output*, chọn version vừa chạy, rồi Run All lần nữa.
5. Kết quả nằm trong `/kaggle/working/`:
   - `results_<DATASET>.json`;
   - `comparison.csv` (bảng so sánh với paper);
   - thư mục `<DATASET>/` (log và output trung gian).
