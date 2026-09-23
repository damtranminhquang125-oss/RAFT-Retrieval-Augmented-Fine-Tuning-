# RAGFT: RAFT Fine-tuning cho Medical QA

** Dataset và Model có thể xem thêm tại: https://drive.google.com/drive/folders/13BQ4yZEb5-RJn47dMURsaLG2xsuE-Rz8?usp=sharing

## 1. Giới thiệu

RAGFT là một thử nghiệm fine-tuning mô hình ngôn ngữ cho bài toán hỏi đáp y khoa có truy xuất tài liệu (Retrieval-Augmented Generation - RAG). Dự án áp dụng ý tưởng RAFT (Retrieval-Augmented Fine-Tuning) để dạy mô hình phân biệt hai tình huống:

- **Grounded**: câu trả lời được hỗ trợ đầy đủ bởi một tài liệu trong prompt và phải trích dẫn bằng chứng nguyên văn.
- **Memory**: các tài liệu được cung cấp không đủ để trả lời, vì vậy mô hình phải sử dụng kiến thức đã học và không được tạo citation giả.

Mô hình được sử dụng là `Qwen/Qwen2.5-3B-Instruct`, fine-tune bằng QLoRA trên Google Colab.

## 2. Kết quả

Kết quả dưới đây là output của lần chạy hiện tại trong `test.ipynb`:

| Tập đánh giá | Token F1 | ROUGE-L | Đúng format RAFT |
|---|---:|---:|---:|
| Toàn bộ test | 48.9% | 46.6% | 68.0% |
| Grounded | 58.1% | 56.3% | 77.9% |
| Memory | 28.5% | 25.3% | 46.2% |

Một số chỉ số bổ sung:

- Citation normalized match trong input context: **70.9%**.
- Citation normalized match với oracle context: **70.3%**.
- Grounded format error: **22.1%**.
- Memory quote error: **7.7%**.

Kết quả cho thấy mô hình xử lý chế độ Grounded tốt hơn chế độ Memory. Tuy nhiên, khả năng tuân thủ format và chất lượng câu trả lời Memory vẫn còn nhiều dư địa để cải thiện.

## 3. Dataset nguồn và dataset đã generate

### Dataset nguồn: MedRAG/textbooks

Notebook sử dụng split `train` của dataset `MedRAG/textbooks` trên Hugging Face. Đây là tập văn bản y khoa đã được chia thành các chunk, dùng làm nguồn tài liệu và oracle context.

Trong code hoặc tài liệu khác, dataset này có thể được gọi nhầm là “MegRAG”. Tên dataset thực tế được sử dụng trong notebook là **`MedRAG/textbooks`**.

Các chunk được chọn tương đối đều giữa các textbook. Mỗi chunk có thể được dùng để sinh nhiều câu hỏi và làm nguồn tạo dữ liệu RAFT.

### Dataset RAFT đã generate

Pipeline sinh dữ liệu tạo câu hỏi, tài liệu gây nhiễu và câu trả lời có format cố định. Dataset hợp lệ của lần chạy hiện tại có:

- **1.734 samples**.
- **1.203 samples Grounded**.
- **531 samples Memory**.
- Mỗi chunk sinh **2 câu hỏi**.
- Mỗi sample sử dụng các distractor documents được chọn bằng BM25.
- Dataset được lưu thành JSONL tại `raft_dataset.zip`.

Mỗi sample thường gồm các trường:

```text
id
source_chunk_id
question
instruction
cot_answer
answer_mode
context
oracle_context
```

Ý nghĩa của từng trường:

| Trường | Kiểu dữ liệu | Ý nghĩa |
|---|---|---|
| `id` | `string` | ID duy nhất của sample. Được dùng để truy xuất sample và lưu prediction tương ứng khi đánh giá. |
| `source_chunk_id` | `int` hoặc `string` | ID của chunk trong dataset nguồn đã dùng để sinh câu hỏi. Trường này được dùng làm group khi chia train/test để tránh data leakage. |
| `question` | `string` | Câu hỏi y khoa được sinh từ source chunk. Đây là nội dung cần mô hình trả lời. |
| `instruction` | `string` | Prompt đầy đủ đưa cho model, gồm các document trong context và câu hỏi. Trường này được dùng làm phần user input khi fine-tuning và inference. |
| `cot_answer` | `string` | Câu trả lời chuẩn (gold answer) do pipeline sinh ra. Câu trả lời tuân theo format RAFT: Grounded có `Evidence:`, Memory có `Reasoning:`, và kết thúc bằng đúng một `<ANSWER>:`. |
| `answer_mode` | `string` | Nhãn của sample, có giá trị `grounded` hoặc `memory`. Nhãn này dùng để stratify khi chia dữ liệu và để tính metric theo từng chế độ. |
| `context` | `dict` | Các document được đưa vào prompt. Thông thường gồm danh sách nội dung document và title tương ứng; các document có thể bao gồm oracle hoặc distractor. |
| `oracle_context` | `string` | Nội dung document nguồn có câu trả lời đầy đủ cho câu hỏi. Trường này dùng để tạo gold answer và kiểm tra citation, không nhất thiết được đưa vào input của Memory sample. |

### Cấu trúc `context`

`context` thường có dạng:

```python
{
        "sentences": [["document 1", "document 2", "..."]],
        "titles": [["title 1", "title 2", "..."]],
}
```

`sentences` chứa nội dung các document được đánh số trong prompt. `titles` chứa tiêu đề tương ứng theo cùng vị trí. Một số sample có thể dùng tên khóa hoặc cấu trúc lồng nhau hơi khác tùy phiên bản checkpoint, nhưng ý nghĩa vẫn là danh sách document và metadata của document.

### Khác nhau giữa hai chế độ

- **Grounded**: `context` chứa oracle document cùng các distractor; `oracle_context` là document dùng làm bằng chứng; `cot_answer` phải có quote hợp lệ từ oracle.
- **Memory**: `context` chỉ chứa distractor hoặc các document không đủ để trả lời; `oracle_context` vẫn được giữ làm metadata/ground truth để đánh giá, nhưng không được đưa vào `instruction`; `cot_answer` không được chứa quote.

Với sample Grounded, oracle document được đưa vào context. Với sample Memory, oracle document bị loại khỏi context để kiểm tra khả năng sử dụng kiến thức nền mà không bịa citation.

## 4. Phương pháp đã áp dụng

### Sinh câu hỏi và câu trả lời

- Dùng Groq API để sinh câu hỏi từ các chunk y khoa.
- Sinh câu trả lời Grounded có phần `Evidence:` và quote từ oracle document.
- Sinh câu trả lời Memory có phần `Reasoning:` nhưng không có quote.
- Kiểm tra tính hợp lệ trước khi ghi sample vào dataset.

### Tạo hard negative bằng BM25

BM25 được dùng để chọn các tài liệu có nội dung gần với câu hỏi nhưng không phải oracle document. Các distractor này làm cho bài toán khó hơn và buộc mô hình phải kiểm tra nội dung thay vì chỉ dựa vào sự xuất hiện của tài liệu.

### Chia train/test theo group

Dữ liệu được chia bằng `StratifiedGroupKFold`:

- `answer_mode` dùng để stratify.
- `source_chunk_id` dùng làm group.
- Các sample cùng nguồn không xuất hiện đồng thời trong train và test.
- `random_state=42` được giữ cố định để có thể tái lập.

Kết quả chia dữ liệu:

```text
Train raw: 1.484 samples
Test:        250 samples
Train final: 1.937 samples
```

Chỉ tập train được oversample theo tỷ lệ:

```text
1x Grounded + 2x Memory
```

### Fine-tuning bằng QLoRA

Base model được load ở 4-bit với NF4. Chỉ các projection của attention được huấn luyện bằng LoRA:

```text
r=16
lora_alpha=32
target_modules=[q_proj, k_proj, v_proj, o_proj]
lora_dropout=0.1
```

Loss chỉ được tính trên phần completion để tập trung vào câu trả lời của assistant.

## 5. Pipeline

```text
MedRAG/textbooks
        |
        v
Chọn và chia chunk nguồn
        |
        v
Groq sinh câu hỏi
        |
        v
Chọn BM25 distractors
        |
        +------------------------------+
        |                              |
        v                              v
Grounded: có oracle              Memory: bỏ oracle
        |                              |
        +--------------+---------------+
                       v
             Kiểm tra format và citation
                       |
                       v
              Checkpoint JSON theo chunk
                       |
                       v
          Split train/test theo source_chunk_id
                       |
                       v
             Oversample Memory trên train
                       |
                       v
             QLoRA fine-tuning Qwen 3B
                       |
                       v
             Load adapter và inference
                       |
                       v
       Đánh giá answer, format và citation
```

## 6. Cấu trúc thư mục

```text
RAGFT/
├── raft.ipynb             # Sinh dataset RAFT và fine-tune model
├── test.ipynb             # Load adapter, inference và đánh giá
├── README.md              # Tài liệu dự án
└── raft_dataset.zip       # Data sau khi generate
```

## 7. Công nghệ sử dụng

- Python và Jupyter Notebook.
- Google Colab và Google Drive.
- Hugging Face `datasets`.
- Hugging Face `transformers`.
- `peft` và `bitsandbytes` cho LoRA/QLoRA.
- `trl` cho supervised fine-tuning.
- Free Groq API để sinh dữ liệu tổng hợp.
- BM25 để chọn hard negative documents.
- `scikit-learn` cho chia dữ liệu.
- `rouge-score`, pandas và các metric tự định nghĩa cho đánh giá.
- Base model: `Qwen/Qwen2.5-3B-Instruct`.
- Free GPU T4 của Google Colab và Kaggle

## 8. Cài đặt

Khuyến nghị sử dụng Google Colab với GPU T4 16 GB.

```bash
pip install -q groq datasets tqdm matplotlib seaborn pandas
pip install -q transformers peft trl bitsandbytes accelerate
pip install -q rouge-score scikit-learn
```

### Cách chạy

1. Mở `raft.ipynb` trên Google Colab.
2. Mount Google Drive và cấu hình secret `GROQ_API_KEY` trong Colab.
3. Chạy các cell sinh dataset. Có thể chạy nhiều lần vì pipeline hỗ trợ checkpoint/resume.
4. Chạy phần fine-tuning để lưu LoRA adapter.
5. Mở `test.ipynb`, tải dataset và adapter, sau đó chạy inference và đánh giá.

Không hard-code Groq API key hoặc Hugging Face token trong notebook. Nếu token từng bị lộ, cần revoke và tạo token mới.

## 9. Nâng cấp trong tương lai

- Tăng kích thước dataset và số lượng textbook được sử dụng.
- Cải thiện prompt sinh câu hỏi để giảm câu hỏi trùng lặp hoặc quá dễ.
- Thử nhiều chiến lược hard negative và retrieval khác nhau ngoài BM25.
- Cân bằng lại dữ liệu Grounded/Memory hoặc thử các tỷ lệ oversampling khác.
- Tăng chất lượng đánh giá bằng human evaluation và bộ câu hỏi y khoa độc lập.
- Thử các base model lớn hơn hoặc các phiên bản Qwen mới hơn.
- Tối ưu system prompt và context length cho inference.
- Theo dõi riêng các lỗi hallucination, citation sai và trả lời thiếu thông tin.
- Lưu prediction, metric và cấu hình training theo version để so sánh các lần chạy.

## 10. Giới hạn

- Dataset sinh bằng LLM có thể chứa câu hỏi, câu trả lời hoặc reasoning chưa chính xác.
- Groq free tier giới hạn số token trong ngày, khiến quá trình sinh dữ liệu phải chia thành nhiều lần chạy.
- Các metric Token F1 và ROUGE-L không phản ánh đầy đủ tính đúng đắn y khoa.
- Citation normalized match chỉ kiểm tra mức độ khớp văn bản, chưa đánh giá sâu chất lượng lập luận.
- Kích thước test set còn nhỏ và được tạo từ cùng nguồn MedRAG, nên khả năng tổng quát sang nguồn dữ liệu khác chưa được đảm bảo.
- QLoRA chỉ cập nhật adapter, không cập nhật toàn bộ trọng số của base model.
- Việc chạy trên GPU T4 với context dài có thể chậm và dễ gặp giới hạn VRAM.
- Mô hình không được dùng để chẩn đoán hoặc đưa ra quyết định y khoa trong thực tế.
