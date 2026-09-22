# RAGFT: RAFT Fine-tuning for Grounded and Memory QA

Dự án xây dựng dữ liệu RAFT từ MedRAG textbooks, fine-tune `Qwen/Qwen2.5-3B-Instruct` bằng QLoRA, sau đó đánh giá khả năng phân biệt hai chế độ trả lời:

- `grounded`: câu trả lời có bằng chứng trích nguyên văn từ một document trong prompt.
- `memory`: document không hỗ trợ đầy đủ câu trả lời, model trả lời bằng kiến thức đã học và không được tạo citation.

## Thành phần

- `raft.ipynb`: tải dữ liệu MedRAG, sinh câu hỏi/đáp án bằng Groq, tạo distractor documents và lưu checkpoint.
- `test.ipynb`: chia dữ liệu, oversample `memory`, fine-tune LoRA, tải adapter và đánh giá model.
- `raft_dataset.zip`: dữ liệu RAFT đã tạo, nếu được đóng gói từ checkpoint.
- `raft_qwen25_3b_lora.zip`: LoRA adapter đã fine-tune.

## Yêu cầu môi trường

Khuyến nghị chạy trên Google Colab có GPU T4 16 GB.

Các thư viện chính:

```bash
pip install -q groq datasets tqdm matplotlib seaborn pandas
pip install -q transformers peft trl bitsandbytes accelerate datasets
pip install -q rouge-score scikit-learn
```

Model sử dụng:

```text
Qwen/Qwen2.5-3B-Instruct
```

## 1. Sinh dataset bằng `raft.ipynb`

Notebook `raft.ipynb` thực hiện các bước:

1. Mount Google Drive.
2. Tải `MedRAG/textbooks`.
3. Chọn các chunk nguồn theo từng textbook.
4. Sinh câu hỏi từ chunk bằng Groq.
5. Chọn distractor documents bằng BM25.
6. Với mỗi câu hỏi, tạo một trong hai chế độ:
   - Có oracle document: `grounded`, bắt buộc quote.
   - Không có oracle document: `memory`, không được quote.
7. Kiểm tra label trước khi ghi.
8. Lưu từng chunk vào checkpoint JSON để có thể resume.

Các checkpoint được lưu tại:

```text
/content/drive/MyDrive/raft_project/raft_checkpoint_chunks
```

Mỗi file checkpoint có dạng:

```text
chunk_000994.json
```

và chứa `source_chunk_id`, `sample_count` và danh sách `samples`.

### Lưu ý về API key

Không hard-code Groq API key hoặc Hugging Face token trong notebook. Hãy dùng Colab Secrets:

```python
from google.colab import userdata

GROQ_API_KEY = userdata.get("GROQ_API_KEY")
```

Nếu token đã từng xuất hiện trong notebook hoặc bị commit, hãy revoke token đó và tạo token mới.

## 2. Chia dữ liệu trong `test.ipynb`

Dữ liệu được chia theo `source_chunk_id`, không chia ngẫu nhiên từng sample. Điều này bảo đảm các sample cùng chunk không bị đưa sang cả train và test, giúp giảm leakage.

Split sử dụng:

```python
StratifiedGroupKFold(
    n_splits=7,
    shuffle=True,
    random_state=42,
)
```

`answer_mode` được dùng để stratify, còn `source_chunk_id` được dùng làm group.

Các kiểm tra bắt buộc:

```python
assert train_chunk_ids.isdisjoint(test_chunk_ids)
assert len(train_samples_raw) + len(test_samples) == len(raft_samples)
```

### Oversampling memory

Oversampling chỉ được thực hiện trên train, không thực hiện trên test:

```text
Train: 1x grounded + 2x memory
Test: giữ nguyên phân bố tự nhiên
```

Các bản sao memory được deep-copy và gán ID mới để tránh trùng ID:

```text
<original_id>_dup
```

Test set phải được giữ nguyên để phản ánh đúng phân bố thực tế và tránh đánh giá quá lạc quan.

## 3. Fine-tune bằng QLoRA

Model được load ở 4-bit với NF4, sau đó chỉ train LoRA trên các projection của attention:

```python
load_in_4bit=True
bnb_4bit_quant_type="nf4"
bnb_4bit_compute_dtype=torch.float16
bnb_4bit_use_double_quant=True
```

LoRA hiện tại:

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.10,
    bias="none",
    task_type="CAUSAL_LM",
)
```

Cấu hình training khuyến nghị:

```python
SFTConfig(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    weight_decay=0.05,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    max_length=2048,
    completion_only_loss=True,
    fp16=True,
)
```

Nếu GPU không đủ VRAM với `max_length=2048`, dùng:

```python
per_device_train_batch_size=2
gradient_accumulation_steps=4
```

Effective batch size vẫn tương đương.

`max_length=2048` quan trọng vì system prompt, nhiều document và completion có thể dài. Cấu hình `max_length=1200` trước đây làm khoảng 98.6% sample bị truncation, khiến model dễ bỏ sót `Evidence:`, `Reasoning:` hoặc `<ANSWER>:`.

## 4. Lưu và tải adapter

Adapter được lưu tại:

```text
/content/drive/MyDrive/raft_project/raft_qwen25_3b_lora
```

Nếu adapter được đóng gói thành ZIP:

```text
/content/drive/MyDrive/raft_project/raft_qwen25_3b_lora.zip
```

Cần giải nén trước khi gọi `PeftModel.from_pretrained()`. Thư mục adapter sau khi giải nén phải chứa tối thiểu:

```text
adapter_config.json
adapter_model.safetensors
tokenizer_config.json
tokenizer.json
```

Tokenizer nên được tải từ thư mục adapter để giữ `chat_template.jinja` đã lưu:

```python
tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_DIR,
    local_files_only=True,
)
```

Sau đó load base model và ghép adapter:

```python
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_DIR,
    local_files_only=True,
)
```

Không gọi `prepare_model_for_kbit_training()` khi chỉ tải model để inference.

## 5. Format dữ liệu hội thoại

Mỗi sample được chuyển thành conversational dataset:

```python
def format_sample(sample):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_RAFT},
            {"role": "user", "content": sample["instruction"]},
        ],
        "completion": [
            {"role": "assistant", "content": sample["cot_answer"]},
        ],
    }
```

`SYSTEM_RAFT` quy định:

- Grounded bắt đầu bằng `Evidence:`.
- Grounded phải có quote nằm trong document.
- Memory bắt đầu bằng `Reasoning:`.
- Memory không được có `Evidence:` hoặc quote.
- Mọi output có đúng một `<ANSWER>:` ở dòng cuối.

Loss chỉ nên tính trên completion:

```python
completion_only_loss=True
```

Không cần bật đồng thời `assistant_only_loss=True` và `completion_only_loss=True`.

## 6. Đánh giá

Notebook đánh giá các nhóm metric sau:

### Answer quality

- `Token F1`
- `ROUGE-L`

Đây là metric phụ vì cùng một đáp án y khoa có thể được diễn đạt bằng nhiều cách.

### Format compliance

- Đúng một `<ANSWER>:` trong toàn bộ output.
- `<ANSWER>:` nằm ở dòng cuối.
- Grounded bắt đầu bằng `Evidence:`.
- Memory bắt đầu bằng `Reasoning:`.
- Grounded có quote hợp lệ.
- Memory không có quote.

### Citation

- Citation normalized match trong input context.
- Citation normalized match với oracle context.
- `citation_compliance` chỉ áp dụng cho grounded; memory hiển thị `N/A`.

Metric citation cho phép khác biệt whitespace và một số biến thể Unicode dash/quotation mark để tránh phạt oan do OCR. Vì vậy đây là normalized match, không phải exact character match tuyệt đối.

### Metric quan trọng nhất

Lỗi chính cần theo dõi:

```text
Memory quote error
```

Đây là tỷ lệ sample gold là `memory` nhưng model vẫn sinh quote. Ngoài ra theo dõi:

```text
Grounded format error
Full RAFT format compliance
Citation compliance
```

Chạy đánh giá:

```python
rows = [
    evaluate_raft_prediction(
        sample,
        predictions[sample["id"]],
    )
    for sample in all_test_samples
]

metrics_df = pd.DataFrame(rows)
print_raft_summary(metrics_df)
```

## 7. Đọc kết quả training

Chọn checkpoint dựa trên validation loss và metric RAFT held-out. `load_best_model_at_end=True` chỉ chọn theo `eval_loss`, không đảm bảo checkpoint tốt nhất về `memory_quote_error` hoặc format compliance.

Vì vậy cần kiểm tra riêng:

```text
memory_quote_error
memory full_format_compliance
grounded citation_compliance
grounded_format_error
```

Nếu train loss tiếp tục giảm nhưng validation loss tăng, đó là dấu hiệu overfitting. Không nên tăng epoch chỉ dựa trên train loss; hãy xem metric theo từng mode.

## 8. Lưu ý inference

Khi đánh giá, prompt phải dùng đúng `SYSTEM_RAFT` đã dùng lúc train:

```python
messages = [
    {"role": "system", "content": SYSTEM_RAFT},
    {"role": "user", "content": instruction},
]
```

Độ dài input inference nên nhất quán với training. Nếu train dùng `max_length=2048`, không nên cắt input inference xuống `1536` khi prompt chứa nhiều document. Có thể dùng:

```python
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=2048,
).to(model.device)
```

Dùng `do_sample=False` để kết quả đánh giá có thể lặp lại.

## Luồng chạy đề xuất

```text
raft.ipynb
    |
    v
checkpoint JSON trên Google Drive
    |
    v
test.ipynb: load + split theo source_chunk_id
    |
    v
oversample memory trên train
    |
    v
QLoRA fine-tuning Qwen2.5-3B-Instruct
    |
    v
lưu adapter LoRA
    |
    v
tải lại adapter + inference
    |
    v
đánh giá held-out theo mode, format và citation
```

## Bảo mật và tái lập

- Không commit API key hoặc Hugging Face token.
- Giữ `random_state=42` và `random.Random(42)` để tái lập split/shuffle.
- Không oversample test set.
- Không chia các sample cùng `source_chunk_id` sang hai tập.
- Lưu raw prediction cùng metric để kiểm tra các lỗi cụ thể.
