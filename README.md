# WattBot 2026 baseline

Pipeline RAG ưu tiên evidence, triển khai theo roadmap: kiểm tra dữ liệu → PDF → retrieval → trả lời có nguồn → tính toán Decimal → validation → scorer chính thức.

Hiện đã chạy và kiểm tra **phần CPU**. Phần sinh đáp án bằng GPU cần chạy tiếp trên máy chủ; chưa có điểm RAG answer hay submission leaderboard.

## Kết quả đã đo tại máy phát triển

- Dữ liệu: 245 train, 317 test, 122 tài liệu trong metadata.
- Tải thành công 121 PDF; `agl2025` còn lỗi HTTP 403 / timeout. Không thay nguồn bằng tài liệu khác.
- Parse: 2.290 trang, 5.803 chunks; 7 trang được gắn cờ cần OCR.
- BM25, top 50 chunks: macro gold citation recall **0,90764** trên 240 câu có đáp án; đủ toàn bộ gold refs ở **216/240 câu**.
- Context sau bỏ chunks trùng trang, giữ 5 chunks: citation recall **0,81667**.
- Scorer gốc tự chấm train: **1,000**. Baseline từ chối tất cả: **0,06939**, chỉ dùng kiểm tra hệ thống chấm điểm.

Chi tiết và giới hạn: [docs/BASELINE.md](docs/BASELINE.md). Recall tài liệu không chứng minh đã lấy đúng đoạn evidence hay trả lời đúng.

## Chạy trên máy chủ SSH của bạn

Cấu hình đã nhận: 8 × RTX 2080 Ti, 11 GB mỗi GPU; driver 535.161.07. Các GPU đang được sử dụng. Ảnh cung cấp cho thấy GPU 2 còn khoảng 5,9 GiB, nhưng cần kiểm tra lại trước khi chạy. Khởi đầu bằng `Qwen/Qwen2.5-3B-Instruct`, NF4 4-bit, tính toán FP16, một GPU; chưa đo tốc độ hay VRAM thực tế trên server.

```bash
git clone https://github.com/pho416/WAG_BOT.git
cd WAG_BOT
# Nếu đã clone: git pull --ff-only

# Chép WattBot2026.zip bạn đã tải vào thư mục này.
# Python 3.10 hoặc 3.11, Linux, và kết nối Internet để tải dependencies/model/PDF.
GPU_ID=2 bash scripts/setup_gpu.sh

# Tải corpus, tạo chunks, chạy BM25 và đo retrieval.
GPU_ID=2 bash scripts/run_server.sh prepare

# Nếu một PDF không truy cập được, xem manifest. Có thể tiếp tục có chủ đích:
# ALLOW_PARTIAL_CORPUS=1 GPU_ID=2 bash scripts/run_server.sh prepare

# Thử 5 câu trước, sau đó mới chạy toàn bộ train.
GPU_ID=2 bash scripts/run_server.sh smoke
GPU_ID=2 bash scripts/run_server.sh train
```

`setup_gpu.sh` tạo `.venv` riêng và cài PyTorch 2.5.1 CUDA 12.1 cùng các dependencies đã ghim. Không cần sửa môi trường đang chạy của người khác. Nếu môi trường shell chưa có `python3-venv`, cần cài gói đó hoặc tạo môi trường Python tương đương trước.

Runner đặt `CUDA_VISIBLE_DEVICES` theo `GPU_ID` và dùng GPU đó. Không dừng các tiến trình đang có. Loader chừa 2 GiB bộ nhớ làm headroom; nếu không đủ bộ nhớ sẽ dừng. Đổi `GPU_ID` sau khi xem `nvidia-smi`, hoặc đợi tài nguyên phù hợp. Khi có đủ VRAM, có thể thử `MODEL=Qwen/Qwen2.5-7B-Instruct` trong **thư mục output mới** để so sánh công bằng.

Gửi lại `runs/gpu_smoke/`, `runs/gpu_train/evaluation/metrics.json` và trace của câu lỗi để tiếp tục phân tích. Mỗi câu lưu trace và CSV; chạy lại cùng cấu hình sẽ dùng các câu đã hoàn thành.

## Chạy CPU tại Windows

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m wattbot unpack --archive WattBot2026.zip
.\.venv\Scripts\python.exe -m wattbot profile
.\.venv\Scripts\python.exe -m wattbot split
.\.venv\Scripts\python.exe -m wattbot download
.\.venv\Scripts\python.exe -m wattbot ingest
.\.venv\Scripts\python.exe -m wattbot retrieve --context-k 5
.\.venv\Scripts\python.exe -m wattbot retrieval-eval
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Nếu `download` báo có tài liệu lỗi, xem `artifacts/download_manifest.json`. Có thể chạy lại `download --only-failed`; lệnh `ingest` vẫn hỗ trợ corpus chưa đủ và ghi rõ tài liệu thiếu trong `data/processed/ingestion.json`. PDF có sẵn cần đặt đúng tên `data/pdfs/<metadata id>.pdf`; nguồn cục bộ chưa đối chiếu download sẽ được đánh dấu `local_unverified`.

## Thử hybrid và reranker

Sau khi đã có baseline GPU, chạy ablation retrieval:

```bash
GPU_ID=2 bash scripts/run_server.sh hybrid
GPU_ID=2 bash scripts/run_server.sh hybrid-train

# Reranker tùy chọn, chạy CPU để dành VRAM cho LLM:
.venv/bin/python -m wattbot retrieve \
  --dense-dir artifacts/dense --reranker cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --device cpu --context-k 5 --output runs/reranked/train_retrieval.jsonl
.venv/bin/python -m wattbot retrieval-eval \
  --retrieval runs/reranked/train_retrieval.jsonl --output runs/reranked/metrics.json
```

BM25 + dense dùng Reciprocal Rank Fusion, sau đó cross-encoder nếu bật. Dense mặc định là `BAAI/bge-small-en-v1.5`. So sánh trên cùng corpus và cùng split; giữ validation để chọn cấu hình, tránh liên tục chọn thay đổi dựa trên toàn bộ train. Split cố định seed 2026 tạo 196 development / 49 validation, cân bằng gần đúng theo nhiều evidence flags và cohort; chưa tách theo tài liệu.

## Trả lời, kiểm chứng và chấm điểm

- Inference chỉ đọc `id`, `question`, `answer_unit`, `Cohort`. Không đưa gold answers, refs, quotes hay flags vào model hoặc index.
- Structured JSON phải có answer, explanation, trích dẫn bằng `chunk_id` thật và đoạn quote tìm được trong chunk đó.
- Bộ kiểm tra đối chiếu quote, ref_id và toán hạng; Decimal chỉ thực thi biểu thức số cho phép, không chạy mã Python từ model.
- Một lượt model review kiểm tra lại câu trả lời theo evidence. Kiểm tra quote tự động không bảo đảm suy luận đúng; cần đánh giá lỗi thực tế.
- Khi không đủ bằng chứng, `answer_value`, `ref_id`, `ref_url`, `supporting_materials` đều là `is_blank`. `answer_unit=is_blank` chỉ là đơn vị categorical/không chỉ định, không có nghĩa câu hỏi không có đáp án.
- Lỗi model/JSON dừng và lưu trace; không biến lỗi kỹ thuật thành quyết định từ chối.
- Scorer được nạp từ `data/raw/Score.py` nguyên bản. Validator chặn ID thiếu/thừa/trùng, unit bị đổi, nguồn không có trong metadata và từ chối còn kèm evidence.
- Báo cáo theo nhóm tính lại đúng trọng số 2026 (0,75 value + 0,20 citation F1 + 0,05 abstention); xem các điểm chưa nhất quán trong scorer tại [docs/BASELINE.md](docs/BASELINE.md).

Ví dụ chạy một cấu hình riêng:

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m wattbot generate \
  --model Qwen/Qwen2.5-3B-Instruct --retrieval runs/bm25/train_retrieval.jsonl \
  --output-dir runs/experiment_01
.venv/bin/python -m wattbot evaluate --submission runs/experiment_01/predictions.csv \
  --output-dir runs/experiment_01/evaluation
```

Với `--limit 5`, CSV chỉ chứa 5 câu; validator cần `--questions <output-dir>/questions.csv`. Evaluator yêu cầu gold có đúng các ID này, không chấm CSV thiếu hàng với toàn bộ train.

Trước bản nộp cuối, ghim `--revision <HuggingFace commit>` cho generation/embedding và `--reranker-revision` nếu dùng reranker. Thông tin model generation thực tế, hash input/code và cấu hình được lưu trong `run.json`; revision embedding nằm ở `dense.json`. Giữ `requirements-server.lock.txt`, manifests và artifacts cùng bản chạy.

## Sinh CSV test và tiếp tục roadmap

```bash
GPU_ID=2 bash scripts/run_server.sh test
```

CSV ở `runs/gpu_test/predictions.csv`; lệnh chỉ tạo và validate, không nộp lên Kaggle. Đây là bản text baseline. Chưa triển khai visual embeddings/ColPali, MinerU/OCR, VLM đọc hình/bảng hay iterative multi-hop. Đã có dữ liệu block/bbox/caption, render trang, Decimal và kiểm chứng để phát triển các bước đó sau khi có error analysis của baseline.

Notebook tùy chọn: [notebooks/WattBot_GPU.ipynb](notebooks/WattBot_GPU.ipynb). Chạy `python scripts/make_notebook.py` nếu cần tạo lại. Có thể đóng gói dự án cục bộ bằng `python scripts/make_bundle.py`, thêm `--include-pdfs` khi cần chuyển toàn bộ corpus. Các bundle chứa dữ liệu cuộc thi chỉ dùng để chuyển môi trường của bạn, không đưa vào Git.

Tham khảo triển khai: [Qwen2.5-3B model card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), [Transformers 4-bit quantization](https://huggingface.co/docs/transformers/v4.48.0/quantization/bitsandbytes), [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/).
