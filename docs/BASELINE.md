# CPU baseline — 2026-10-27

Phạm vi đã chạy: unpack/kiểm tra dataset, scorer tự chấm, split, tải/parse PDF, BM25, đo citation recall trên train và baseline từ chối toàn bộ. Chưa chạy LLM GPU, dense model, reranker hay vision. Không có điểm leaderboard.

## Dữ liệu và provenance

| Thành phần | Số lượng |
| --- | ---: |
| Train / test | 245 / 317 |
| Metadata / PDF tải được | 122 / 121 |
| Trang / chunks | 2.290 / 5.803 |
| Trang ít text, cần kiểm tra OCR | 7 |
| Quote / Table / Figure | 194 / 38 / 17 |
| Math / CrossPaper / Reconcile | 25 / 19 / 11 |
| Không có đáp án | 5 |

Các nhóm evidence chồng lấp. `agl2025` chưa tải được do 403 và timeout tại URL metadata. Corpus thiếu được ghi nhận công khai; chưa loại câu liên quan khỏi phép đo để làm đẹp recall. Một số URL arXiv trong metadata không ghim phiên bản; downloader giữ nguyên URL, đánh dấu `version_pinned=false`, và lưu SHA256 của PDF tải được. URL `/abs/` được chuyển thành `/pdf/` với nguyên số phiên bản.

Hash dữ liệu gốc:

```text
metadata.csv  b54eb64f7747493443465a58822f87bc12655ee8aab9be83b0958222bb3c1ca1
train_QA.csv  9cbc25a9cb6133e1ef833fad6eb7fe43f9b72c1533b39d3b1ae94b3172407dca
test_Q.csv    26038b21cb09588a39b0e95e5516e3827dadd499c85c3792d0ea46c10ae7066a
Score.py      e5050458932b7a3fc0f4040303d3ae7c0459a786cf0efbb52c2f1ac338bbd075
```

Chunks tối đa 300 từ, overlap 60 từ, không qua ranh giới trang. Giữ text gốc, ref_id, số trang PDF từ 1, tọa độ block, title, year, URL và captions nhận diện bằng pattern. Section chưa được suy ra; bảng và hình chưa được parse có cấu trúc. Đây là giới hạn của baseline, không xem caption là đã đọc chart.

## Retrieval

BM25 dùng title + text, `k1=1,5`, `b=0,75`, không học theo gold. Lấy 50 chunks; context lấy 5 chunks, tối đa 1 chunk mỗi trang.

Trung bình macro theo câu hỏi có gold refs; 5 câu không có đáp án bị loại khỏi recall, không tính là retrieval hoàn hảo. Đây là **recall tài liệu**, không phải độ chính xác nội dung evidence hay accuracy câu trả lời. Full-train là số đo chẩn đoán; không dùng nó để khẳng định tổng quát hóa.

| Nhóm | n | Recall refs trong top 5 chunks | Recall refs trong top 50 chunks | Recall refs trong context 5 chunks |
| --- | ---: | ---: | ---: | ---: |
| Tất cả có đáp án | 240 | 0,81250 | 0,90764 | 0,81667 |
| Quote | 194 | 0,80155 | 0,88832 | 0,80412 |
| Table | 38 | 0,94737 | 1,00000 | 0,94737 |
| Figure | 17 | 0,94118 | 1,00000 | 0,94118 |
| Math | 25 | 0,72000 | 0,98000 | 0,74000 |
| CrossPaper | 19 | 0,50877 | 0,95614 | 0,53509 |
| Reconcile | 11 | 0,56061 | 0,96970 | 0,60606 |

Top 50 có đầy đủ gold refs ở 216/240 câu (90%). Context mất đáng kể nguồn cho CrossPaper/Reconcile. Bước tiếp theo cần đo hybrid/reranking và chiến lược giữ đa dạng tài liệu, rồi kiểm tra tác động lên điểm trả lời thực tế. Không kết luận model đọc được hình/bảng từ các recall trong bảng này.

`document_recall@k` trong JSON deduplicate tài liệu từ cùng pool top-50 chunks; không phải một phép tìm kiếm tài liệu độc lập trên toàn corpus.

## Scorer

- Ground truth tự chấm bằng scorer gốc: **1,000**.
- Từ chối tất cả 245 câu, format đúng: **0,0693877551**. Đây là điểm sàn kiểm tra evaluator, không phải kết quả RAG.
- Điểm tổng gốc dùng đúng 0,75 answer + 0,20 citation F1 + 0,05 NA.
- `LAST_BREAKDOWN` theo nhóm vẫn dùng công thức cũ 0,75/0,15/0,10 với citation Jaccard. Báo cáo riêng tính lại đúng công thức 2026.
- `score()` chỉ join ba cột answer_value/answer_unit/ref_id trước khi gọi `row_bits`, nên phần clean abstention của nó không nhìn thấy ref_url/supporting_materials. Validator dự án kiểm tra đủ trường.
- Scorer chấp nhận giao của ID thay vì bắt buộc đủ hàng. Validator/evaluator dự án yêu cầu tập ID khớp hoàn toàn và không trùng.

Không sửa file scorer gốc và không tận dụng các điểm chưa nhất quán để tạo submission. Mỗi báo cáo lưu hash scorer và prediction CSV.

## Những việc còn cần chạy

1. SSH server: smoke 5 câu với model 3B 4-bit trên một GPU đủ trống, kiểm tra JSON, latency và VRAM.
2. Chạy train baseline, chấm scorer và phân tích nhóm lỗi. Kiểm tra holdout 196/49 đã tạo trước khi chọn cải tiến.
3. Ablation BM25 → hybrid → reranker trên cùng corpus/split.
4. Bổ sung parsing bảng/OCR/vision và iterative multi-hop dựa trên lỗi đã đo.
5. Khi chọn được cấu hình, ghim revision model/dependencies, chạy lại test và tạo CSV; việc nộp leaderboard là bước riêng.
