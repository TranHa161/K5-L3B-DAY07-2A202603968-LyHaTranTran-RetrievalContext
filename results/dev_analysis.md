# Checkpoint 3 — Thử nghiệm trên dev, phân tích lỗi, chọn cấu hình

Corpus: IDR 2026 Part 1, 271 chunk (Day 6 strategy B), index pgvector `google/embeddinggemma-2` 768d, cosine exact.
Dev: 17 câu có đáp án (d01–d17) + 2 câu thiếu bằng chứng (d18, d19). Model agent `gemini-2.5-flash`, temperature 0.
Tiêu chí chọn công bố trước: [selection_criteria.md](selection_criteria.md). Bảng đầy đủ: [dev_comparison.md](dev_comparison.md)
(tạo lại bằng `python src/compare.py --split dev`). Chi phí embedding/BM25 = 0 USD (local); chi phí model tính theo token.

## 1. Ba phương pháp retrieval (query gốc, top-5, budget 6000 ký tự)

| Phương pháp | Hit@1 | Evidence Hit@3 | MRR@10 | Context Evidence | Miss@3 |
|---|---|---|---|---|---|
| Dense (pgvector Day 6) | 10/17 | 14/17 | 0.705 | 14/17 | d02, d07, d12 |
| BM25 (Okapi, k1=1.5, b=0.75) | 13/17 | 16/17 | 0.860 | 16/17 | d03 |
| Hybrid RRF (k=60, pool 20+20) | 13/17 | **17/17** | **0.863** | **17/17** | – |

Filter chapter (do câu hỏi cung cấp): dense có/không filter cho kết quả giống hệt nhau trên dev.
Sweep top-k: dense top-8 đưa evidence vào context 17/17 (top-3/5: 14/17); BM25 miss d03 ở mọi k (từ vựng câu hỏi
"components make up" khác văn bản "sum of ... public and publicly guaranteed debt").

## 2. Cấu hình agent và giả thuyết

"Proxy đúng" = trả lời ∧ citation chứa evidence gold ∧ ≥80% con số trong `expected_answer` xuất hiện.
Proxy này **quá chặt**: 7 câu (d07, d09, d10, d11, d12, d14, d15) trả lời đúng ý chính nhưng thiếu số phụ trong
đáp án chuẩn (ví dụ "up from 8.2 percent in 2024") nên bị tính sai ở **mọi** cấu hình — không đổi thứ hạng; chấm tay
sẽ quyết định 1 hay 2 điểm. Cột "Câu khác biệt" chỉ liệt kê lỗi riêng của cấu hình đó (ngoài 7 câu chung).

| Run | Retriever | Prompt | top-k / budget | Proxy đúng | Context Evidence | Từ chối nhầm | Abstention | Search TB | Token vào/ra | s trung vị | Câu khác biệt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense-baseline | dense | v1 | 5 / 6000 | 8/17 | 13/17 | 3/17 | 2/2 | 1.00 | 41,963 / 2,751 | 4.3 | d02 (từ chối), d17 (thiếu ý) |
| **C1** dense-v2-research | dense | v2 | 5 / 6000 | 9/17 | 14/17 | 1/17 | 2/2 | 1.11 | 51,192 / 2,864 | 4.8 | d17; d07 trả lời sai số (80.8% excl. China) |
| **C4** dense-k8-b9000 | dense | v1 | 8 / 9000 | 9/17 | 16/17 | 0/17 | 2/2 | 1.00 | 54,919 / 3,022 | 3.3 | d17 (thiếu ý) |
| bm25-baseline | bm25 | v1 | 5 / 6000 | 10/17 | 17/17 | 0/17 | 2/2 | 1.00 | 41,957 / 2,611 | 3.3 | – |
| hybrid-baseline | hybrid | v1 | 5 / 6000 | 10/17 | 17/17 | 0/17 | 2/2 | 1.00 | 42,810 / 2,940 | 3.2 | – |
| **C2** hybrid-v2-research | hybrid | v2 | 5 / 6000 | 10/17 | 17/17 | 0/17 | 2/2 | 1.05 | 50,084 / 2,945 | 3.1 | – |
| **C3** hybrid-k3-b3000 | hybrid | v1 | 3 / 3000 | 7/17 | 14/17 | 1/17 | 2/2 | 1.00 | 31,711 / 2,430 | 3.1 | d02 (từ chối), d05 (sai), d17 (lệch) |

- **H1 (C1, C2) — prompt v2 buộc tìm lại trước khi từ chối sẽ cứu các câu dense bỏ sót.** Một phần đúng:
  agent chỉ tìm lại ở 2/19 câu. Dense: cứu d02, nhưng d07 chuyển từ từ chối sang trả lời sai (dùng số 80.8% của
  nhóm "excluding China"), d12 vẫn từ chối. +22% token. Hybrid: không đổi chất lượng, +17% token. → Không chọn.
- **H2 (C3) — context nhỏ hơn (top-3, 3000 ký tự) giữ chất lượng mà rẻ hơn.** Bác bỏ: −26% token nhưng mất 3 câu
  (d02 từ chối; d05 trả lời "$227 billion long-term stock" thay vì mức tăng; d17 lệch ý vì chunk p125 bị loại).
- **H3 (C4) — dense với top-8 bù được miss của dense.** Đúng: từ chối nhầm 3 → 0, Context Evidence 13 → 16/17, nhưng
  +31% token và d17 vẫn thiếu ý chính; vẫn kém hybrid top-5 với ít token hơn.
- Sửa parser citation (`[a, b]`) trước đó: dense từ chối nhầm 5 → 3, hybrid 4 → 0 (run `*-v0-strict-citation-parser`).

## 3. Phân tích lỗi dev

1. **d06/d09/d04/d12 — parser citation quá chặt (lỗi code).** Model viết `[id1, id2]`; code chỉ chấp nhận `[id]` nên
   đổi câu trả lời đúng thành từ chối. Sửa `cited_ids()` + test; đo lại: xem dòng v0 ở trên.
2. **d02, d07, d12 — dense không lấy được evidence trong top-5.** d07: query hỏi "share ... private creditors", chunk
   gold p040-c001 nói "75.2 percent of total long-term net debt inflows" nhưng dense xếp p040 nhóm excl. China (80.8%) cao hơn.
   Agent v1 từ chối thay vì tìm lại. Khắc phục: hybrid (BM25 bắt "private creditors", "long-term net debt inflows")
   hoặc tăng top-k (C4). Prompt tìm lại (C1) giúp d02 nhưng gây trả lời sai ở d07 → nguy cơ nhầm nhóm đối tượng.
3. **d17 — câu trả lời lệch ý do context thiếu chunk p125.** Dense đưa p114/p126 (asymmetries khác) và model liệt kê
   lý do đúng với nguồn nhưng không phải lý do câu hỏi hỏi (IDS loại trừ khoản vay PNG không tách được). Hybrid/BM25
   đưa p125 vào context → trả lời đúng. Citation "hợp lệ" không đồng nghĩa trả lời đúng câu hỏi.
4. **d18/d19 — từ chối đúng ở mọi cấu hình**, kể cả d19 (Viet Nam có xuất hiện trên p58 nhưng không có số FDI riêng).

## 4. Cấu hình chọn

**Chốt: `hybrid-baseline`** — hybrid RRF, prompt v1, top-5, budget 6000 ký tự, tối đa 3 search ([chosen_config.json](chosen_config.json)).
Theo tiêu chí 1–3, hybrid hòa BM25 và hơn mọi cấu hình dense/C1–C4. Tiêu chí 4: BM25 ít hơn ~2% token vào nhưng chậm
hơn 0,09 s/câu → coi là hòa; phân định bằng retrieval (Evidence Hit@3 17/17 so với 16/17) và giữ vector store Day 6 trong
pipeline. Prompt v2 tốn thêm 17% token không cải thiện; context nhỏ làm giảm chất lượng. Baseline so sánh trên test: `dense-baseline`.
