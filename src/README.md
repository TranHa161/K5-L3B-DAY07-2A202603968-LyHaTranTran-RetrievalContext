# Code Day 7 — cách chạy

Tái sử dụng môi trường Day 6 (torch, sentence-transformers, psycopg) và thêm `google-genai` ([requirements.txt](../requirements.txt)).

## Chuẩn bị (một lần)

```powershell
# 1. Bật vector store Day 6 (dùng đúng project name để giữ volume có dữ liệu)
docker start day6_student_starter-postgres-1      # hoặc: docker compose -p day6_student_starter up -d (trong Day 6/starter)

# 2. Python của venv Day 6 + UTF-8
$env:PYTHONUTF8=1
$PY="C:\Users\Admin\Documents\AI20K_LAB\K5-L3B-DAY06-LyHaTranTran-2A202603968-DataIngestion\starter\.venv\Scripts\python.exe"
& $PY -m pip install google-genai==3.3.0

# 3. Key Gemini: copy .env.example -> .env, điền GEMINI_API_KEY (DATABASE_URL đọc từ Day 6 starter/.env)
```

## Chạy

```powershell
& $PY src/prepare_data.py                                   # Day 6 -> data/day7/{chunks,questions,cases}.json
& $PY -m unittest discover -s tests -v                      # test offline (mock)
& $PY src/eval_retrieval.py --method dense  --split dev     # dense | bm25 | hybrid; --context-k, --budget, --no-filter
& $PY src/agent.py --split dev --retriever hybrid --name hybrid-baseline   # --prompt v2, --top-k, --budget, --ids d01 d18
& $PY src/compare.py --split dev                            # bảng results/dev_comparison.md
& $PY src/make_reviews.py                                   # reviews.json + agent_metrics.json (nhãn chấm tay)
```

Cấu hình đã chốt: [results/chosen_config.json](../results/chosen_config.json). Phân tích dev: [results/dev_analysis.md](../results/dev_analysis.md).

| File | Vai trò |
|---|---|
| `prepare_data.py` | Chuyển chunk strategy B + benchmark Day 6, thêm câu thiếu bằng chứng, kiểm evidence nằm trọn trong chunk |
| `retrievers.py` | `DenseRetriever` (pgvector Day 6), `BM25Retriever`, `HybridRRFRetriever` |
| `context.py` | Chọn context: bỏ trùng, giữ thứ tự, budget ký tự |
| `eval_retrieval.py` | Hit@1, Evidence Hit@3, MRR@10, Context Evidence Rate |
| `agent.py` | Agent Gemini function calling (`search`/`answer`/`abstain`), trace đầy đủ |
| `compare.py` | Bảng so sánh retrieval + agent (proxy tự động) |
| `make_reviews.py` | Nhãn chấm tay (bản nháp) và tổng hợp bằng `starter/agent_metrics.py` |
