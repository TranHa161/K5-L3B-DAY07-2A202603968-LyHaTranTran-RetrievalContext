# So sánh cấu hình trên test

## Retrieval (query gốc, filter chapter do người dùng cung cấp)

| Run | Hit@1 | Evidence Hit@3 | MRR@10 | Context k | Budget (ký tự) | Context Evidence | Ký tự context TB | ms trung vị | Miss@3 |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 4/6 | 5/6 | 0.7917 | 5 | 6000 | 6/6 | 4468 | 1.1 | t04 |
| dense | 4/6 | 6/6 | 0.8056 | 5 | 6000 | 6/6 | 4213 | 295.0 | - |
| hybrid | 5/6 | 6/6 | 0.9167 | 5 | 6000 | 6/6 | 4652 | 272.9 | - |

## Agent (Gemini, temperature 0) — proxy tự động, cần chấm tay xác nhận

| Run | Retriever | Prompt | top-k | Budget | Proxy đúng | Context Evidence | Citation có evidence | Từ chối nhầm | Abstention đúng | Lỗi | Search TB | Token vào/ra | s trung vị | Câu lỗi (proxy) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense-baseline | dense | v1 | 5 | 6000 | 3/6 | 6/6 | 6/6 | 0/6 | 2/2 | 0 | 1.12 | 18,761/1,028 | 2.6799999999999997 | t03, t04, t05 |
| hybrid-baseline | hybrid | v1 | 5 | 6000 | 3/6 | 6/6 | 6/6 | 0/6 | 2/2 | 0 | 1 | 17,273/983 | 3.55 | t03, t04, t05 |
