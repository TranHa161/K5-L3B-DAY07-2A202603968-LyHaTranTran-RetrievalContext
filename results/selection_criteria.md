# Tiêu chí chọn cấu hình (công bố trước khi chạy thử nghiệm Checkpoint 3)

Ghi ngày 2026-11-16, trước khi chạy các cấu hình cải thiện trên dev. Chỉ dùng **dev** (d01–d19); không xem test.

Thứ tự ưu tiên (so sánh lần lượt, chỉ xét tiêu chí sau khi tiêu chí trước hòa):

1. **Chất lượng câu trả lời (dev, 17 câu có đáp án):** số câu trả lời đúng. Khi chọn sơ bộ dùng proxy tự động
   `answered ∧ citation chứa evidence gold ∧ ≥ 80% con số trong expected_answer xuất hiện trong câu trả lời`;
   cấu hình được chọn và baseline phải được chấm tay (0/1/2) xác nhận.
2. **Abstention:** phải từ chối đúng 2/2 câu thiếu bằng chứng (d18, d19). Cấu hình trả lời bừa câu thiếu bằng chứng bị loại.
3. **Context Evidence Rate** của context cuối gửi model.
4. **Chi phí và thời gian:** tổng token vào/ra và thời gian trung vị mỗi câu. Hòa ở 1–3 thì chọn cấu hình rẻ hơn;
   chấp nhận tốn hơn tối đa ~1,5× token nếu tiêu chí 1 tăng ít nhất 1 câu.

Retrieval-only metrics (Hit@1, Evidence Hit@3, MRR@10) dùng để chẩn đoán, không quyết định trực tiếp.
Sau khi chốt, ghi cấu hình vào `results/chosen_config.json` rồi mới chạy test với baseline (dense) và cấu hình chọn.
