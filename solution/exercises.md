# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Temperature càng thấp (tiệm cận 0.0) thì câu trả lời càng mang tính tất định (deterministic), chính xác và có xu hướng lặp lại cùng một sự thật phổ biến qua nhiều lần chạy. Ngược lại, temperature càng cao (1.0 - 1.5) thì câu trả lời càng ngẫu nhiên, đa dạng và sáng tạo hơn nhưng cũng tăng dần nguy cơ câu chữ lan man, thiếu tự nhiên hoặc sai lệch thông tin (hallucination).

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Nên đặt temperature ở mức thấp, khoảng 0.0 đến 0.3 (lý tưởng là ~0.1 - 0.2). Lý do là chatbot chăm sóc khách hàng cần sự chuẩn xác cao, tính nhất quán (consistency) và bám sát chính sách/thông tin sản phẩm của doanh nghiệp. Mức temperature thấp giúp giảm thiểu tối đa hiện tượng "ảo giác" (hallucination) hay bịa đặt thông tin sai lệch gây rủi ro kinh doanh, đồng thời vẫn giữ được câu trả lời tự nhiên.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> - Ước tính chi phí: Theo bảng giá lab ($0.010/1K token output cho GPT-4o vs $0.0006/1K token output cho GPT-4o-mini), GPT-4o đắt hơn GPT-4o-mini khoảng 16.7 lần (0.010 / 0.0006 ≈ 16.67). Với 30.000 lượt gọi/ngày × 350 token ≈ 10.5M token output/ngày, chi phí output cho mini chỉ khoảng $6.3/ngày (~$189/tháng), trong khi GPT-4o tốn tới $105/ngày (~$3,150/tháng).
> - Trường hợp GPT-4o xứng đáng: Các tác vụ yêu cầu lập luận logic phức tạp, giải quyết khiếu nại khách hàng VIP/nhạy cảm, phân tích tài liệu pháp lý/tài chính, hoặc sinh code chuyên sâu — nơi một sai sót nhỏ có thể gây thiệt hại tài chính lớn.
> - Trường hợp nên dùng mini: Các tác vụ hỗ trợ khách hàng cơ bản (hỏi đáp FAQ có tài liệu tham chiếu), phân loại ý định (intent classification), tóm tắt tin nhắn ngắn, hoặc các tính năng người dùng miễn phí ở quy mô lớn để tối ưu chi phí.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Khi system prompt định hướng model là "giáo viên tiểu học", phản hồi trở nên ngắn gọn, sử dụng từ vựng đơn giản, dễ hiểu và có xu hướng dùng các ví dụ gần gũi với trẻ em. Ngược lại, khi định hướng là "chuyên gia tài chính", phản hồi dài hơn, dùng thuật ngữ chuyên ngành, cung cấp thông tin chi tiết và sâu sắc hơn về mặt kỹ thuật hoặc kinh doanh.

Điều này cho thấy system prompt có vai trò định hình hành vi, phong cách giao tiếp (tone & style), độ dài, và mức độ chuyên sâu của phản hồi. Nó giúp điều hướng model đi đúng hướng mong muốn mà không cần phải mô tả chi tiết trong từng prompt, từ đó cải thiện tính nhất quán và khả năng kiểm soát đầu ra của LLM.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Số token theo tiktoken là 107 (chênh lệch 7% so với ước lượng). Tiếng Việt thường tốn nhiều token hơn tiếng Anh cùng độ dài vì tiếng Việt là ngôn ngữ đơn âm tiết (mỗi từ thường chỉ có 1 âm tiết hoặc ít âm tiết), trong khi tiếng Anh có xu hướng ghép nhiều âm tiết vào cùng một từ. Hơn nữa, cách mã hóa token của GPT (dựa trên BPE) được tối ưu hóa cho tiếng Anh, dẫn đến việc các từ hoặc cụm từ tiếng Việt cần nhiều byte và nhiều token hơn để biểu diễn.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi hiển thị phản hồi cho các tác vụ tạo văn bản dài (generate dài), như viết báo cáo, tóm tắt tài liệu, hoặc soạn thảo nội dung sáng tạo. Nó giúp cải thiện trải nghiệm người dùng (UX) đáng kể bằng cách giảm cảm giác chờ đợi (perceived latency) và cho phép người dùng đọc nội dung khi nó đang được tạo ra.

Ngược lại, non-streaming phù hợp hơn cho các tác vụ xử lý ngắn, có tính chất giao dịch (transactional), yêu cầu kết quả tức thì và đầy đủ trước khi chuyển sang bước tiếp theo. Ví dụ bao gồm: phân loại dữ liệu (classification), trích xuất thông tin cụ thể (extraction), xác thực (validation) hoặc các lệnh API yêu cầu trả về JSON chuẩn. Trong các trường hợp này, việc nhận toàn bộ kết quả một lần giúp giảm độ phức tạp của luồng xử lý và tránh lỗi do dữ liệu chưa hoàn chỉnh trong quá trình streaming.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
>Exponential backoff giúp giảm tải đột ngột cho API và tránh xung đột.
Khi API bị quá tải, việc các client cùng chờ một khoảng thời gian cố định (ví dụ: 1 giây) trước khi thử lại sẽ khiến lưu lượng truy cập dồn về cùng một thời điểm, làm trầm trọng thêm tình trạng quá tải và có thể gây ra hiệu ứng "thác đổ" (cascading failure).
Ngược lại, exponential backoff tăng dần thời gian chờ (ví dụ: 1 giây, 2 giây, 4 giây, 8 giây, ...), giúp phân tán các yêu cầu retry ra theo thời gian, cho phép API có cơ hội phục hồi dần dần và giảm thiểu áp lực đồng thời lên hệ thống.
---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> - Persona đã chọn: Trợ giảng AI thân thiện, chuyên hỗ trợ học viên giải đáp thắc mắc lập trình và kiến thức nền tảng.
> - System prompt:
>   ```text
>   "Bạn là trợ giảng thân thiện của khóa học AI, chuyên giải đáp thắc mắc về lập trình và LLM. Hãy trả lời ngắn gọn, súc tích bằng tiếng Việt, tập trung vào trọng tâm và đưa ra ví dụ code minh họa khi cần thiết."
>   ```
> - Giải thích các lựa chọn từ ngữ quan trọng:
>   1. *"trả lời ngắn gọn, súc tích"*: Ràng buộc mô hình không sinh văn bản dài dòng lan man, giúp tiết kiệm số lượng output tokens (giảm chi phí API và giảm đáng kể độ trễ latency khi streaming trên giao diện dòng lệnh CLI).
>   2. *"bằng tiếng Việt"*: Chỉ định rõ ràng ngôn ngữ phản hồi để mô hình luôn phản hồi bằng tiếng Việt đồng nhất, tránh tình trạng tự động chuyển sang tiếng Anh khi gặp các thuật ngữ kỹ thuật chuyên sâu.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> - Hạn chế lớn nhất hiện tại: Cắt tỉa lịch sử theo cơ chế trượt cố định (sliding window cứng) chỉ giữ lại 3 lượt gần nhất (`history[-6:]`). Điều này khiến trợ lý bị mất ngữ cảnh (context loss) — nếu cuộc trò chuyện kéo dài qua 4 lượt, các thông tin quan trọng ở các lượt đầu (như tên người dùng, yêu cầu bài toán cụ thể đang làm) sẽ bị xóa hoàn toàn.
> - Đề xuất cải thiện: Triển khai cơ chế **Bộ đệm tóm tắt lịch sử (Conversation Summary Buffer Memory)** kết hợp kiểm soát ngân sách token (Token Budget).
> - Mô tả cách triển khai:
>   1. Đếm token tích lũy của mảng `history` bằng `count_tokens()`.
>   2. Khi tổng token của `history` vượt quá ngưỡng quy định (ví dụ: 1.000 tokens), không xóa bỏ trực tiếp các tin nhắn cũ mà gọi LLM (có thể dùng model chi phí thấp như `gpt-4o-mini`) để tóm tắt các tin nhắn cũ đó thành một đoạn ngắn gọn.
>   3. Lưu đoạn tóm tắt này vào một biến `summary` và ghép vào đầu danh sách messages: `[{"role": "system", "content": f"{persona}\n\n[Tóm tắt ngữ cảnh trước]: {summary}"}] + recent_history`.
>   4. Như vậy, trợ lý vừa duy trì được ngữ cảnh xuyên suốt cuộc trò chuyện dài, vừa kiểm soát được độ dài prompt và chi phí API một cách tối ưu.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
