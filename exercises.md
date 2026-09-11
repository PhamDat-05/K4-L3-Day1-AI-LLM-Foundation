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
> Khi temperature tăng, câu trả lời thường đa dạng và sáng tạo hơn nhưng cũng khó đoán hơn; ở temperature thấp, model có xu hướng trả lời ổn định và tập trung hơn. Với code hiện tại, các lời gọi thực tế dùng model Gemini qua endpoint tương thích OpenAI, nên nội dung cụ thể có thể thay đổi giữa các lần chạy.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ bắt đầu với temperature khoảng 0.2–0.4. Mức thấp giúp câu trả lời nhất quán, chính xác và ít bịa hơn, phù hợp với việc trả lời câu hỏi khách hàng; chỉ tăng lên khi chatbot cần cách diễn đạt linh hoạt hơn.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Theo bảng giá gốc trong code, workload có 10.000 × 3 × 350 = 10.500.000 token output mỗi ngày. GPT-4o tốn khoảng 105 USD/ngày, còn GPT-4o-mini khoảng 6,30 USD/ngày, tức GPT-4o đắt khoảng 16,7 lần. Tuy nhiên code hiện tại mặc định gọi Gemini và dùng bảng giá Gemini; model nhẹ phù hợp cho FAQ đơn giản, còn model mạnh hơn đáng dùng cho phân tích phức tạp hoặc yêu cầu chất lượng cao.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Với persona giáo viên tiểu học, phản hồi thường ngắn, dùng từ đơn giản và có ví dụ gần gũi với trẻ em. Với persona chuyên gia tài chính, phản hồi thường dùng thuật ngữ kỹ thuật, giải thích sâu hơn và có thể dài hơn. System prompt được gửi ở role `system` trước user prompt trong code, nên nó định hướng vai trò, mức độ chi tiết, từ vựng và cách trình bày của model; kết quả cụ thể vẫn có thể thay đổi theo từng lần gọi Gemini.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với model Gemini mặc định hiện tại, `tiktoken.encoding_for_model` không nhận diện model nên code rơi vào fallback `max(1, len(text)//4)`, chứ không đếm token Gemini thật. Ví dụ một đoạn khoảng 100 từ và 500 ký tự cho khoảng 125 token theo fallback, trong khi công thức `số từ/0.75` cho khoảng 133,3 token, chênh khoảng 6,25% so với cách ước lượng theo từ. Con số chính xác phụ thuộc đoạn văn; tiếng Việt thường tốn nhiều token hơn tiếng Anh vì cách tách từ, dấu và mức độ phổ biến trong bộ mã hóa.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming hữu ích khi phản hồi dài hoặc cần hiển thị ngay phần đầu để người dùng cảm thấy hệ thống phản hồi nhanh, chẳng hạn chatbot hội thoại. Trong code hiện tại, `streaming_chatbot` gửi lịch sử trực tiếp với `stream=True`, in từng chunk và giữ tối đa 3 lượt, nhưng không đặt system persona. `run_assistant` cũng stream nhưng thêm persona vào role `system`. Non-streaming phù hợp khi cần toàn bộ câu trả lời trước khi xử lý tiếp, kiểm tra định dạng hoặc lưu dữ liệu.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff tăng dần thời gian chờ giữa các lần thử, giúp API có thời gian giảm tải và giảm số request lặp lại khi đang quá tải. Nếu hàng nghìn client cùng retry sau đúng một delay cố định, chúng sẽ tạo ra các đợt request đồng thời, làm tình trạng quá tải kéo dài. Trong code, lần gọi đầu chạy ngay; sau mỗi lỗi, delay là `base_delay * 2^attempt`, rồi thử lại tối đa `max_retries` lần.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Persona tôi chọn là: "Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt và giải thích khái niệm bằng ví dụ dễ hiểu." Cụm "trợ giảng thân thiện" định hướng giọng điệu hỗ trợ, còn "ngắn gọn bằng tiếng Việt" giúp câu trả lời dễ đọc và phù hợp với người học trong khóa. Code truyền persona này vào role `system` cho toàn bộ phiên trợ lý.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là history chỉ giữ tối đa 3 lượt gần nhất, nên trợ lý có thể quên thông tin quan trọng ở đầu phiên. Tôi sẽ lưu các lượt cũ vào cơ sở dữ liệu hoặc file, sau đó tìm lại những đoạn liên quan bằng embedding trước mỗi lần gọi API. Chỉ những đoạn được truy xuất cùng history gần nhất mới được đưa vào `messages`, giúp giữ ngữ cảnh mà không làm prompt tăng vô hạn.

---
