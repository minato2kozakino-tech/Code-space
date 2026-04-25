1. Framework tích hợp & kiến trúc RNN
👉 Khuyến nghị tối ưu (thực tế + nhẹ máy):
TF-IDF = retrieval context (chính)
RNN (PyTorch) = generator + hiểu ngữ cảnh hội thoại
Pipeline đề xuất:

Input user
   ↓
TF-IDF (vector hóa)
   ↓
Retrieve top-k câu gần nhất trong dataset
   ↓
Ghép context + input
   ↓
RNN (GRU/LSTM) xử lý
   ↓
Output
Chọn RNN nào?
⚡ GRU → nhẹ hơn, phù hợp điện thoại (Snapdragon 680 của Anh)
🧠 LSTM → tốt hơn chút về dài hạn nhưng nặng hơn
👉 Em đề xuất:
→ dùng GRU trước, nếu đủ tài nguyên thì nâng cấp LSTM
2. Định dạng CSV & tiền xử lý
Format chuẩn:
Csv
human,bot
"Xin chào","Chào bạn!"
"Bạn là ai?","Tôi là AI"
Tiền xử lý nên có:
lowercase
remove ký tự rác
tokenize (đơn giản thôi, không cần quá nặng)
padding sequence
Sentence segmentation?
👉 Không cần thiết nếu data đã là hội thoại từng dòng
Chỉ cần nếu:
dữ liệu là paragraph dài
hoặc crawl từ web
👉 Với dataset của Anh: → bỏ qua segmentation cho nhẹ
3. Vai trò TF-IDF
👉 Em khuyên dùng 1 mục tiêu duy nhất cho sạch kiến trúc:
✔ TF-IDF = retrieval context
KHÔNG nên dùng cho:
phân loại intent (sẽ yếu)
language detection (không ổn)
Vì sao?
Dataset của Anh là hội thoại → TF-IDF cực mạnh trong việc tìm câu gần giống
RNN sẽ học cách phản hồi dựa trên context
👉 Tóm lại:

TF-IDF = memory
RNN = brain
4. Nhận diện yêu cầu sinh code
Có 2 cách:
❌ Cách 1: keyword (không nên)
dễ miss
không scale
✔ Cách 2: intent classification (khuyên dùng)
Cách làm nhẹ:
dùng TF-IDF + LogisticRegression (sklearn)
label:
chat
code
Pipeline đầy đủ:

Input
 ↓
Intent classifier
 ↓
 ├── nếu "code" → gọi module code generator riêng
 └── nếu "chat" → đi qua TF-IDF + RNN
Module code nên tách riêng?
👉 BẮT BUỘC tách riêng 💡
Vì:
logic khác hoàn toàn
tránh làm bẩn model hội thoại
dễ scale sau này (LLM / template / AST)
