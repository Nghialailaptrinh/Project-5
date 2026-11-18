# Project 5: Machine Learning & Deep Learning với PyTorch

Dự án triển khai các mô hình học máy và học sâu cơ bản bằng **PyTorch** (theo chuẩn bài tập CS188 UC Berkeley), bao gồm 6 bài toán từ Perceptron, Hồi quy phi tuyến, Phân loại ảnh MNIST (MLP & CNN), Nhận diện ngôn ngữ bằng RNN cho đến cơ chế Causal Self-Attention.

---

## Cấu trúc các file chính

Toàn bộ nội dung cài đặt của dự án được thực hiện trong 3 file cốt lõi:
- **`models.py`**: Định nghĩa kiến trúc các mô hình nơ-ron (`Module`, `Parameter`, `Linear`) và phép toán lan truyền tiến (`forward`, `Convolve`).
- **`losses.py`**: Định nghĩa các hàm mục tiêu / hàm mất mát (`mse_loss`, `cross_entropy`) cho từng bài toán.
- **`train.py`**: Cài đặt vòng lặp huấn luyện (`DataLoader`, `optim.Adam`, lan truyền ngược và điều kiện dừng sớm).

---

## Mô tả chi tiết theo từng Question (`models.py`, `losses.py`, `train.py`)

### Question 1: Perceptron
Phân loại dữ liệu nhị phân $x \in \mathbb{R}^d$ vào hai lớp $\{+1, -1\}$ bằng mô hình tuyến tính không dùng hệ số chệch (bias).

- **`models.py` (`PerceptronModel`)**:
  - Khởi tạo vector trọng số `self.w = Parameter(ones(1, dimensions))`.
  - `get_weights()`: Trả về tham số `self.w`.
  - `forward(x)`: Tính điểm số tuyến tính qua tích vô hướng `(x @ self.w.T).reshape(1)`.
  - `get_prediction(x)`: Trả về nhãn `1` nếu `score.item() >= 0`, ngược lại trả về `-1`.
- **`losses.py`**:
  - Không sử dụng hàm loss khả vi; mô hình cập nhật trực tiếp theo quy tắc Perceptron khi dự đoán sai.
- **`train.py` (`train_perceptron`)**:
  - Chạy trong khối `with no_grad():` với `DataLoader(dataset, batch_size=1, shuffle=True)`.
  - Duyệt qua từng mẫu `(x, y)`, nếu `prediction != y` thì cập nhật trực tiếp `model.w += y * x` và tăng biến đếm `mistakes`.
  - Hội tụ và dừng vòng lặp ngay khi kết thúc 1 epoch với `mistakes == 0`.
- **Kết quả**: Đạt **5/5 điểm** (hội tụ 100% trên tập huấn luyện).

---

### Question 2: Hồi quy phi tuyến (Non-linear Regression)
Xấp xỉ hàm phi tuyến $y = \sin(x)$ trên đoạn $[-2\pi, 2\pi]$.

- **`models.py` (`RegressionModel`)**:
  - Kiến trúc MLP 3 lớp tuyến tính (`1 -> 64 -> 64 -> 1`, tổng cộng 4.353 tham số): `Linear(1, 64)`, `Linear(64, 64)`, `Linear(64, 1)`.
  - `forward(x)`: Áp dụng hàm kích hoạt `relu` ở 2 lớp ẩn đầu tiên (`relu(self.linear1(x))`, `relu(self.linear2(hidden1))`) và giữ nguyên đầu ra tuyến tính ở lớp cuối `self.linear3(hidden2)` để dự đoán cả giá trị âm lẫn dương trong $[-1, 1]$.
- **`losses.py` (`regression_loss`)**:
  - Tính sai số bình phương trung bình (MSE) giữa dự đoán `y_pred` và nhãn thật `y`: `mse_loss(y_pred, y).reshape(1)`.
- **`train.py` (`train_regression`)**:
  - Sử dụng `DataLoader(dataset, batch_size=50, shuffle=True)` và bộ tối ưu `optim.Adam(model.parameters(), lr=0.01)`.
  - Huấn luyện tối đa `2000` epoch theo chu trình chuẩn: `zero_grad() -> forward -> regression_loss -> backward() -> step()`.
  - Dừng sớm (early stopping) khi `average_loss <= 0.01` (vượt ngưỡng yêu cầu $\le 0.02$ của autograder).
- **Kết quả**: Đạt **5/5 điểm**.

---

### Question 3: Phân loại chữ số viết tay (Digit Classification – MNIST)
Phân loại ảnh xám chữ số viết tay kích thước $28 \times 28$ (trải phẳng thành vector 784 chiều) vào 10 lớp từ `0` đến `9`.

- **`models.py` (`DigitClassificationModel`)**:
  - Kiến trúc mạng truyền thẳng `784 -> 256 -> 128 -> 10` (235.146 tham số) gồm: `Linear(784, 256)`, `Linear(256, 128)`, `Linear(128, 10)`.
  - `forward(x)`: Qua 2 lớp ẩn dùng `relu` và trả về trực tiếp 10 điểm số `logits` ở lớp cuối (không dùng `relu` ở đầu ra).
- **`losses.py` (`digitclassifier_loss`)**:
  - Sử dụng hàm Cross-Entropy nhận trực tiếp logits và nhãn one-hot 10 chiều: `cross_entropy(y_pred, y).reshape(1)`.
- **`train.py` (`train_digitclassifier`)**:
  - Khởi tạo `DataLoader(dataset, batch_size=100, shuffle=True)` và `optim.Adam(model.parameters(), lr=0.001)`.
  - Chạy tối đa `20` epoch, cuối mỗi epoch kiểm tra `dataset.get_validation_accuracy()` và dừng sớm khi đạt `validation_accuracy >= 0.98` (tạo biên an toàn so với mốc 97% trên tập test).
- **Kết quả**: Hội tụ ở Epoch 6 (`validation_accuracy = 0.9808`), đạt **97.80%** trên tập test (**5/5 điểm**).

---

### Question 4: Nhận diện ngôn ngữ bằng RNN (Language Identification)
Xác định ngôn ngữ của một từ có độ dài $L$ bất kỳ thuộc 1 trong 5 ngôn ngữ (`English`, `Spanish`, `Finnish`, `Dutch`, `Polish`) từ bảng chữ cái gồm 47 ký tự.

- **`models.py` (`LanguageIDModel`)**:
  - Khai báo kích thước trạng thái ẩn `hidden_size = 256` (tổng 79.109 tham số cố định cho mọi độ dài từ), gồm:
    - `input_layer = Linear(47, 256)`
    - `hidden_layer = Linear(256, 256, bias=False)`
    - `output_layer = Linear(256, 5)`
  - `forward(xs)`: Khởi tạo `hidden = zeros(batch_size, 256)`, duyệt tuần tự qua từng ký tự `char` trong danh sách `xs` (độ dài $L$) và cập nhật trạng thái ẩn theo công thức đệ quy:
    `hidden = relu(self.input_layer(char) + self.hidden_layer(hidden))`.
    Sau ký tự cuối cùng, đưa `hidden` qua `self.output_layer(hidden)` để lấy 5 logits.
- **`losses.py` (`languageid_loss`)**:
  - Tính hàm mất mát Cross-Entropy cho bài toán phân loại 5 lớp: `cross_entropy(y_pred, y).reshape(1)`.
- **`train.py` (`train_languageid`)**:
  - Dùng `DataLoader(dataset, batch_size=64, shuffle=True)` và `optim.Adam(model.parameters(), lr=0.001)`.
  - Chuyển đổi chiều dữ liệu đầu vào bằng `x = batch["x"].movedim(1, 0)` để đưa từ dạng `(batch_size, L, 47)` sang `(L, batch_size, 47)` trước khi truyền vào `model(x)`.
  - Huấn luyện tối đa `35` epoch, chuyển sang `model.eval()` cuối mỗi epoch để đánh giá và dừng sớm nếu `validation_accuracy >= 0.88` (mốc yêu cầu của autograder là 81%).
- **Kết quả**: Đạt **82.80%** trên tập test (**5/5 điểm**).

---

### Question 5: Mạng nơ-ron tích chập (Convolutional Neural Network – CNN)
Phân loại chữ số MNIST rút gọn bằng phép tích chập 2D tự cài đặt kết hợp các lớp kết nối đầy đủ.

- **`models.py` (`Convolve` & `DigitConvolutionalModel`)**:
  - Hàm `Convolve(input, weight)`: Cài đặt trực tiếp phép tích chập 2D (cross-correlation với `stride=1`, không padding) trên ảnh đầu vào `(in_h, in_w)` và bộ lọc `(k_h, k_w)`, tạo ma trận đầu ra `zeros((out_h, out_w))` với `out_h = in_h - k_h + 1`, `out_w = in_w - k_w + 1` bằng cách trượt và cộng dồn `weight[i, j] * input[i:i + out_h, j:j + out_w]`.
  - Lớp `DigitConvolutionalModel`:
    - Khởi tạo kernel `self.convolution_weights = Parameter(ones((3, 3)))`, cùng 2 lớp `Linear(26 * 26, 128)` và `Linear(128, 10)` (tổng 87.955 tham số).
    - `forward(x)`: Reshape đầu vào từ `(B, 784)` thành `(B, 28, 28)`, áp dụng `Convolve` cho từng ảnh trong batch để thu được feature map `(B, 26, 26)`, làm phẳng bằng `flatten(start_dim=1)` thành vector 676 chiều, rồi truyền qua `relu(self.linear1(x))` và `self.linear2(x)`.
- **`losses.py` (`digitconvolution_Loss`)**:
  - Tính Cross-Entropy loss giữa logits dự đoán và nhãn thật: `cross_entropy(y_pred, y).reshape(1)`.
- **`train.py` (`Train_DigitConvolution`)**:
  - Sử dụng `DataLoader(dataset, batch_size=20, shuffle=True)`, `optim.Adam(model.parameters(), lr=0.001)`, chạy tối đa `30` epoch.
  - Cuối mỗi epoch chuyển sang `model.eval()` trong khối `with no_grad():` để kiểm tra `validation_accuracy` và dừng sớm khi `validation_accuracy >= 0.85` (ngưỡng yêu cầu là 80%).
- **Kết quả**: Hội tụ sau 2 epoch (`validation_accuracy = 0.9200`), đạt **98.00%** trên tập test (**3/3 điểm**).

---

### Question 6: Cơ chế Attention (Scaled Dot-Product Attention & Causal Mask)
Cài đặt khối Causal Self-Attention giúp mô hình tổng hợp thông tin ngữ cảnh từ các token hiện tại và quá khứ mà không nhìn trước tương lai.

- **`models.py` (`Attention`)**:
  - `__init__(self, layer_size, block_size)`:
    - Khai báo 3 lớp chiếu tuyến tính `k_layer`, `q_layer`, `v_layer` có cùng kích thước `Linear(layer_size, layer_size)`.
    - Đăng ký mặt nạ tam giác dưới `mask` qua `register_buffer("mask", tril(ones(block_size, block_size)).view(1, 1, block_size, block_size))`.
  - `forward(self, input)`: Với tensor đầu vào kích thước `(B, T, C)`:
    1. Tính `q = self.q_layer(input)`, `k = self.k_layer(input)`, `v = self.v_layer(input)`.
    2. Tính điểm tương đồng đã chuẩn hóa: `scores = matmul(q, movedim(k, 1, 2)) / (self.layer_size ** 0.5)`.
    3. Áp mặt nạ nhân quả (gán `-inf` cho các vị trí tương lai): `scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))[0]`.
    4. Chuẩn hóa xác suất bằng `weights = softmax(scores, dim=-1)` và trả về tích `matmul(weights, v)`.
- **`losses.py` & `train.py`**:
  - Question 6 kiểm tra trực tiếp tính đúng đắn của phép tính `Attention.forward()` trong `models.py`, không yêu cầu định nghĩa hàm loss hay hàm train riêng.
- **Kết quả**: Đạt **2/2 điểm** (`PASS: check_attention`).

---

## Hướng dẫn cài đặt và chạy lại chương trình

### 1. Yêu cầu môi trường
- **Python**: 3.8 trở lên (khuyến nghị Python 3.9 – 3.11)
- **Thư viện cần thiết**: `torch`, `numpy`, `matplotlib`

Cài đặt các thư viện bằng `pip`:
```bash
pip install torch numpy matplotlib
```

Kiểm tra môi trường và đồ họa bằng lệnh tích hợp sẵn của dự án:
```bash
python autograder.py --check-dependencies
```

---

### 2. Chạy chấm điểm toàn bộ dự án (Question 1 – Question 6)
Mở terminal tại thư mục gốc của dự án (`Project-5`) và chạy:

```bash
python autograder.py
```

Nếu muốn chạy nhanh ở chế độ **không hiển thị cửa sổ đồ họa** (headless):
```bash
python autograder.py --no-graphics
```

---

### 3. Chạy kiểm tra riêng từng Question
Sử dụng cờ `-q` (hoặc `--question`) kèm mã câu hỏi (`q1` đến `q6`):

```bash
# Question 1: Perceptron
python autograder.py -q q1

# Question 2: Non-linear Regression
python autograder.py -q q2

# Question 3: Digit Classification (MNIST MLP)
python autograder.py -q q3

# Question 4: Language Identification (RNN)
python autograder.py -q q4

# Question 5: Convolutional Neural Network (CNN)
python autograder.py -q q5

# Question 6: Attention Mechanism
python autograder.py -q q6
```

Có thể kết hợp thêm `--no-graphics` hoặc `--mute` (ẩn log in ra từ quá trình train):
```bash
python autograder.py -q q3 --no-graphics
```

---

## Tổng kết điểm số Autograder

| Câu hỏi | Nội dung | Hàm / Lớp kiểm tra chính | Điểm số |
| :--- | :--- | :--- | :---: |
| **Question 1** | Perceptron | `PerceptronModel`, `train_perceptron` | **5 / 5** |
| **Question 2** | Non-linear Regression | `RegressionModel`, `regression_loss`, `train_regression` | **5 / 5** |
| **Question 3** | Digit Classification | `DigitClassificationModel`, `digitclassifier_loss`, `train_digitclassifier` | **5 / 5** |
| **Question 4** | Language Identification (RNN) | `LanguageIDModel`, `languageid_loss`, `train_languageid` | **5 / 5** |
| **Question 5** | Digit Convolution (CNN) | `Convolve`, `DigitConvolutionalModel`, `digitconvolution_Loss`, `Train_DigitConvolution` | **3 / 3** |
| **Question 6** | Causal Self-Attention | `Attention` | **2 / 2** |
| **Tổng cộng** | | | **25 / 25** |
