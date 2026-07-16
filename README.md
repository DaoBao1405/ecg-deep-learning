# ECG Deep Learning

Dự án phân loại nhịp tim điện tâm đồ (ECG) thành 5 lớp bằng các mô hình học sâu: RNN, LSTM và CNN-LSTM.

## Mục tiêu

So sánh hiệu quả của các kiến trúc mạng tuần tự trong bài toán phân loại tín hiệu ECG:

- RNN làm mô hình cơ sở (baseline).
- LSTM để học tốt hơn các phụ thuộc dài trong chuỗi.
- CNN-LSTM để kết hợp trích xuất đặc trưng cục bộ và học quan hệ theo thời gian.

## Cấu trúc dự án

- `rnn-ecg.ipynb`: xây dựng, huấn luyện và đánh giá RNN và LSTM.
- `ecg-cnn-lstm.ipynb`: xây dựng, huấn luyện và đánh giá CNN-LSTM.
- `requirements.txt`: danh sách các thư viện Python cần thiết.

Các checkpoint `best_ecg_rnn.pth`, `best_ecg_lstm.pth` và `best_ecg_cnn_lstm.pth` sẽ được tạo sau khi huấn luyện.

## Dữ liệu

Dự án sử dụng bộ dữ liệu MIT-BIH Arrhythmia đã tiền xử lý, gồm:

- `mitbih_train.csv`
- `mitbih_test.csv`

Mỗi mẫu ECG có 187 điểm tín hiệu; cột cuối là nhãn thuộc một trong 5 lớp. Dữ liệu huấn luyện được chia theo tỉ lệ 80% train và 20% validation; tập `mitbih_test.csv` được dùng để kiểm tra.

## Mô hình

### RNN

RNN được dùng làm mô hình cơ sở để học chuỗi tín hiệu ECG. Mô hình sử dụng input shape `(batch_size, 187, 1)`, hidden size 64, 2 tầng recurrent và dropout 0.2.

### LSTM

RNN cơ bản có thể gặp hiện tượng mất dần gradient khi học các quan hệ dài trong chuỗi. LSTM được dùng để cải thiện vấn đề này nhờ các cổng có khả năng lưu giữ và chọn lọc thông tin quan trọng.

Tuy nhiên, LSTM vẫn xử lý tín hiệu chủ yếu theo tuần tự và không được thiết kế chuyên biệt để trích xuất đặc trưng hình thái cục bộ của sóng ECG. Trong kết quả thực nghiệm, LSTM có F1-score thấp ở nhãn 1 (0.1418) và nhãn 3 (0.0000). Điều này cho thấy cấu hình LSTM hiện tại chưa nhận diện hiệu quả một số lớp hiếm và các mẫu có biến đổi cục bộ. Kết quả cũng chịu ảnh hưởng bởi mất cân bằng dữ liệu: nhãn 1 có 556 mẫu và nhãn 3 có 162 mẫu trong tập kiểm tra.

### CNN-LSTM

CNN-LSTM bổ sung CNN 1D để trích xuất đặc trưng cục bộ của dạng sóng ECG trước khi LSTM học quan hệ theo thời gian. Sự kết hợp này giúp mô hình vừa nhận biết biến đổi ngắn hạn của tín hiệu, vừa khai thác ngữ cảnh của toàn bộ nhịp tim.

## Cài đặt

```bash
pip install -r requirements.txt
```

Nếu muốn sử dụng GPU NVIDIA, hãy cài PyTorch theo phiên bản CUDA phù hợp với máy từ trang chính thức của PyTorch.

## Cách chạy

1. Tải hai tệp dữ liệu `mitbih_train.csv` và `mitbih_test.csv`.
2. Cập nhật đường dẫn đọc dữ liệu trong notebook:

```python
train = pd.read_csv("path/to/mitbih_train.csv")
test = pd.read_csv("path/to/mitbih_test.csv")
```

3. Mở Jupyter Notebook và chạy các cell theo thứ tự:

```bash
jupyter notebook
```

Sau đó mở `rnn-ecg.ipynb` hoặc `ecg-cnn-lstm.ipynb`.

## Kết quả thực nghiệm

Kết quả được đánh giá trên tập kiểm tra gồm 21.891 mẫu.

| Mô hình | Test Loss | Test Accuracy | Macro F1-score | Weighted F1-score |
| --- | ---: | ---: | ---: | ---: |
| RNN | 0.8086 | 82.62% | 0.2119 | 0.7535 |
| LSTM | 0.4385 | 91.54% | 0.5429 | 0.9055 |
| CNN-LSTM | 0.0680 | **98.12%** | **0.8946** | **0.9807** |

CNN-LSTM đạt kết quả tốt nhất. Macro F1-score được báo cáo bên cạnh Accuracy vì dữ liệu mất cân bằng giữa các lớp.

## Đánh giá

Mỗi notebook cung cấp các chỉ số và biểu đồ sau:

- Training loss và validation loss.
- Training accuracy và validation accuracy.
- Test loss và test accuracy.
- Precision, recall và F1-score theo từng lớp.
- ROC/AUC đa lớp.
- Confusion matrix.

## Lưu ý

- Mô hình tự động sử dụng GPU nếu CUDA khả dụng.
- Checkpoint tốt nhất được lưu theo validation loss.
- Cần tải lại checkpoint tốt nhất trước khi đánh giá cuối cùng trên tập test.
- Các mô hình hiện dùng một số siêu tham số khác nhau, như learning rate và trọng số lớp. Vì vậy, bảng kết quả thể hiện hiệu năng của từng cấu hình đã thử nghiệm, chưa phải so sánh kiến trúc hoàn toàn kiểm soát.

