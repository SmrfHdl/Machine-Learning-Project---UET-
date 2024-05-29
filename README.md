# **Thông Tin Dự Án**

> **Tên Dự Án**: Nhận diện biển báo giao thông
>
> **Tên nhóm:** Hội người cao tuổi

**Ngày thực hiện:** 14/05/2024

>

**Repostories Github:** https://github.com/SmrfHdl/Machine-Learning-Project---UET-.git

# **Các thành viên của nhóm:**

| Họ tên                       | MSSV     |
| ---------------------------- | -------- |
| Nguyễn Viết Vũ (Trưởng nhóm) | 22022632 |
| Phạm Văn Trường              | 22022564 |
| Trần An Thắng                | 22022525 |

</aside>

# 1. Tổng quan về dự án

## 1.1. Tổng quan

Ngày nay, cơ sở hạ tầng giao thông ngày càng phát triển, việc phát hiện biển báo để cung cấp thông tin đến người tham gia giao thông là một điều rất quan trọng. Tận dụng những kiến thức đã được học ở môn Machine Learning, kết hợp với những điều gần gũi với đời sống. Nhóm tôi đã chọn đề tài “Phát hiện các loại biển báo giao thông” làm đề tài nghiên cứu. Giúp mọi người có thể hiểu đâu là biển báo giao thông và chúng có ý nghĩa thế nào.

## 1.2. Mô tả bài toán

1. Input: Một bức ảnh có chứa biển báo
2. Output: Tên biển báo

# 2. Xây dựng bộ dữ liệu

Về dữ liệu, nhóm tôi sẽ sử dụng bộ dữ liệu biển báo giao thông nổi tiếng đó là German Traffic Sign.

Bộ dữ liệu German Traffic Sign (GTSRB) là một bộ dữ liệu chứa hình ảnh về các biển báo giao thông Đức, được sử dụng phổ biến trong lĩnh vực nhận dạng biển báo giao thông và thị giác máy tính. Bộ dữ liệu này thường được sử dụng để huấn luyện và đánh giá các mô hình học máy và mạng nơ-ron sâu trong việc nhận dạng các biển báo giao thông.

## 2.1. Thông tin về bộ dữ liệu

Bộ dữ liệu này gồm khoảng gần 40k ảnh chia thành 43 folder là 43 loại biển báo khác nhau. Mỗi folder sẽ có 1 file CSV chứa thông tin các ảnh trong thư mục.

| Nhãn | Biển tương ứng               | Nhãn | Biển tương ứng            | Nhãn | Biển tương ứng             | Nhãn | Biển tương ứng                |
| ---- | ---------------------------- | ---- | ------------------------- | ---- | -------------------------- | ---- | ----------------------------- |
| 0    | Speed limit (20km/h)         | 12   | Priority road             | 24   | Road narrows on the right  | 36   | Go straight or right          |
| 1    | Speed limit (30km/h)         | 13   | Yield                     | 25   | Road work                  | 37   | Go straight or left           |
| 2    | Speed limit (50km/h)         | 14   | Stop                      | 26   | Traffic signals            | 38   | Keep right                    |
| 3    | Speed limit (60km/h)         | 15   | No vehicles               | 27   | Pedestrians                | 39   | Keep left                     |
| 4    | Speed limit (70km/h)         | 16   | Veh > 3.5 tons prohibited | 28   | Children crossing          | 40   | Roundabout mandatory          |
| 5    | Speed limit (80km/h)         | 17   | No Entry                  | 29   | Bicycles crossing          | 41   | End of no passing             |
| 6    | End of speed limit (80km/h)  | 18   | General caution           | 30   | Beware of ice/snow         | 42   | End no passing veh > 3.5 tons |
| 7    | Speed limit (100km/h)        | 19   | Dangerous curve left      | 31   | Wild animals crossing      |      |                               |
| 8    | Speed limit (120km/h)        | 20   | Dangerous curve right     | 32   | End speed + passing limits |      |                               |
| 9    | No passing                   | 21   | Double curve              | 33   | Turn right ahead           |      |                               |
| 10   | No passing veh over 3.5 tons | 22   | Bumpy road                | 34   | Turn left ahead            |      |                               |
| 11   | Right-of-way at intersection | 23   | Slippery road             | 35   | Ahead only                 |      |                               |

Dưới đây là một số thông tin chi tiết về bộ dữ liệu GTSRB:

- **Số Lượng Lớp**: 43 (mỗi lớp tương ứng với một loại biển báo giao thông)
- **Kích Thước Hình Ảnh**: Đa dạng, thường là 32x32 pixel hoặc 64x64 pixel
- **Số Lượng Hình Ảnh**: Khoảng 50,000 hình ảnh được chia thành tập huấn luyện, tập xác thực và tập kiểm thử
- **Định Dạng Hình Ảnh**: PNG hoặc JPEG

## 2.2. Đọc và xử lý dữ liệu

Bởi vì tập dữ liệu ở dạng ảnh, không thể trực tiếp đưa vào model để tiến hành trainning, vì vậy tôi cần phải trích xuất những đặc trưng của ảnh. Và kỹ thuật được sử dụng ở đây là HOG (Histograms of Oriented Gradients)

### 2.2.1. HOG (Histograms of Oriented Gradients)

Tôi sẽ sử dụng HOG((histogram of oriented gradients) để trích xuất đặc trưng trên những vùng mà cửa sổ trượt qua.

Histograms of Oriented Gradients (HOG) là một kỹ thuật trích xuất đặc trưng từ hình ảnh, được sử dụng rộng rãi trong các bài toán thị giác máy tính như phát hiện đối tượng và nhận dạng mẫu. HOG được giới thiệu bởi Navneet Dalal và Bill Triggs vào năm 2005 trong bài báo "Histograms of Oriented Gradients for Human Detection".

**Nguyên lý hoạt động:**

HOG hoạt động dựa trên việc đếm số lần xuất hiện của các gradient định hướng (hướng của sự thay đổi cường độ màu sắc) trong các vùng cục bộ của hình ảnh. Quá trình trích xuất đặc trưng HOG bao gồm các bước chính sau:

- **Tính toán gradient:** Sử dụng bộ lọc Sobel để tính toán hướng và độ lớn của gradient tại mỗi pixel trong ảnh.
- **Phân loại hướng:** Chia hình ảnh thành các ô (cell) nhỏ, thường là 8x8 pixel, và tính histogram của các gradient trong mỗi ô, với các hướng được chia thành các thùng (bin) dựa trên độ lớn của chúng.
- **Chuẩn hóa khối:** Ghép các ô thành các khối (block) lớn hơn, thường là 2x2 ô, và chuẩn hóa các histogram để giảm thiểu ảnh hưởng của ánh sáng và độ tương phản.
- **Tạo vector đặc trưng:** Kết hợp tất cả các vector đặc trưng của các khối trong toàn bộ hình ảnh thành một vector đặc trưng duy nhất.

**Lý do chọn HOG để trích xuất đặc trưng:**

- HOG sẽ trích xuất những đặc trưng của ảnh, từ đó giảm chiều dữ liệu, giảm thời thời gian tính toán và giảm độ phức tạp của mô hình.
- HOG tập trung vào sự thay đổi cường độ (gradient) hơn là giá trị tuyệt đối của cường độ ánh sáng. Do đó, ngay cả khi ánh sáng yếu, các cạnh và đường viền trong ảnh vẫn tạo ra các gradient mà HOG có thể sử dụng để trích xuất đặc trưng. Từ đó giảm bớt các bước để tiền xử lý đối với những ảnh quá sáng hoặc quá tối.

Tôi sẽ viết hàm trích xuất đặc trưng với đầu vào là đường dẫn ảnh:

```python
def compute_hog(image_path):
    try:
        image = Image.open(image_path)
        image = image.convert("L")
        image = image.resize((128, 128))

        hog, _ = feature.hog(np.array(image), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                             visualize=True, transform_sqrt=True, block_norm='L2-Hys')
        return hog
    except Exception as e:
        logging.error(f"Lỗi khi xử lý ảnh {image_path}: {e}")
        return None
```

Resize ảnh thành kích thước 128x128 pixels nhằm giúp cho đặc trưng ảnh thu được không quá ít, đồng thời cũng không tạo ra quá nhiều tham số giảm thời gian tải dữ liệu và huấn luyện mô hình.

### 2.2.2. Đọc dữ liệu và trích xuất đặc trưng

Để đọc và trích xuất đặc trưng từ dữ liệu tôi sử dụng đoạn code:

```python
def load_data_from_csv(csv_file):
    data = []
    labels = []

    with open(csv_file, 'r') as file:
        lines = file.readlines()[1:]  # Bỏ qua dòng tiêu đề
        for line in lines:
            parts = line.strip().split(',')
            image_path = parts[-1]
            hog = compute_hog(image_path)
            if hog is not None:
                data.append(hog)
                labels.append(int(parts[-2]))
    return np.array(data), np.array(labels)

X_train, Y_train = load_data_from_csv("./data/Train.csv")
```

Đặc trưng của mỗi ảnh được trích xuất ra từ tập training sẽ là một điểm dữ liệu trong một không gian đa chiều. Ở đây với ảnh được resize thành 128\*128px thì số chiều của không gian này sẽ là:

$$
(\frac{128}{8}_{(cells)}-1).(\frac{128}{8}_{(cells)}-1).(9_{(bins)}.4_{(blocks)})=8100
$$

```python
array([0.21429373, 0.23669159, 0.23669159, ..., 0.01807657, 0.00958505, 0. ]) # Arr 1 chiều 8100 features như một điểm trong không gian đa chiều
```

### **2.2.3. Thống kê và xử lý dữ liệu không cân bằng**

Số lượng nhãn chênh lệch là rất lớn, điều này sẽ làm phát sinh một số vấn đề:

- **Hiệu suất không đồng đều:** Mô hình sẽ có xu hướng dự đoán nhãn có nhiều dữ liệu hơn. Điều này làm cho mô hình kém hiệu quả trong việc dự đoán nhãn có ít dữ liệu.
- **Độ chính xác bị lệch:** Các chỉ số đánh giá như độ chính xác (accuracy) sẽ bị lệch nếu không có sự cân bằng giữa các nhãn. Mô hình có thể đạt độ chính xác cao đơn giản vì nó đoán đúng nhiều mẫu từ nhãn có nhiều dữ liệu.

Vấn đề dữ liệu không cân bằng có thể được giải quyết với các mô hình Ensemble như Random Forest, tuy nhiên để có thể sử dụng mô hình SVM thì tôi sẽ cần phải sử dụng kỹ thuật Resampling bằng SMOTE của thư viện ‘imblearn’. SMOTE (Synthetic Minority Over-sampling) và ADASYN (Adaptive synthetic sampling) là các phương pháp sinh mẫu nhằm gia tăng kích thước mẫu của nhóm thiểu số trong trường hợp xảy ra mất cân bằng mẫu. Để gia tăng kích thước mẫu, với mỗi một mẫu thuộc nhóm thiểu số ta sẽ lựa chọn ra 𝑘 mẫu láng giềng gần nhất với nó và sau đó thực hiện tổ hợp tuyến tính để tạo ra mẫu giả lập:

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, Y_train)
```

Sau khi tính toán giá trị trung bình là khoảng 1000 dữ liệu trên mỗi nhãn, tôi tiến hành tạo thêm dữ liệu để các nhãn có số dữ liệu quá thấp đạt tới 1000 dữ liệu.

Kết quả cho thấy hàm đã hoạt động tốt và cho ra tập dữ liệu khá cân bằng.

### 2.2.4. Chia tập dữ liệu

Tiếp theo để có thể đánh giá và tối ưu hóa mô hình tôi sẽ tiến hành chia tập dữ liệu thành 3 phần train - val - test với tỉ lệ 70% - 10% - 15%

```python
from sklearn.model_selection import train_test_split

x_train, x_remaining, y_train, y_remaining = train_test_split(X_train_resampled, y_train_resampled, test_size=0.25, random_state=42)
x_test, x_val, y_test, y_val = train_test_split(x_remaining, y_remaining, test_size=0.4, random_state=42)
```

Dữ liệu đã sẵn sàng, tôi sẽ tiến tới bước training model!!!

# 3. Training và đánh giá mô hình

## 3.1. Chọn base model để training

Những model được tôi sử dụng trong bài toán này sẽ là Support Vector Machine(SVM), K-Nearest Neighbor(KNN) và Random Forest(RF)

### **3.1.1. Support Vector Machine (SVM)**:

- SVM là một lựa chọn phổ biến khi làm việc với dữ liệu có số lượng lớn các đặc trưng, như trong trường hợp của HOG.
- SVM hoạt động tốt khi dữ liệu là tuyến tính hoặc có thể chuyển đổi thành tuyến tính thông qua các hàm kernel.
- SVM có thể được điều chỉnh bằng cách thay đổi siêu tham số như C (tham số đòi hỏi mạnh mẽ) và kernel để đạt được hiệu suất tốt nhất.

### 3.1.2. Random Forest:

- Random Forest là một phương pháp học máy ensemble dựa trên cây quyết định.
- Đối với dữ liệu HOG, Random Forest có thể được sử dụng để xây dựng một tập hợp các cây quyết định để phân loại ảnh.
- Đặc điểm của Random Forest là không nhạy cảm với overfitting và có khả năng làm việc tốt với dữ liệu có nhiễu.

### 3.1.3. **K-Nearest Neighbors (KNN)**:

- KNN là một phương pháp học máy đơn giản nhưng mạnh mẽ trong việc phân loại dữ liệu.
- KNN hoạt động bằng cách xác định nhãn cho một điểm dữ liệu mới bằng cách so sánh nó với các điểm dữ liệu trong tập huấn luyện gần nhất (K điểm gần nhất).
- HOG có thể được sử dụng như là đặc trưng đầu vào cho mỗi điểm dữ liệu.
- KNN không cần huấn luyện trước và có thể áp dụng trực tiếp vào dữ liệu mới mà không cần phải tái huấn luyện toàn bộ mô hình.

→ Tuy nhiên, KNN có thể đòi hỏi nhiều tài nguyên tính toán khi số lượng điểm dữ liệu trong tập huấn luyện lớn.

## 3.2. Đánh giá các base model

### 3.2.1. **Đánh giá mô hình SVM**

```python
Đánh giá mô hình SVM trên dữ liệu validation:
Accuracy: 0.9451306413301662
Classification Report:
              precision    recall  f1-score   support

           0       0.90      0.90      0.90        60
           1       0.93      0.91      0.92       720
           2       0.90      0.96      0.93       750
           3       0.89      0.85      0.87       450
           4       0.97      0.97      0.97       660
           5       0.84      0.88      0.86       630
           6       0.99      0.79      0.88       150
           7       0.91      0.93      0.92       450
           8       0.91      0.87      0.89       450
           9       0.99      0.97      0.98       480
          10       0.96      0.98      0.97       660
          11       0.95      0.94      0.95       420
          12       0.99      1.00      0.99       690
          13       1.00      1.00      1.00       720
          14       0.98      0.99      0.98       270
          15       0.95      1.00      0.97       210
          16       0.97      0.97      0.97       150
          17       1.00      1.00      1.00       360
          18       0.99      0.93      0.96       390
          19       0.92      1.00      0.96        60
          20       0.87      0.91      0.89        90
```

```python
          21       0.93      0.83      0.88        90
          22       0.93      0.80      0.86       120
          23       0.92      0.85      0.89       150
          24       0.95      0.99      0.97        90
          25       0.92      0.96      0.94       480
          26       0.92      0.82      0.87       180
          27       1.00      1.00      1.00        60
          28       0.94      0.96      0.95       150
          29       0.79      0.93      0.85        90
          30       0.80      0.77      0.78       150
          31       0.92      0.99      0.95       270
          32       1.00      0.92      0.96        60
          33       0.97      1.00      0.98       210
          34       0.97      0.99      0.98       120
          35       0.98      0.98      0.98       390
          36       0.98      0.98      0.98       120
          37       1.00      0.95      0.97        60
          38       0.99      0.99      0.99       690
          39       0.98      1.00      0.99        90
          40       0.96      0.98      0.97        90
          41       0.91      0.83      0.87        60
          42       0.99      0.88      0.93        90

    accuracy                           0.95     12630
   macro avg       0.94      0.93      0.94     12630
weighted avg       0.95      0.95      0.94     12630
```

Trong báo cáo phân loại của mô hình SVM, chúng ta thấy rằng precision, recall và f1-score đều cao cho hầu hết các lớp. Điều này cho thấy mô hình SVM có khả năng phân loại tốt trên cả các lớp lớn và nhỏ. Precision và recall gần như đều cao cho tất cả các lớp, chỉ có một số lớp có một vài điểm số thấp hơn, nhưng vẫn đạt được hiệu suất tốt (ví dụ: lớp 6, lớp 21, lớp 22).

Sở dĩ mô hình SVM đạt được độ chính xác xao như vậy là vì mô hình này đặc biệt hoạt động tốt trên các tập dữ liệu nhiều chiều như đặc trưng HOG.

### 3.2.2. **Đánh giá mô hình Random Forest**

```python
Đánh giá mô hình Random Forest trên dữ liệu validation:
Accuracy: 0.9346793349168646
Classification Report:
              precision    recall  f1-score   support

           0       1.00      0.97      0.98        60
           1       0.95      0.87      0.91       720
           2       0.86      0.97      0.91       750
           3       0.97      0.84      0.90       450
           4       0.95      0.95      0.95       660
           5       0.82      0.90      0.86       630
           6       0.93      0.75      0.83       150
           7       0.89      0.92      0.91       450
           8       0.87      0.80      0.83       450
           9       0.96      0.97      0.97       480
          10       0.93      0.97      0.95       660
          11       0.91      0.96      0.93       420
          12       0.98      1.00      0.99       690
          13       1.00      1.00      1.00       720
          14       1.00      0.96      0.98       270
          15       0.98      0.99      0.98       210
          16       0.99      0.97      0.98       150
          17       0.99      0.99      0.99       360
          18       0.97      0.88      0.93       390
          19       0.97      1.00      0.98        60
          20       0.90      0.91      0.91        90
```

```python
          21       0.98      0.63      0.77        90
          22       0.99      0.84      0.91       120
          23       0.93      0.95      0.94       150
          24       1.00      0.98      0.99        90
          25       0.90      0.96      0.93       480
          26       0.82      0.77      0.80       180
          27       0.98      0.90      0.94        60
          28       0.94      0.95      0.94       150
          29       0.95      0.90      0.93        90
          30       0.90      0.79      0.84       150
          31       0.90      0.98      0.94       270
          32       0.94      0.82      0.88        60
          33       0.96      1.00      0.98       210
          34       0.99      0.99      0.99       120
          35       0.96      0.98      0.97       390
          36       0.98      0.98      0.98       120
          37       0.98      0.90      0.94        60
          38       0.96      0.99      0.98       690
          39       0.99      1.00      0.99        90
          40       0.94      0.84      0.89        90
          41       0.85      0.87      0.86        60
          42       0.97      0.78      0.86        90

    accuracy                           0.93     12630
   macro avg       0.94      0.92      0.93     12630
weighted avg       0.94      0.93      0.93     12630
```

Bảng đánh giá cho thấy mô hình có hiệu suất tốt trên hầu hết các lớp, với precision, recall và F1-score đều ổn định. Random Forest hoạt động tốt trên dữ liệu HOG vì khả năng xử lý không gian đặc trưng lớn, dễ dàng ánh xạ các mẫu dữ liệu và không bị ảnh hưởng bởi biến đổi không quan trọng của dữ liệu.

### 3.2.3. **Đánh giá mô hình KNN**

```python
Đánh giá mô hình KNN trên dữ liệu validation:
Accuracy: 0.66270783847981
Classification Report:
              precision    recall  f1-score   support

           0       0.09      0.97      0.16        60
           1       0.87      0.41      0.55       720
           2       0.93      0.57      0.71       750
           3       0.67      0.46      0.54       450
           4       0.94      0.74      0.83       660
           5       0.53      0.67      0.59       630
           6       0.74      0.73      0.73       150
           7       0.84      0.68      0.75       450
           8       0.70      0.76      0.73       450
           9       0.95      0.45      0.61       480
          10       0.86      0.70      0.77       660
          11       1.00      0.08      0.15       420
          12       1.00      0.97      0.98       690
          13       1.00      0.99      1.00       720
          14       0.98      0.93      0.95       270
          15       0.30      0.99      0.46       210
          16       0.36      0.99      0.53       150
          17       1.00      0.71      0.83       360
          18       0.96      0.17      0.29       390
          19       0.24      0.98      0.38        60
          20       0.43      0.89      0.58        90
```

```python
          21       0.47      0.72      0.57        90
          22       0.39      0.78      0.53       120
          23       0.37      0.56      0.44       150
          24       0.27      0.86      0.42        90
          25       0.99      0.26      0.41       480
          26       0.38      0.58      0.46       180
          27       0.32      1.00      0.49        60
          28       0.38      0.50      0.43       150
          29       0.35      0.73      0.47        90
          30       0.26      0.34      0.30       150
          31       0.84      0.86      0.85       270
          32       0.73      0.93      0.82        60
          33       0.94      0.88      0.91       210
          34       0.49      0.98      0.65       120
          35       1.00      0.66      0.79       390
          36       0.78      1.00      0.88       120
          37       0.76      0.92      0.83        60
          38       1.00      0.66      0.79       690
          39       0.96      0.97      0.96        90
          40       0.52      0.90      0.66        90
          41       0.38      0.62      0.47        60
          42       0.54      0.88      0.67        90

    accuracy                           0.66     12630
   macro avg       0.66      0.73      0.63     12630
weighted avg       0.81      0.66      0.68     12630
```

- Tại biển báo nhãn 0 (Speed Limit 20) có chỉ số precision rất thấp (9%) điều này xảy ra là do hầu hết các biển báo speed limit có hình dáng khá tương đồng, chỉ thay đổi mỗi con số ở trong biển báo. Việc sử dụng HOG trích xuất đặc trưng ảnh với tỉ lệ ảnh là 128\*128px cũng sẽ làm cho những con số trong biển báo trở nên khó phân biệt hơn với mô hình KNN, chính vì vậy mà tỉ lệ precision của biển nhãn 0 rất thấp và tỉ lệ recall lại cao hơn so với các biển speed limit khác (từ 1 - 8). Ngoài ra số support cũng thấp hơn so với các biển còn lại cũng dẫn đến việc chỉ số recall cao.
- Điều này cũng xảy ra tương tự với biển số 11 và biển số 27 khi hình dáng khá tương tự nhau:

Những ảnh có hình dáng tương tự nhau sẽ gây ra sự nhầm lẫn lớn đối với mô hình KNN

Nhìn chung việc sử dụng mô hình KNN để đánh giá với các tập dữ liệu lớn và có độ phức tạp cao về cả dữ liệu và không gian đặc trưng sẽ đem lại hiệu quả rất kém. Với không gian với 8100 chiều thì các mẫu sẽ trở nên xa nhau trong không gian đặc trưng, làm cho việc tìm láng giềng gần nhất trở nên khó khăn và không chính xác.

## 3.3. Tối ưu model

Sau quá trình đánh giá thì tôi đã quyết định chọn 2 model để tiến hành stacking sử dụng Logistic Regrestion để chọn kết quả tốt nhất. Trước đó tôi sẽ tiến hành tối ưu model SVM sử dụng phương pháp Grid Search:

### 3.3.1. Tối ưu model SVM

```python
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# Danh sách các giá trị của siêu tham số cần điều chỉnh
param_grid = {'C': [0.01, 0.03, 0.05, 0.07, 0.1]}

# Tạo GridSearchCV với mô hình SVM và siêu tham số đã cho
grid_search = GridSearchCV(estimator=model_svm, param_grid=param_grid, cv=5)

# Huấn luyện GridSearchCV trên dữ liệu
grid_search.fit(x_train, y_train)

# Lấy kết quả của Grid Search
results = grid_search.cv_results_
```

Tôi sẽ sử dụng Grid Search để tìm kiếm siêu tham số tối ưu cho mô hình SVM. Cụ thể, chúng ta đã xác định một danh sách các giá trị cho tham số "C", đây là tham số quan trọng trong SVM.

Sau đó, tôi sử dụng GridSearchCV để thử tất cả các giá trị trong danh sách đó và đánh giá hiệu suất của mô hình SVM với mỗi giá trị của "C" trên tập dữ liệu huấn luyện. Bằng cách này, tôi có thể tìm ra giá trị tối ưu cho "C" mà cải thiện hiệu suất của mô hình.

Sau khi thử với các giá trị C khác nhau từ $10^{-3}$ cho đến 10 thì giá trị tốt nhất là 0.01 và tôi tiếp tục thu hẹp phạm vi và nhận thấy giá trị C mặc định vẫn là 0.01 sẽ cho ra độ chính xác cao nhất.

Vì thế tôi sẽ giữ nguyên siêu tham số này để tiến hành stacking

### 3.3.2. Training cho **meta-model**

Quy trình:

Do tài nguyên của máy tôi không đủ để thực hiện gridsearch tìm siêu tham số tối ưu nhất với mô hình Random Forest nên tôi sẽ tiến hành luôn bước huấn luyện cho meta model

```python
model_svm = SVC(probability=True)
model_rf = RandomForestClassifier()
model_svm.fit(x_train, y_train) # Giữ nguyên siêu tham số mặc định
model_rf.fit(x_train, y_train)
svm_test_prob = model_svm.predict_proba(x_test)
rf_test_prob = model_rf.predict_proba(x_test)
```

Đầu tiên tôi khởi tạo 2 mô hình cơ sở hiệu quả nhất trong 3 mô hình tôi đã đánh giá trước đó. Sau đó tạo mô một ma trận đặc trưng làm đầu vào huấn luyện cho meta-model:

```python
# Kết hợp xác suất dự đoán để tạo thành ma trận đặc trưng đầu vào cho mô hình tổng hợp
train_stacked_features = np.hstack((svm_train_prob, rf_train_prob))
test_stacked_features = np.hstack((svm_test_prob, rf_test_prob))
# Huấn luyện mô hình tổng hợp (Logistic Regression) trên tập huấn luyện
stacked_model = LogisticRegression()
stacked_model.fit(train_stacked_features, y_train)
```

## 3.4. Đánh giá meta-model

### 3.4.1. Accuracy

```python
Accuracy of svm model: 0.942596991290578
Accuracy of random forest model: 0.9338875692794932
Accuracy of stacked model: 0.9451702296120348
```

Có thể thấy bằng

# 4. Hướng phát triển và cải tiến

## **4.1. Cải tiến trong tương lai**

**Về data:**

- Bổ sung đa dạng các loại biển báo khác, đặc biệt là biển báo giao thông Việt Nam
- Tìm hiểu thêm các kĩ thuật tăng cường data. Việc áp dụng không hiệu quả các kĩ thuật tăng cường dữ liệu trong dự án này càng cho thấy tuy số lượng dữ liệu quan trọng nhưng chất lượng dữ liệu cũng là một yếu tố ảnh hưởng mạnh mẽ tới độ chính xác model.
- Việc sử dụng thư viện SMOTE để tăng cường dữ liệu ảnh sau đó chia thành 3 tập có thể làm cho dữ liệu bị overfitting làm giảm hiệu quả đánh giá. Chúng tôi sẽ tìm hiểu thêm những cách khác để tăng cường dữ liệu, tuy số lượng dữ liệu quan trọng nhưng chất lượng dữ liệu cũng là một yếu tố ảnh hưởng mạnh mẽ tới độ chính xác model.

**Về thuật toán:**

- Áp dụng các kỹ thuật Ensemble khác thay vì chỉ sử dụng Stacking
- Tăng số lượng base model để tăng độ nhận diện của meta-model
- Tìm kiếm các siêu tham số hiệu quả hơn khi có đủ điều kiện

## 4.2. Hướng phát triển

- Thu thập thêm các ảnh có ngoại cảnh và thực hiện label bằng tay sử dụng **bounding box.**
- Nhận diện biển báo trong thời gian thực và trả về đầu ra dưới dạng âm thanh cho người dùng.
- Nhận biết được biển báo và phân loại được chính xác loại biển báo.

# 5. Video demo

Youtube link

# 6. Nguồn tham khảo

1. Data set: [GTSRB - German Traffic Sign Recognition Benchmark (kaggle.com)](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
2. HOG: https://viblo.asia/p/tim-hieu-ve-phuong-phap-mo-ta-dac-trung-hog-histogram-of-oriented-gradients-V3m5WAwxZO7
3. Mất cân bằng dữ liệu: [Khoa học dữ liệu (phamdinhkhanh.github.io)](https://phamdinhkhanh.github.io/2020/02/17/ImbalancedData.html)
4. GridSearchCV: [Tự học ML | Điều chỉnh siêu tham số SVM bằng GridSearchCV | ML » Cafedev.vn](https://cafedev.vn/tu-hoc-ml-dieu-chinh-sieu-tham-so-svm-bang-gridsearchcv-ml/)
5. SVM: https://machinelearningcoban.com/2017/04/09/smv/
