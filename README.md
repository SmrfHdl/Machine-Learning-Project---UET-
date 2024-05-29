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

# **Bảng phân chia nhiệm vụ**
| Nhiệm vụ  	                          | Người phụ trách                               | 
| --------------------------------------- | --------------------------------------------- | 
|Tạo và update repo	                      |Phạm Văn Trường                                |
|Thu thập và tiền xử lý dữ liệu	          |Nguyễn Viết Vũ                                 |
|Lựa chọn và triển khai mô hình           |Phạm Văn Trường, Nguyễn Viết Vũ, Trần An Thắng |  
|Đánh giá hiệu suất và tối ưu hóa mô hình |Trần An Thắng                                  |
|Training và đánh giá meta-model          |Phạm Văn Trường	                              |
|VIết báo cáo                             |Nguyễn Viết Vũ	                              |

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


Resize ảnh thành kích thước 128x128 pixels nhằm giúp cho đặc trưng ảnh thu được không quá ít, đồng thời cũng không tạo ra quá nhiều tham số giảm thời gian tải dữ liệu và huấn luyện mô hình.

### 2.2.2. Đọc dữ liệu và trích xuất đặc trưng

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

Vấn đề dữ liệu không cân bằng có thể được giải quyết với các mô hình Ensemble như Random Forest, tuy nhiên để có thể sử dụng mô hình SVM thì tôi sẽ cần phải sử dụng kỹ thuật Resampling bằng SMOTE của thư viện ‘imblearn’. SMOTE (Synthetic Minority Over-sampling) và ADASYN (Adaptive synthetic sampling) là các phương pháp sinh mẫu nhằm gia tăng kích thước mẫu của nhóm thiểu số trong trường hợp xảy ra mất cân bằng mẫu. Để gia tăng kích thước mẫu, với mỗi một mẫu thuộc nhóm thiểu số ta sẽ lựa chọn ra 𝑘 mẫu láng giềng gần nhất với nó và sau đó thực hiện tổ hợp tuyến tính để tạo ra mẫu giả lập.

Sau khi tính toán giá trị trung bình là khoảng 1000 dữ liệu trên mỗi nhãn, tôi tiến hành tạo thêm dữ liệu để các nhãn có số dữ liệu quá thấp đạt tới 1000 dữ liệu.

Kết quả cho thấy hàm đã hoạt động tốt và cho ra tập dữ liệu khá cân bằng.

### 2.2.4. Chia tập dữ liệu

Tiếp theo để có thể đánh giá và tối ưu hóa mô hình tôi sẽ tiến hành chia tập dữ liệu thành 3 phần train - val - test với tỉ lệ 70% - 10% - 15%

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
```

Trong báo cáo phân loại của mô hình SVM, chúng ta thấy rằng precision, recall và f1-score đều cao cho hầu hết các lớp. Điều này cho thấy mô hình SVM có khả năng phân loại tốt trên cả các lớp lớn và nhỏ. Precision và recall gần như đều cao cho tất cả các lớp, chỉ có một số lớp có một vài điểm số thấp hơn, nhưng vẫn đạt được hiệu suất tốt (ví dụ: lớp 6, lớp 21, lớp 22).

Sở dĩ mô hình SVM đạt được độ chính xác xao như vậy là vì mô hình này đặc biệt hoạt động tốt trên các tập dữ liệu nhiều chiều như đặc trưng HOG.

### 3.2.2. **Đánh giá mô hình Random Forest**

```python
Đánh giá mô hình Random Forest trên dữ liệu validation:
Accuracy: 0.9346793349168646
```

Bảng đánh giá cho thấy mô hình có hiệu suất tốt trên hầu hết các lớp, với precision, recall và F1-score đều ổn định. Random Forest hoạt động tốt trên dữ liệu HOG vì khả năng xử lý không gian đặc trưng lớn, dễ dàng ánh xạ các mẫu dữ liệu và không bị ảnh hưởng bởi biến đổi không quan trọng của dữ liệu.

### 3.2.3. **Đánh giá mô hình KNN**

```python
Đánh giá mô hình KNN trên dữ liệu validation:
Accuracy: 0.66270783847981
```

- Tại biển báo nhãn 0 (Speed Limit 20) có chỉ số precision rất thấp (9%) điều này xảy ra là do hầu hết các biển báo speed limit có hình dáng khá tương đồng, chỉ thay đổi mỗi con số ở trong biển báo. Việc sử dụng HOG trích xuất đặc trưng ảnh với tỉ lệ ảnh là 128\*128px cũng sẽ làm cho những con số trong biển báo trở nên khó phân biệt hơn với mô hình KNN, chính vì vậy mà tỉ lệ precision của biển nhãn 0 rất thấp và tỉ lệ recall lại cao hơn so với các biển speed limit khác (từ 1 - 8). Ngoài ra số support cũng thấp hơn so với các biển còn lại cũng dẫn đến việc chỉ số recall cao.
- Điều này cũng xảy ra tương tự với biển số 11 và biển số 27 khi hình dáng khá tương tự nhau:

Những ảnh có hình dáng tương tự nhau sẽ gây ra sự nhầm lẫn lớn đối với mô hình KNN

Nhìn chung việc sử dụng mô hình KNN để đánh giá với các tập dữ liệu lớn và có độ phức tạp cao về cả dữ liệu và không gian đặc trưng sẽ đem lại hiệu quả rất kém. Với không gian với 8100 chiều thì các mẫu sẽ trở nên xa nhau trong không gian đặc trưng, làm cho việc tìm láng giềng gần nhất trở nên khó khăn và không chính xác.

## 3.3. Tối ưu model

Sau quá trình đánh giá thì tôi đã quyết định chọn 2 model để tiến hành stacking sử dụng Logistic Regrestion để chọn kết quả tốt nhất. Trước đó tôi sẽ tiến hành tối ưu model SVM sử dụng phương pháp Grid Search:

### 3.3.1. Tối ưu model SVM

Tôi sẽ sử dụng Grid Search để tìm kiếm siêu tham số tối ưu cho mô hình SVM. Cụ thể, chúng ta đã xác định một danh sách các giá trị cho tham số "C", đây là tham số quan trọng trong SVM.

Sau đó, tôi sử dụng GridSearchCV để thử tất cả các giá trị trong danh sách đó và đánh giá hiệu suất của mô hình SVM với mỗi giá trị của "C" trên tập dữ liệu huấn luyện. Bằng cách này, tôi có thể tìm ra giá trị tối ưu cho "C" mà cải thiện hiệu suất của mô hình.

Sau khi thử với các giá trị C khác nhau từ $10^{-3}$ cho đến 10 thì giá trị tốt nhất là 0.01 và tôi tiếp tục thu hẹp phạm vi và nhận thấy giá trị C mặc định vẫn là 0.01 sẽ cho ra độ chính xác cao nhất.

Vì thế tôi sẽ giữ nguyên siêu tham số này để tiến hành stacking

### 3.3.2. Training cho **meta-model**

Quy trình:

Do tài nguyên của máy tôi không đủ để thực hiện gridsearch tìm siêu tham số tối ưu nhất với mô hình Random Forest nên tôi sẽ tiến hành luôn bước huấn luyện cho meta model

Đầu tiên tôi khởi tạo 2 mô hình cơ sở hiệu quả nhất trong 3 mô hình tôi đã đánh giá trước đó. Sau đó tạo mô một ma trận đặc trưng làm đầu vào huấn luyện cho meta-model.

## 3.4. Đánh giá meta-model

### 3.4.1. Accuracy

```python
Accuracy of svm model: 0.942596991290578
Accuracy of random forest model: 0.9338875692794932
Accuracy of stacked model: 0.9451702296120348
```

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
