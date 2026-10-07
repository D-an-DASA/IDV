# Logistic Regression

## Training Process

Quá trình xây dựng mô hình Logistic Regression được thực hiện theo các bước chính:

1. **Chuẩn bị dữ liệu**
   - Tách biến đầu vào `X` và biến mục tiêu `y = HeartDiseaseorAttack`.
   - Chia dữ liệu thành tập `Train` và `Test` theo tỷ lệ 80/20.
   - Sử dụng `stratify=y` để duy trì tỷ lệ giữa hai lớp.
   - Giữ tập `Test` độc lập và chỉ sử dụng ở bước đánh giá cuối cùng.

2. **Tiền xử lý và Scaling**
   - Sử dụng `StandardScaler` để chuẩn hóa các biến đầu vào.
   - Chỉ `fit` scaler trên tập `Train`, sau đó áp dụng cùng scaler cho `Test` nhằm tránh data leakage.

3. **Xây dựng Baseline**
   - Xây dựng Logistic Regression ban đầu với `class_weight=None`.
   - Kết quả cho thấy Accuracy cao nhưng Recall thấp do dữ liệu mất cân bằng mạnh giữa hai lớp.
   - Từ đó, quá trình fine-tuning được tập trung vào việc điều chỉnh `class_weight` nhằm cải thiện khả năng phát hiện lớp Positive.

4. **Hyperparameter Fine-tuning**
   - Sử dụng `GridSearchCV` với 3-fold Cross-Validation.
   - Đánh giá đồng thời bằng F1-score, ROC-AUC và PR-AUC.
   - Thử nghiệm nhiều giá trị của `C` và các chiến lược `class_weight`.
   - Theo dõi chênh lệch giữa Train và Cross-Validation nhằm kiểm tra dấu hiệu overfitting.
   - Kết quả cho thấy `class_weight` có ảnh hưởng lớn hơn nhiều so với `C`.
   - Cấu hình `{0:1, 1:4}` cho kết quả F1 tốt nhất trong phạm vi tìm kiếm.

5. **Threshold Fine-tuning**
   - Sau khi lựa chọn cấu hình Logistic Regression, sử dụng predicted probability trên tập Train để khảo sát nhiều classification threshold.
   - So sánh Precision, Recall và F1-score tại các threshold khác nhau.
   - Do bài toán tập trung vào phát hiện nguy cơ bệnh tim, Recall được ưu tiên thay vì chỉ tối đa hóa F1.
   - Chọn threshold `0.37`, là threshold đạt Recall khoảng 70% trong khi vẫn duy trì mức Precision phù hợp.

6. **Final Test Evaluation**
   - Giữ nguyên cấu hình model và threshold đã được xác định trước đó.
   - Chỉ sử dụng tập Test ở bước cuối để đánh giá.
   - Báo cáo Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC và Log Loss.

---

## Trạng thái hiện tại

### Final Model 1

**Logistic Regression — Class Weight Balanced**

| Parameter      |    Value |
| -------------- | -------: |
| C              |     0.01 |
| Penalty        |       l2 |
| Class Weight   | balanced |
| Max Iterations |     1000 |
| Threshold      |     0.50 |

### Evaluation

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.7534 |
| Precision | 0.2481 |
| Recall    | 0.7970 |
| F1-score  | 0.3785 |
| ROC-AUC   | 0.8470 |
| PR-AUC    | 0.3629 |
| Log Loss  | 0.4948 |

### Conclusion

Model 1 đánh dấu bước cải thiện đầu tiên sau baseline bằng cách sử dụng `class_weight="balanced"` để xử lý vấn đề mất cân bằng lớp. So với baseline, mô hình giảm Accuracy và Precision nhưng cải thiện rất mạnh Recall, từ 12.58% lên 79.70%, cho thấy khả năng phát hiện các trường hợp Positive được cải thiện đáng kể.

Kết quả cho thấy việc xử lý class imbalance có ảnh hưởng lớn đến hành vi phân loại của Logistic Regression. Tuy nhiên, Precision vẫn ở mức thấp và F1-score chỉ đạt 0.3785. Do đó, việc sử dụng `balanced` chưa tạo ra sự cân bằng tối ưu giữa Precision và Recall.

Model 1 được sử dụng làm cơ sở để tiếp tục fine-tune trọng số của lớp Positive thay vì tiếp tục chỉ dựa vào Accuracy.

---

### Final Model 2 — Current

**Logistic Regression — Custom Class Weight**

| Parameter      |        Value |
| -------------- | -----------: |
| C              |          0.3 |
| Penalty        |           l2 |
| Class Weight   | `{0:1, 1:4}` |
| Max Iterations |         1000 |
| Threshold      |         0.37 |

### Evaluation

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.8035 |
| Precision | 0.2827 |
| Recall    | 0.7064 |
| F1-score  | 0.4038 |
| ROC-AUC   | 0.8470 |
| PR-AUC    | 0.3630 |
| Log Loss  | 0.3244 |

### Conclusion

Model 2 tiếp tục fine-tune Logistic Regression bằng cách thay thế `class_weight="balanced"` bằng các trọng số tùy chỉnh cho lớp Positive. Grid Search được mở rộng với nhiều giá trị `C` và các mức trọng số khác nhau cho class 1.

Kết quả tốt nhất đạt được với `class_weight={0:1, 1:4}` và `C=0.3`, với Cross-Validation F1-score đạt khoảng 0.4157. Khoảng cách giữa Train và Validation rất nhỏ, cho thấy mô hình có khả năng tổng quát hóa ổn định và không có dấu hiệu overfitting đáng kể.

Sau bước Grid Search, threshold tiếp tục được fine-tune trên tập Train. Thay vì tối đa hóa F1 một cách độc lập, threshold được lựa chọn với mục tiêu duy trì Recall ở mức ít nhất khoảng 70%, phù hợp hơn với mục tiêu phát hiện các trường hợp có nguy cơ bệnh tim. Threshold `0.37` đạt Recall 70.64% trên tập Test và F1-score 40.38%.

So với Model 1, Model 2 giảm Recall từ 79.70% xuống 70.64% nhưng cải thiện Precision từ 24.81% lên 28.27% và F1-score từ 37.85% lên 40.38%. Accuracy cũng tăng từ 75.34% lên 80.35%.

Do đó, Model 2 được lựa chọn làm **Logistic Regression model cuối cùng**, vì đạt sự cân bằng phù hợp hơn giữa khả năng phát hiện Positive cases và số lượng False Positive trong bối cảnh bài toán dự đoán nguy cơ bệnh tim.
