# IDV:

# XGBoost:

## Training Process

Quá trình xây dựng mô hình dự đoán được thực hiện theo các bước chính:

1. **Chuẩn bị dữ liệu**
   - Tách biến đầu vào `X` và biến mục tiêu `y`.
   - Chia dữ liệu thành tập `Train` và `Test`.
   - Giữ tập `Test` độc lập, không sử dụng trong quá trình tuning.

2. **Xây dựng mô hình ban đầu**
   - Sử dụng XGBoost cho bài toán phân loại nhị phân.
   - Thiết lập các hyperparameter ban đầu để xây dựng baseline.

3. **Hyperparameter Fine-tuning**
   - Thử nghiệm nhiều cấu hình của XGBoost.
   - Sử dụng Cross-Validation để đánh giá khả năng tổng quát hóa.
   - Theo dõi các metric như ROC-AUC và PR-AUC.
   - Kiểm tra sự chênh lệch giữa kết quả trên tập Train và Cross-Validation nhằm phát hiện overfitting.
   - Chọn cấu hình mô hình phù hợp nhất dựa trên hiệu năng và khả năng tổng quát hóa.

4. **Threshold Fine-tuning**
   - Sử dụng dự đoán Out-of-Fold (OOF) trên tập Train để đánh giá nhiều classification threshold.
   - So sánh Precision, Recall và F1-score ở các threshold khác nhau.
   - Chọn threshold phù hợp với mục tiêu cân bằng giữa khả năng phát hiện Positive và số lượng False Positive.

5. **Final Test Evaluation**
   - Retrain mô hình với cấu hình đã chọn trên toàn bộ tập Train.
   - Áp dụng threshold đã được xác định từ OOF.
   - Chỉ sử dụng tập Test ở bước cuối để đánh giá mô hình.
   - Báo cáo các metric cuối cùng: Accuracy, Precision, Recall, F1-score, ROC-AUC và PR-AUC.

---

## Trạng thái hiện tại

### Final Model 1

**XGBoost**

| Parameter                        | Value |
| -------------------------------- | ----: |
| Learning Rate                    |  0.28 |
| Number of Estimators             |   350 |
| Max Depth                        |     9 |
| Subsample                        |   0.9 |
| Column Sample by Tree            |   0.8 |
| Min Child Weight                 |     5 |
| Gamma                            |  0.05 |
| L1 Regularization (`reg_alpha`)  |     0 |
| L2 Regularization (`reg_lambda`) |     2 |
| Threshold                        |  0.15 |

### Evaluation

| Metric    |  Score |
| --------- | -----: |
| PR-AUC    |      — |
| ROC-AUC   | 0.8232 |
| Precision | 0.2813 |
| Recall    | 0.5500 |
| F1-score  | 0.3723 |
| Accuracy  | 0.8430 |

### Conclusion

Do là model đầu tiên, quá trình training còn mang tính thử nghiệm và chiến lược fine-tuning tương đối hạn chế, chủ yếu sử dụng F1-score làm tiêu chí đánh giá. Mô hình đạt kết quả Accuracy khá cao nhưng có khoảng cách lớn giữa khả năng học trên tập train và khả năng tổng quát hóa. Việc sử dụng max_depth=9 cùng các thiết lập độ phức tạp cao khiến mô hình có dấu hiệu overfitting, cho thấy Accuracy/F1 trên một tập dữ liệu chưa đủ để đánh giá toàn diện chất lượng mô hình trong bối cảnh dữ liệu mất cân bằng.

### Final Model 2

**XGBoost**

| Parameter                        | Value |
| -------------------------------- | ----: |
| Learning Rate                    |  0.28 |
| Number of Estimators             |   450 |
| Max Depth                        |     6 |
| Subsample                        |   0.9 |
| Column Sample by Tree            |   0.8 |
| Min Child Weight                 |     5 |
| Gamma                            |  0.05 |
| L1 Regularization (`reg_alpha`)  |     0 |
| L2 Regularization (`reg_lambda`) |     2 |
| Threshold                        |  0.20 |

### Evaluation

| Metric    |  Score |
| --------- | -----: |
| PR-AUC    | 0.3324 |
| ROC-AUC   | 0.8450 |
| Precision | 0.3234 |
| Recall    | 0.5309 |
| F1-score  | 0.4020 |
| Accuracy  | 0.8663 |

### Conclusion

Ở Model 2, chiến lược training được cải thiện bằng cách chuyển sang Average Precision / PR-AUC làm tiêu chí chính, phù hợp hơn với bài toán phân loại mất cân bằng. Đồng thời, max_depth được điều chỉnh theo hướng giảm độ phức tạp của cây nhằm cải thiện khả năng generalization. Tuy nhiên, quá trình fine-tuning lúc này vẫn còn hạn chế do một số giá trị tốt nhất nằm ở boundary của grid search nhưng chưa được nhận diện và kiểm tra đầy đủ. Vì vậy, Model 2 đã cải thiện đáng kể so với Model 1 nhưng chưa thực sự xác định được optimum của hyperparameter.

### Final Model 3

**XGBoost**

| Parameter                        | Value |
| -------------------------------- | ----: |
| Learning Rate                    |  0.15 |
| Number of Estimators             |   300 |
| Max Depth                        |     6 |
| Subsample                        |   0.9 |
| Column Sample by Tree            |   0.8 |
| Min Child Weight                 |     5 |
| Gamma                            |  0.05 |
| L1 Regularization (`reg_alpha`)  |     0 |
| L2 Regularization (`reg_lambda`) |     2 |
| Threshold                        |  0.20 |

### Evaluation

| Metric    |  Score |
| --------- | -----: |
| PR-AUC    | 0.3623 |
| ROC-AUC   | 0.8577 |
| Precision | 0.3320 |
| Recall    | 0.5546 |
| F1-score  | 0.4154 |
| Accuracy  | 0.8679 |

### Conclusion

Model 3 tiếp tục sử dụng PR-AUC làm tiêu chí chính và tập trung fine-tune cặp learning_rate và n_estimators để tìm sự cân bằng giữa tốc độ học và số lượng cây. Kết quả cho thấy mô hình cải thiện rõ rệt về PR-AUC và ROC-AUC so với Model 2. Tuy nhiên, tương tự Model 2, quá trình tìm kiếm vẫn chưa xử lý triệt để vấn đề boundary của search space, đặc biệt ở các hyperparameter liên quan đến độ phức tạp và regularization. Điều này khiến một số giá trị được xem là tốt nhất khi thực tế quá trình tìm kiếm vẫn có khả năng tiếp tục cải thiện.

# Final Model 4 — Current

**XGBoost**

| Parameter                        | Value |
| -------------------------------- | ----: |
| Learning Rate                    |  0.10 |
| Number of Estimators             |   450 |
| Max Depth                        |     3 |
| Subsample                        |   0.9 |
| Column Sample by Tree            |   0.8 |
| Min Child Weight                 |   400 |
| Gamma                            |  0.10 |
| L1 Regularization (`reg_alpha`)  |     0 |
| L2 Regularization (`reg_lambda`) |     2 |
| Threshold                        |  0.20 |

### Evaluation

| Metric    |  Score |
| --------- | -----: |
| --------- |  ----: |
| PR-AUC    | 0.3771 |
| ROC-AUC   | 0.8616 |
| Precision | 0.3357 |
| Recall    | 0.5514 |
| F1-score  | 0.4173 |
| Accuracy  | 0.8697 |

### Conclusion

Model 4 đánh dấu sự thay đổi chính trong training strategy. Sau khi nhận diện hạn chế từ Model 2 và Model 3, quá trình fine-tuning được thực hiện theo hướng coarse-to-fine, trước tiên mở rộng phạm vi tìm kiếm để xác định xu hướng và vùng optimum, sau đó thu hẹp dần quanh vùng có kết quả tốt.

Đặc biệt, min_child_weight và gamma được fine-tune qua nhiều vòng để theo dõi đồng thời CV PR-AUC, Train PR-AUC và Train–CV gap. Kết quả cho thấy min_child_weight=400 và gamma=0.10 đạt mức cân bằng hợp lý giữa khả năng học và generalization; sau đó mức cải thiện của CV PR-AUC bắt đầu giảm đáng kể, cho thấy mô hình đang tiến vào vùng diminishing returns / saturation.

Cuối cùng, learning_rate, n_estimators và max_depth được fine-tune lại trong bối cảnh regularization mới. Kết quả cuối cùng cho thấy cấu hình đơn giản hơn với max_depth=3, learning_rate=0.10 và n_estimators=450 đạt khả năng tổng quát hóa tốt nhất trong phạm vi đã khảo sát. Model 4 đạt PR-AUC 0.3771, ROC-AUC 0.8616 và F1-score 0.4173, trở thành mô hình được lựa chọn cuối cùng.
