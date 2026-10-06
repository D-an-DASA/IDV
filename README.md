# Báo cáo quá trình huấn luyện và trạng thái mô hình

## 1. Mục tiêu

Mục tiêu của mô hình là xây dựng một mô hình phân loại nhị phân sử dụng **XGBoost** để dự đoán `HeartDiseaseorAttack` dựa trên 21 đặc trưng về nhân khẩu học, sức khỏe, lối sống và khả năng tiếp cận dịch vụ y tế.

Do dữ liệu có sự mất cân bằng lớp lớn, trong đó nhóm `0` chiếm khoảng 91.5% và nhóm `1` khoảng 8.5%, quá trình đánh giá không chỉ dựa trên Accuracy mà tập trung vào **Precision, Recall, F1-score và ROC-AUC**.

---

## 2. Quy trình training

Quá trình xây dựng mô hình được thực hiện theo các giai đoạn:

```text
Train/Test Split
      ↓
Baseline XGBoost
      ↓
Hyperparameter Tuning
      ↓
Complexity Analysis
      ↓
Regularization Tuning
      ↓
OOF Evaluation
      ↓
Threshold Tuning
      ↓
Final Training
      ↓
Final Test Evaluation
```

Dữ liệu được chia thành:

- `X_train`: 236,600 mẫu
- `X_test`: 59,151 mẫu
- 21 features
- Target: `HeartDiseaseorAttack`
- `random_state = 42`
- Stratified train/test split

`X_test` được giữ riêng và không sử dụng trong quá trình lựa chọn hyperparameter hoặc threshold.

---

## 3. Hyperparameter tuning

Ban đầu, mô hình được thử nghiệm với các tổ hợp `learning_rate`, `max_depth` và `n_estimators` bằng GridSearchCV.

Sau đó, RandomizedSearchCV được sử dụng để mở rộng không gian tìm kiếm, bao gồm cả:

- `subsample`
- `colsample_bytree`
- `min_child_weight`
- `gamma`
- `reg_alpha`
- `reg_lambda`

RandomizedSearch tìm được một cấu hình tốt với F1 CV khoảng **0.202**. Từ cấu hình này, các vòng thử nghiệm tiếp tục tập trung vào độ phức tạp và regularization.

---

## 4. Phân tích độ phức tạp

Các vòng thử nghiệm cho thấy khi tăng độ phức tạp của mô hình, F1 trên training tăng rất mạnh trong khi F1 trên Cross-Validation chỉ tăng tương đối nhỏ.

Ví dụ:

```text
Learning rate = 0.20
Max depth     = 7
Trees         = 350

Train F1 ≈ 0.446
CV F1    ≈ 0.210
```

Trong các cấu hình phức tạp hơn:

```text
Learning rate = 0.28
Max depth     = 9
Trees         = 375

Train F1 ≈ 0.815
CV F1    ≈ 0.233
```

Khoảng cách lớn giữa Training F1 và CV F1 cho thấy mô hình có dấu hiệu **overfitting** khi độ phức tạp tăng.

Tuy nhiên, việc tăng complexity vẫn giúp cải thiện CV F1 trong một phạm vi nhất định. Do đó, quá trình tuning không tiếp tục tăng complexity một cách không kiểm soát mà chuyển sang regularization.

---

## 5. Regularization tuning

Các tham số regularization được thử nghiệm lần lượt.

Kết quả tốt nhất:

| Parameter          |  Giá trị |
| ------------------ | -------: |
| `min_child_weight` |    **5** |
| `gamma`            | **0.05** |
| `reg_lambda`       |    **2** |
| `reg_alpha`        |    **0** |

Cấu hình này giúp giảm đáng kể mức độ overfitting so với cấu hình ban đầu, đồng thời cải thiện F1 trên Cross-Validation.

F1 CV của candidate đạt khoảng:

> **0.2337**

---

## 6. Candidate model hiện tại

Cấu hình được lựa chọn:

```python
XGBClassifier(
    learning_rate=0.28,
    max_depth=9,
    n_estimators=350,
    subsample=0.9,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.05,
    reg_alpha=0,
    reg_lambda=2,
    random_state=42,
    eval_metric="logloss"
)
```

Đây được xem là **Final Model Candidate**.

Mặc dù `max_depth=9` và `learning_rate=0.28` tương đối cao, mô hình được giữ lại vì kết quả trên dữ liệu unseen vẫn ổn định và không cho thấy sự sụt giảm mạnh so với OOF evaluation.

---

## 7. OOF Evaluation và Threshold Tuning

Sau khi lựa chọn candidate, sử dụng **Out-of-Fold predictions** trên training set để đánh giá khả năng tổng quát hóa và lựa chọn threshold.

Ở threshold mặc định `0.5`:

```text
Precision ≈ 0.407
Recall    ≈ 0.164
F1        ≈ 0.234
```

Do dữ liệu mất cân bằng, threshold `0.5` cho Recall khá thấp.

Threshold được thử nghiệm trên OOF predictions và kết quả tốt nhất theo F1 là:

> **Threshold = 0.15**

Tại threshold này:

```text
Precision ≈ 0.276
Recall    ≈ 0.514
F1        ≈ 0.359
```

Threshold được lựa chọn hoàn toàn từ OOF data, không sử dụng `X_test`.

---

## 8. Final Test Evaluation

Sau khi hoàn thành model và threshold selection, mô hình được train lại trên toàn bộ training set và chỉ sau đó mới đánh giá trên `X_test`.

Kết quả:

| Metric    | Final Test |
| --------- | ---------: |
| Accuracy  | **0.8430** |
| Precision | **0.2813** |
| Recall    | **0.5500** |
| F1-score  | **0.3723** |
| ROC-AUC   | **0.8232** |

Confusion Matrix:

```text
                 Predicted
                 0       1
Actual 0      47114    7032
Actual 1       2252    2753
```

Mô hình phát hiện được **2,753 / 5,005** trường hợp thuộc class 1, tương ứng Recall khoảng **55.0%**.

---

## 9. So sánh OOF và Test

| Metric    |   OOF |      Test |
| --------- | ----: | --------: |
| Precision | 0.276 | **0.281** |
| Recall    | 0.514 | **0.550** |
| F1        | 0.359 | **0.372** |
| ROC-AUC   | 0.815 | **0.823** |

Kết quả Test không giảm so với OOF mà còn cao hơn nhẹ ở các metric chính.

Điều này cho thấy candidate có khả năng **generalize tương đối ổn trên dữ liệu chưa từng được sử dụng trong quá trình tuning**.

---

## 10. Trạng thái hiện tại

### 🟢 Model: ACCEPTED AS FINAL CANDIDATE

Mô hình hiện tại được chấp nhận làm **Final Model** của project dựa trên:

- Hyperparameter đã được tìm kiếm có hệ thống.
- Regularization đã được kiểm tra.
- OOF evaluation được sử dụng để đánh giá generalization.
- Threshold được lựa chọn trên OOF, không dùng test set.
- Final test set được giữ độc lập cho đến bước đánh giá cuối.
- Kết quả OOF và Test tương đối ổn định.

Tuy nhiên, mô hình vẫn có Precision tương đối thấp và có dấu hiệu complexity cao. Vì vậy, trong báo cáo cần trình bày rõ rằng mô hình được sử dụng cho **phân tích và ước lượng xác suất/risk score**, không phải công cụ chẩn đoán y khoa.

### Thông số cần đóng băng

```text
Model:
XGBoost

learning_rate = 0.28
max_depth = 9
n_estimators = 350
subsample = 0.9
colsample_bytree = 0.8
min_child_weight = 5
gamma = 0.05
reg_alpha = 0
reg_lambda = 2
random_state = 42

Decision threshold = 0.15

Final Test:
Accuracy  = 0.8430
Precision = 0.2813
Recall    = 0.5500
F1        = 0.3723
ROC-AUC   = 0.8232
```

Từ thời điểm này, các thông số trên được xem là **frozen configuration**. `X_test` không tiếp tục được sử dụng để tuning model hoặc threshold.
