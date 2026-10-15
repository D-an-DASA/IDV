# Danh mục biến

Các nhóm dưới đây được sắp xếp theo chủ đề của các câu hỏi và biến tính toán trong BRFSS 2015. Mã gốc được giữ để đối chiếu với [Codebook BRFSS 2015](https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf).

## Biến đích

| STT | Tên biến / mã gốc | Ý nghĩa và mã hóa chính | Vai trò / lưu ý |
|---:|---|---|---|
| 1 | HeartDiseaseorAttack (`_MICHD`) | Biến đích nhị phân: 1 = người tham gia từng báo cáo bệnh mạch vành hoặc nhồi máu cơ tim; 0 = không báo cáo hai tình trạng này. | Nhãn cần mô tả/phân loại; tự khai báo, không phải xác nhận chẩn đoán. |

## Tình trạng sức khỏe tổng quát

| STT | Tên biến / mã gốc | Ý nghĩa và mã hóa chính | Vai trò / lưu ý |
|---:|---|---|---|
| 16 | GenHlth (`GENHLTH`) | Tự đánh giá sức khỏe chung theo thứ bậc: 1 = xuất sắc, 2 = rất tốt, 3 = tốt, 4 = trung bình, 5 = kém. | Tóm tắt cảm nhận sức khỏe; giữ tính thứ bậc khi phân tích. |
| 17 | MentHlth (`MENTHLTH`) | Số ngày trong 30 ngày vừa qua sức khỏe tinh thần không tốt do căng thẳng, trầm cảm hoặc vấn đề cảm xúc; khoảng 0–30 ngày, trong đó mã 88 thường chuyển thành 0. | Biến đếm ngày, không phải chẩn đoán sức khỏe tâm thần. |
| 18 | PhysHlth (`PHYSHLTH`) | Số ngày trong 30 ngày vừa qua sức khỏe thể chất không tốt do bệnh tật/chấn thương; khoảng 0–30 ngày, mã 88 thường biểu thị 0 ngày. | Biến đếm phản ánh gánh nặng sức khỏe gần đây. |
| 19 | DiffWalk (`DIFFWALK`) | Khó khăn nghiêm trọng khi đi bộ hoặc leo cầu thang: có/không sau chuẩn hóa. | Chỉ báo hạn chế chức năng, có thể liên hệ tình trạng sức khỏe và khả năng vận động. |

## Bệnh nền và chỉ số sức khoẻ

| STT | Tên biến / mã gốc | Ý nghĩa và mã hóa chính | Vai trò / lưu ý |
|---:|---|---|---|
| 2 | HighBP (`_RFHYPE5`) | Tăng huyết áp theo câu trả lời khảo sát. Sau chuẩn hóa thường biểu diễn 1 = có, 0 = không; mã gốc có mã không biết/từ chối cần loại hoặc xử lý. | Tình trạng nguy cơ/bệnh nền liên quan tim mạch. |
| 3 | HighChol (`TOLDHI2`) | Người tham gia từng được nhân viên y tế cho biết cholesterol máu cao: thường 1 = có, 2 = không ở mã gốc; tái mã hóa về nhị phân sau làm sạch. | Chỉ báo rối loạn lipid máu theo tự khai báo. |
| 5 | BMI (`_BMI5`) | Chỉ số khối cơ thể được CDC tính từ chiều cao và cân nặng tự khai báo. Mã `_BMI5` thường lưu BMI với hai chữ số thập phân ngầm, cần chia 100 để ra đơn vị kg/m². | Biến số liên tục đại diện tình trạng cơ thể; lưu ý sai số tự khai báo. |
| 6 | BMICategory (dẫn xuất) | Phân nhóm từ BMI để trực quan hóa/phân tích: thiếu cân &lt;18,5; bình thường 18,5 đến &lt;25; thừa cân 25 đến &lt;30; béo phì ≥30 kg/m². Cần quy định xử lý giá trị BMI thiếu/ngoài phạm vi. | Biến dẫn xuất dễ diễn giải theo nhóm; không thay thế BMI liên tục trong mọi mô hình. |
| 8 | Stroke (`CVDSTRK3`) | Người tham gia từng được báo đã bị đột quỵ: 1 = có, 0 = không sau chuẩn hóa. | Bệnh sử mạch máu não liên quan đến nguy cơ tim mạch tổng thể. |
| 9 | Diabetes (`DIABETE3`) | Tình trạng tiểu đường được tự khai báo; mã gốc phân biệt có tiểu đường, chỉ tiểu đường thai kỳ, tiền tiểu đường/đường huyết cao, không mắc và một số mã đặc biệt. Không nên gộp các nhóm nếu không nêu quy tắc. | Bệnh nền chuyển hóa; giữ đúng phân biệt nhóm theo notebook/codebook. |

## Hành vi sức khỏe

| STT | Tên biến / mã gốc | Ý nghĩa và mã hóa chính | Vai trò / lưu ý |
|---:|---|---|---|
| 7 | Smoker (`SMOKE100`) | Từng hút ít nhất 100 điếu thuốc trong đời: 1 = có, 0 = không sau chuẩn hóa; không cho biết chắc tình trạng hút hiện tại. | Đại diện tiền sử phơi nhiễm thuốc lá, một hành vi nguy cơ. |
| 10 | PhysActivity (`_TOTINDA`) | Có tham gia hoạt động thể lực ngoài công việc thường lệ trong 30 ngày gần đây hay không; thường chuẩn hóa 1 = có, 0 = không. | Đại diện hành vi vận động; phép đo ở mức có/không, không phản ánh đầy đủ cường độ/thời lượng. |
| 11 | Fruits (`_FRTLT1`) | Chỉ báo ăn trái cây ít nhất một lần mỗi ngày theo biến CDC tính toán; thường nhị phân 1 = đạt ngưỡng, 0 = không đạt. | Đại diện khía cạnh chế độ ăn; không phải phép đo toàn bộ chất lượng dinh dưỡng. |
| 12 | Veggies (`_VEGLT1`) | Chỉ báo ăn rau ít nhất một lần mỗi ngày theo biến CDC tính toán; thường nhị phân 1 = đạt ngưỡng, 0 = không đạt. | Đại diện khía cạnh chế độ ăn, với giới hạn về độ chi tiết tần suất/khẩu phần. |
| 13 | HvyAlcoholConsump (`_RFDRHV5`) | Biến CDC tính toán về mức uống rượu bia nặng theo ngưỡng giới tính và số lượng đồ uống/tuần; thường 1 = đạt định nghĩa uống nặng, 0 = không. | Chỉ báo hành vi sử dụng rượu; cần diễn giải theo định nghĩa BRFSS, không phải chẩn đoán rối loạn sử dụng rượu. |

## Tiếp cận và sử dụng dịch vụ y tế

| STT | Tên biến / mã gốc | Ý nghĩa và mã hóa chính | Vai trò / lưu ý |
|---:|---|---|---|
| 4 | CholCheck (`_CHOLCHK`) | Thời điểm xét nghiệm cholesterol gần nhất; thường phân biệt đã kiểm tra trong 5 năm với chưa kiểm tra/kiểm tra quá 5 năm/không bao giờ. Mã không biết hoặc từ chối cần xử lý riêng. | Chỉ báo theo dõi và sử dụng dịch vụ dự phòng; không đồng nhất với cholesterol cao. |
| 14 | AnyHealthcare (`HLTHPLN1`) | Có bất kỳ hình thức bảo hiểm/bao phủ chăm sóc sức khỏe nào, gồm các hình thức trong định nghĩa câu hỏi: 1 = có, 0 = không sau chuẩn hóa. | Đại diện khả năng tiếp cận hệ thống y tế. |
| 15 | NoDocbcCost (`MEDCOST`) | Trong 12 tháng qua, có thời điểm không đi khám vì chi phí hay không: thường 1 = có trở ngại chi phí, 0 = không. | Đo rào cản tài chính đối với việc sử dụng dịch vụ y tế. |

## Nhân khẩu học và kinh tế – xã hội

| STT | Tên biến / mã gốc | Ý nghĩa và mã hóa chính | Vai trò / lưu ý |
|---:|---|---|---|
| 20 | Sex (`SEX`) | Giới tính theo lựa chọn trong bảng hỏi BRFSS; mã thường gặp 1 = nam, 2 = nữ. | Biến phân loại nhân khẩu học; cần dùng nhãn thay vì hiểu mã số như đại lượng. |
| 21 | Age (`_AGEG5YR`) | Nhóm tuổi do CDC tạo theo khoảng tuổi, thường bắt đầu 18–24, tiếp theo các khoảng 5 năm và nhóm cuối 80 tuổi trở lên; mã 14 thường biểu thị thiếu/không xác định. | Biến thứ bậc quan trọng để mô tả khác biệt nguy cơ theo tuổi. |
| 22 | Education (`EDUCA`) | Trình độ học vấn cao nhất theo các mức từ chưa học/lớp tiểu học đến tốt nghiệp đại học trở lên; mã 9 thường là từ chối. | Biến thứ bậc đại diện một khía cạnh điều kiện kinh tế – xã hội. |
| 23 | Income (`INCOME2`) | Nhóm thu nhập hộ gia đình hằng năm theo các khoảng tăng dần; mã 77/99 thường chỉ “không biết”/“từ chối” và không phải mức thu nhập. | Biến thứ bậc kinh tế – xã hội; cần xử lý mã đặc biệt như thiếu. |

## Địa lý và thông tin phỏng vấn

| STT | Tên biến / mã gốc | Ý nghĩa và mã hóa chính | Vai trò / lưu ý |
|---:|---|---|---|
| 24 | State (`_STATE`) | Mã FIPS của bang/vùng khảo sát. Mã chỉ là nhãn địa lý, không phải thang đo liên tục. | Hữu ích cho phân tích phân tầng/địa lý; chỉ dùng làm predictor khi mục tiêu có lý do rõ. |
| 25 | InterviewDate (`IDATE`) | Ngày phỏng vấn ở dạng mã ngày/tháng/năm trong tệp BRFSS; có thể chuyển sang kiểu ngày nếu cần. | Thông tin hành chính/thời điểm thu thập, không mặc nhiên là nguy cơ cá nhân. |
| 26 | InterviewMonth (`IMONTH`) | Tháng thực hiện phỏng vấn, lưu theo mã tháng. | Có thể hỗ trợ phân tích mùa vụ/thời điểm; cần chứng minh liên quan trước khi đưa vào dự đoán. |
| 27 | InterviewDay (`IDAY`) | Ngày trong tháng thực hiện phỏng vấn, thường từ 1 đến 31. | Thông tin thời điểm thu thập; thường chỉ dùng kiểm tra hoặc phân tích dữ liệu. |
| 28 | InterviewYear (`IYEAR`) | Năm phỏng vấn; trong tập năm 2015, giá trị dự kiến cố định hoặc gần như cố định. | Thường không có ích cho phân biệt trong một năm dữ liệu; cân nhắc loại khỏi mô hình. |
