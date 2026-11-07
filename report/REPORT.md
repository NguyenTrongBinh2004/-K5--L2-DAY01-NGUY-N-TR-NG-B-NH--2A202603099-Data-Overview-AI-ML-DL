# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 6/11/2026

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**3.13.18/2.11.0+cpu/8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Là cho cả bức ảnh. 
Vì sao bạn nghĩ vậy? 
Trong JSON không có tọa độ nào (không có bbox hay polygon), nên record không chỉ ra vị trí của vật nào.
Mỗi ảnh chỉ có một danh sách xếp hạng (rank 1 đến 5), mô hình đang trả lời “cả ảnh này giống lớp nào nhất”
Ảnh traffic có nhiều xe, nhưng cab không cho biết xe nào là cab.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint? Đi kèm taxonomy và checkpoint, không phải mô hình tự nghĩ ra.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp? Để nhãn không bị hiểu mơ hồ.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp? Guideline phải có quy tắc chọn đúng một lớp.
6. Vì sao không được lấy `score` của model làm ground truth? Score là độ tự tin của mô hình, không phải sự thật.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=bus ; `score`=0.912557 ; `bbox_xyxy`=[93.17, 187.95, 223.01, 320.91] ; `bbox_width`=129.84 ; `bbox_height`=132.96
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Box được xác định bởi góc trên-trái (x_min, y_min) và góc dưới-phải (x_max, y_max) tính bằng pixel. Gốc tọa độ (0,0) ở góc trên-trái ảnh, trục X hướng sang phải, trục Y hướng xuống dưới. Với ảnh kích thước 640x428, chiều rộng box là (x_max - x_min) và chiều cao là (y_max - y_min) pixel, tất cả tọa độ đều nằm trong giới hạn của ảnh.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
Mình chọn hai ngưỡng là 0.20 và 0.60.
Ở ngưỡng 0.20, mô hình đưa ra 17 predictions.
Ở ngưỡng 0.60, mô hình đưa ra 6 predictions.
Có thể thấy ngưỡng (threshold) càng thấp thì số lượng hộp dự đoán (predictions) càng nhiều, và ngược lại.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Hạ ngưỡng xuống 0.20 sinh ra nhiều box rác (potted plant, spoon), reviewer phải tốn công check thủ công từng hộp để xóa False Positive. Ngược lại, nâng lên 0.60 đỡ phải check box rác nhưng lại dễ bỏ sót vật thật (False Negative như mất cup, bowl).
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Mỗi vật thể thuộc danh sách lớp được gán đúng một hộp, ôm sát phần nhìn thấy của vật: cạnh hộp chạm vào mép ngoài cùng của vật ở cả bốn phía, không chừa viền thừa và không cắt mất phần nào của vật. Nếu vật bị che khuất hoặc bị cắt ở mép ảnh, chỉ vẽ hộp quanh phần nhìn thấy, không đoán phần bị che.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Trên ảnh kitchen, hộp person 0.61 ở góc dưới bên trái chỉ chứa bàn tay người đang thao tác trên bàn bếp. Vật này bị cắt bởi mép ảnh và chồng lên các hộp oven và bowl, nên chỉ thấy một phần rất nhỏ của người.

Mình chọn đưa lên hỏi Lab Coach (escalation) thay vì tự viết thành quy tắc, vì guideline hiện chưa nói rõ khi chỉ thấy một bộ phận cơ thể (như bàn tay) thì có được gán là person hay không. Đây là quyết định về định nghĩa nhãn, có nhiều cách hiểu hợp lý: gán mọi phần cơ thể, chỉ gán khi thấy đủ đầu hoặc thân, hoặc dùng lớp riêng cho bộ phận cơ thể (COCO-80 không có lớp này). Nếu mỗi annotator tự quyết, cùng một tình huống sẽ có người gán, người bỏ, làm nhãn thiếu nhất quán.
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=traffic-002 ; `class_name`=car ; `score`=0.891092 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[446, 70], [445, 71], [444, 71], [443, 72], [442, 72]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Bounding box (box) chỉ xác định vùng hình chữ nhật bao quanh vật thể, không thể hiện chính xác đường biên của vật thể.
Polygon gồm nhiều điểm tọa độ [x, y] nối với nhau để mô tả đường viền thực tế của từng đối tượng. Nhờ đó, polygon thể hiện được hình dạng, kích thước và những phần lồi lõm của vật thể chính xác hơn box.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
instance_id dùng để định danh riêng từng đối tượng được phát hiện trong một ảnh.
Instance ID không phải là class ID (vốn dùng để xác định loại đối tượng như bowl, person, spoon) và cũng không phải là tracking ID (dùng để theo dõi cùng một đối tượng xuyên suốt qua nhiều khung hình video). Instance ID chỉ đơn thuần là mã dùng để phân biệt từng đối tượng riêng biệt xuất hiện trong kết quả của bài thực hành, ví dụ như kitchen-001, kitchen-002.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Quy tắc vẽ biên mask: đường biên phải đi sát đường viền ngoài của phần nhìn thấy của vật, không lấn sang nền hoặc vật khác và không bỏ sót phần rìa của vật. Khi hai vật chạm hoặc chồng nhau, vật ở phía trước giữ nguyên đường biên, vật ở phía sau chỉ lấy phần nhìn thấy, và một điểm ảnh không được thuộc hai mask cùng lúc.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Trên ảnh kitchen, ở góc dưới bên trái có ba bowl nằm sát nhau và chồng lên nhau (hai bát có điểm 0.70 và 0.74, bát còn lại ở giữa bị nhãn chữ che mất điểm số). Hộp của bát phía dưới (0.74) đè lên phần dưới của bát ở giữa, hộp của bát phía trên (0.70) chạm vào phần trên của bát ở giữa. Cả ba bát lại nằm trên mặt bàn (dining table), nên đường biên giữa các bát với nhau và giữa bát với mặt bàn đều không rõ ràng.

Mình dùng guideline cho phần nhìn thấy rõ: theo quy tắc ở câu 4, bát phía trước giữ nguyên đường biên, bát phía sau chỉ lấy phần nhìn thấy, và một điểm ảnh không được thuộc hai mask cùng lúc. Với đoạn biên bị che hoặc mờ đến mức không xác định được (ví dụ chỗ bát ở giữa tiếp giáp với bát bên dưới), mình sẽ escalate hỏi Lab Coach thay vì tự đoán. Lý do là nếu mỗi annotator tự đoán một kiểu thì cùng một tình huống sẽ cho ra các mask khác nhau, làm nhãn thiếu nhất quán.
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |Một nhãn lớp cho cả bức ảnh, chọn từ danh sách lớp cố định (taxonomy).  | Ảnh traffic có nhiều xe buýt và ô tô nhưng top-1 là cab (score 0.51), top-5 toàn là loại xe, không lớp nào khớp hẳn. Ảnh COCO nhưng taxonomy ImageNet. |Chọn một lớp theo quy tắc của guideline (ví dụ vật nổi bật nhất). Không có lớp phù hợp thì escalate, không tự chọn bừa.  | Nhãn có đúng quy tắc chọn lớp không, hai annotator có chọn giống nhau không. Không lấy score làm đáp án. |
| Phát hiện vật thể | Mỗi vật thể một lớp + một hộp bbox_xyxy (pixel). | Ảnh kitchen: person 0.61 chỉ chứa bàn tay, bị cắt ở mép ảnh. Đổi ngưỡng từ 0.35 xuống 0.20 thì từ 11 lên 17 hộp (thêm spoon, potted plant, bottle...), nhiều hộp dễ sai. Nồi, chảo không có trong COCO-80 nên không được phát hiện. | Vẽ hộp ôm sát phần nhìn thấy, gán đủ vật thuộc taxonomy theo guideline dù model tự tin hay không. Ca mơ hồ (chỉ thấy bàn tay) thì hỏi Lab Coach. | Hộp có sát không, đúng lớp không, có sót vật nào không, có hộp thừa (false positive) không. |
| Instance segmentation |Mỗi vật thể một lớp + một đa giác polygon_xy bám biên, mỗi instance có instance_id riêng.  | Ảnh kitchen: ba bowl sát và chồng nhau ở góc dưới-trái, biên với nhau và với dining table không rõ. Mask của potted plant 0.63 thực chất bám lên vật treo trông như bó thảo mộc hoặc lá khô, chỉ là gần đúng. | Vẽ biên sát phần nhìn thấy, vật phía trước giữ biên, vật phía sau lấy phần nhìn thấy, không điểm ảnh nào thuộc hai mask. Đoạn biên không xác định được thì escalate. |Biên có bám đúng viền không, có lấn sang vật khác không, các instance cùng lớp có tách riêng không.  |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Chỉ dùng ba ảnh công khai do notebook cung cấp; không tải ảnh cá nhân, khuôn mặt, biển số, dữ liệu khách hàng hay dữ liệu nội bộ lên Colab hoặc GitHub công khai, và không ghi họ tên, MSSV hay thông tin cá nhân vào báo cáo hoặc output.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Mình sẽ dừng ngay, không xử lý hay đưa dữ liệu đó lên repo, và báo cho Lab Coach (hoặc giáo viên/Lab Lead nếu không liên lạc được Lab Coach). Nếu lỡ đã commit lên GitHub, mình cũng báo ngay để được hướng dẫn xóa.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
