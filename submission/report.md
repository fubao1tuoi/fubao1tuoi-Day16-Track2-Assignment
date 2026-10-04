# Báo cáo Lab 16 – Cloud AI Environment Setup trên GCP

1. Tôi triển khai bài lab trên Google Cloud Platform tại region `us-central1`, zone `us-central1-a`, sử dụng VM CPU `e2-medium`; source commit ban đầu là `55539f6`.
2. Dataset Credit Card Fraud Detection có 284.807 dòng, 31 cột và 492 giao dịch gian lận; dữ liệu được chia train/validation/test với seed 42.
3. Thời gian tải dữ liệu là 2,766525 giây, thời gian huấn luyện LightGBM là 7,947828 giây và best iteration là 130.
4. Trên tập test, mô hình đạt AUC-ROC 0,963446, Accuracy 0,998894, F1-score 0,734177, Precision 0,625899 và Recall 0,887755.
5. Median inference latency của một dòng qua 100 lần đo là 1,326130 ms; throughput batch 1.000 dòng đạt 163.321,89 dòng/giây.
6. Thông tin CPU, RAM và Network sau benchmark được ghi lại trong các ảnh `screenshot-resource-cpu.png`, `screenshot-resource-ram.png` và `screenshot-resource-network.png`.
7. Billing Report ghi nhận 1.367 ₫ cho Compute Engine và 943 ₫ cho Networking; credit/discount bù toàn bộ nên tổng thanh toán là 0 ₫.
8. Tôi đã tải mã nguồn và kết quả benchmark về máy, kiểm tra các artifact và destroy toàn bộ tài nguyên Terraform vào ngày 04/10/2026; `terraform state list` đã trống.