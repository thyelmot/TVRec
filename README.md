# DiffMM-TVS — Phương án 7 (Triangle Velocities Synergy - TVS)

Đây là bản fork của [HKUDS/DiffMM](https://github.com/HKUDS/DiffMM) (paper *DiffMM: Multi-Modal Diffusion Model for Recommendation*, ACM MM 2024) với **Phương án 7** trong kế hoạch tối ưu [`DiffMM_OFM_Optimization_Plan.md`](../../DiffMM_OFM_Optimization_Plan.md) đã được áp dụng trực tiếp vào code. Chi tiết thiết kế + suy diễn công thức đầy đủ ở [`Phuong_An_7_TVS_KeHoachChiTiet.md`](../../Phuong_An_7_TVS_KeHoachChiTiet.md); kết quả kiểm chứng patch thật ở [`verify_cpu_tvs.py`](verify_cpu_tvs.py).

## Thay đổi so với bản gốc

Ý tưởng lấy từ Optical Flow Matching (OFM, CVPR 2025): thay vì dạy mạng hồi quy dữ liệu gốc trực tiếp (data-prediction), ta chuyển sang quy trình hồi quy vận tốc (velocity-prediction) trên 3 quỹ đạo (quỹ đạo chính $x_t$, quỹ đạo phụ 1 $y_t$ hướng tới điểm neo thô, và quỹ đạo phụ 2 $z_t$ đứng yên tại chỗ). 

### 1. Ba quỹ đạo (CT-7.1)
- Quỹ đạo chính (từ $\alpha_0 \rightarrow \alpha_{ref}$): $x_t = \mu_t \alpha_0 + \sigma_t w \alpha_l + \sigma_t \epsilon$
- Quỹ đạo phụ 1 (từ $\alpha_0 \rightarrow \alpha_l$): $y_t = t_{norm} \alpha_l + (1-t_{norm}) \alpha_0 + \sigma_{min} \epsilon_y$
- Quỹ đạo phụ 2 (từ $\alpha_0 \rightarrow \alpha_0$): $z_t = \alpha_0 + \sigma_{min} \epsilon_z$

### 2. Ba vận tốc tương ứng (CT-7.2 & CT-7.3)
- Vận tốc target quỹ đạo chính: $v_{gt\_x} = (1 - \sigma_{min})(w \alpha_l + \epsilon) - x_{start}$
- Vận tốc target quỹ đạo phụ 1: $v_{gt\_y} = \alpha_l - x_{start}$
- Vận tốc target quỹ đạo phụ 2: $v_{gt\_z} = 0$

### 3. Công thức tái tạo dữ liệu không NaN/Inf (CT-7.4)
Thay vì dùng công thức chia cho $1-t$ dễ gây bất ổn định số ở biên, chúng ta đã chứng minh đại số và kiểm chứng số học công thức tái tạo đóng cực kỳ ổn định:
$$\hat{\alpha}_0 = (1 - \sigma_{min}) x_t - \sigma_t v_{pred}$$

Khi tắt chế độ TVS (`velocity_mode=False`), mô hình tự động fallback về Phương án 6 (hồi quy dữ liệu gốc trực tiếp).

| File | Thay đổi |
|---|---|
| `Params.py` | Thêm các tham số mới: `--velocity_mode` (1: bật TVS, 0: tắt), `--lambda_x`, `--lambda_y`, `--lambda_z` để kiểm soát trọng số loss 3 quỹ đạo. |
| `Model.py` | Thêm class `GaussianDiffusionTVS(GaussianDiffusionAnchorOT)` ở cuối file. Override `p_mean_variance` để tự động chuyển đổi velocity $\rightarrow$ data khi `velocity_mode=True`. Override `training_losses` để tính loss TVS trên 3 quỹ đạo. |
| `Main.py` | Cập nhật dòng import `GaussianDiffusionTVS` và khởi tạo mô hình truyền đầy đủ các tham số TVS. |

Toàn bộ phần còn lại (kiến trúc mạng `Denoise` MLP, top-k rebuild đồ thị, MSI/`gc_loss`, Cross-Modal Contrastive Augmentation, Multi-Modal Graph Aggregation, Multi-Task Training) **giữ nguyên 100%** so với bản gốc.

## Đã kiểm chứng trước khi bàn giao

- [x] Patch biên dịch sạch (`py_compile`).
- [x] Kiểm tra số học trên CPU qua `verify_cpu_tvs.py`:
  - Lớp `GaussianDiffusionTVS` khi tắt TVS (`velocity_mode=False`) cho kết quả q_sample, p_mean_variance và p_sample **khớp tuyệt đối** với lớp cha `GaussianDiffusionAnchorOT` (an toàn tuyệt đối để tích hợp).
  - Công thức tái tạo $\hat{\alpha}_0$ (CT-7.4) ổn định số học hoàn toàn và cho phép **tái tạo 100% dữ liệu gốc** với denoiser hoàn hảo (sai số $< 10^{-7}$).
  - Hàm `training_losses` chạy ổn định, không NaN/Inf trên dải tham số rộng.
- [x] Notebook đã dry-run cell-theo-cell (nhánh thành công + nhánh lỗi).

## Dữ liệu

Dữ liệu được tải tự động từ Google Drive khi chạy notebook Colab đi kèm (xem `DiffMM_PhuongAn7_TVS_Colab.ipynb` ở thư mục cha). Định dạng dữ liệu: `trnMat.pkl`, `tstMat.pkl`, `image_feat.npy`, `text_feat.npy` (+ `audio_feat.npy` nếu dùng `tiktok`).

## Chạy cục bộ (không qua Colab)

Để chạy thử nghiệm TVS:
```bash
python Main.py --data tiktok --epoch 50 --velocity_mode 1 --anchor_w 2.0 --lambda_x 1.0 --lambda_y 1.0 --lambda_z 1.0
```

Để fallback về AnchorOT (PA6):
```bash
python Main.py --data tiktok --epoch 50 --velocity_mode 0 --anchor_w 2.0
```
