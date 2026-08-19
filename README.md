# DiffMM-AnchorOT — Phương án 6 (Learnable Coarse Anchor)

Đây là bản fork của [HKUDS/DiffMM](https://github.com/HKUDS/DiffMM) (paper *DiffMM: Multi-Modal
Diffusion Model for Recommendation*, ACM MM 2024) với **Phương án 6** trong kế hoạch tối ưu
[`DiffMM_OFM_Optimization_Plan.md`](../../DiffMM_OFM_Optimization_Plan.md) đã được áp dụng trực tiếp
vào code. Chi tiết thiết kế + suy diễn công thức đầy đủ ở
[`Phuong_An_6_Learnable_Anchor_KeHoachChiTiet.md`](../../Phuong_An_6_Learnable_Anchor_KeHoachChiTiet.md);
kết quả kiểm chứng patch thật ở
[`Phuong_An_6_GiaiDoanB_PatchThat/`](../../Phuong_An_6_GiaiDoanB_PatchThat/README.md).

## Thay đổi so với bản gốc

Ý tưởng lấy từ Optical Flow Matching (OFM, CVPR 2025): thay vì luôn khởi tạo/thêm nhiễu quanh **tâm cố
định** (như Phương án 1/3), dự đoán trước 1 **điểm neo thô** `α_l` — ước lượng sơ bộ, phụ thuộc từng
user, về việc user đó có khả năng thích item nào — rồi dịch tâm nhiễu về phía điểm neo đó:

```
αₜ = s(t)·α₀ + σₜ·anchor_w·α_l + σₜ·ε        (thay cho αₜ = s(t)·α₀ + σₜ·ε của Phương án 1/3)
α_l[u,:] = sigmoid(uEmbeds[u] @ iEmbedsᵀ)      (khong tham so hoc moi - dung embedding da co san)
```

`anchor_w=0` (mặc định trong `Params.py`) làm số hạng điểm neo triệt tiêu hoàn toàn ⟹ **công thức trùng
khít tuyệt đối Phương án 3** (đã kiểm chứng bằng hồi quy số học trên chính `Model.py` này, xem Giai đoạn
B) — an toàn để merge vào codebase, rồi tăng dần `anchor_w>0` để bật hiệu ứng điểm neo.

Trọng số loss CFM (Phương án 2/3) **hoàn toàn không đổi** — đã chứng minh đại số + xác nhận số học rằng
công thức trọng số bất biến với điểm neo (CT-6.7), nên `_precompute_cfm_weight()` được tái sử dụng
nguyên vẹn.

| File | Thay đổi |
|---|---|
| `Params.py` | Thêm `--sigma_min`, `--w_clip`, `--num_sample_steps` (kế thừa từ Phương án 3) và `--anchor_w` (mặc định `0.0`, mới) |
| `Model.py` | Thêm class `GaussianDiffusionOTCFM` (Phương án 3, sao chép nguyên vẹn — làm lớp cha) và `GaussianDiffusionAnchorOT(GaussianDiffusionOTCFM)` ở cuối file — không đụng `GaussianDiffusion` gốc. Override `q_sample`, `p_mean_variance`, `p_sample`, `training_losses`; thêm `_compute_anchor` (không tham số học) |
| `Main.py` | 1 dòng import, 1 dòng khởi tạo (đọc thêm `args.anchor_w`), 6 dòng gọi `training_losses`/`p_sample` (thêm đúng 1 tham số điểm neo — tái sử dụng biến `uEmbeds` **đã có sẵn** trong `trainEpoch`, không cần tính embedding mới) |
| `DataHandler.py` | `self.trnMat.A` → `self.trnMat.toarray()` (scipy ≥1.14 đã gỡ `.A` — lỗi môi trường đã biết, phòng trước) |

Toàn bộ phần còn lại (kiến trúc `Denoise` MLP, top-k rebuild đồ thị, MSI/`gc_loss`, Cross-Modal
Contrastive Augmentation, Multi-Modal Graph Aggregation, Multi-Task Training) **giữ nguyên 100%** so với
bản gốc — MSI đặc biệt không cần sửa gì vì chỉ dùng `α̂₀`/`α₀` cuối cùng, không phụ thuộc cơ chế nhiễu
nội bộ.

Đã kiểm chứng: patch biên dịch sạch (`py_compile`); kiểm tra số học trên CPU gồm — hồi quy `anchor_w=0`
trùng khít tuyệt đối Phương án 3 (kể cả toàn bộ vòng lặp `p_sample`), `cfm_weight` không đổi ở mọi
`anchor_w`, round-trip với denoiser hoàn hảo đúng ở mọi tổ hợp `(anchor_w, α_l)`, không NaN/Inf ở các
trường hợp biên (user rỗng, `anchor_w` rất lớn), và quy mô gần thực tế (batch=1024, ~7000 item, chi phí
tính điểm neo không đáng kể). Xem chi tiết ở
[`Phuong_An_6_GiaiDoanB_PatchThat/README.md`](../../Phuong_An_6_GiaiDoanB_PatchThat/README.md).

**Chưa kiểm chứng** (cần GPU + dữ liệu thật): chạy qua mạng `Denoise` thật, số liệu Recall/NDCG thật —
đặc biệt là liệu điểm neo thô `α_l` có thực sự cải thiện chất lượng gợi ý hay không (rủi ro thực nghiệm
đã nêu ở bản kế hoạch tổng, không thể kiểm chứng ngoài GPU thật).

## Dữ liệu

Thư mục `Datasets/` không chứa sẵn dữ liệu — dữ liệu được tải tự động từ Google Drive khi chạy notebook
Colab đi kèm (xem `DiffMM_PhuongAn6_AnchorOT_Colab.ipynb` ở thư mục cha). Nếu chạy cục bộ, xem
`Datasets/README.md` để biết định dạng dữ liệu cần có: `trnMat.pkl`, `tstMat.pkl`, `image_feat.npy`,
`text_feat.npy` (+ `audio_feat.npy` nếu dùng `tiktok`).

## Chạy cục bộ (không qua Colab)

```bash
python Main.py --data tiktok --reg 1e-4 --ssl_reg 1e-2 --epoch 50 --trans 1 --e_loss 0.1 --cl_method 1 --sigma_min 1e-3 --anchor_w 0.0
python Main.py --data baby   --reg 1e-5 --ssl_reg 1e-1 --keepRate 1 --e_loss 0.01 --sigma_min 1e-3 --anchor_w 0.0
python Main.py --data sports --reg 1e-6 --ssl_reg 1e-2 --temp 0.1 --ris_lambda 0.1 --e_loss 0.5 --keepRate 1 --trans 1 --sigma_min 1e-3 --anchor_w 0.0
```

Đổi `--anchor_w 0.0` thành giá trị `>0` (khuyến nghị thử `1.0`-`5.0`) để bật hiệu ứng điểm neo — chưa có
cơ sở lý thuyết để suy ra giá trị tối ưu, cần quét thực nghiệm.

(Yêu cầu GPU — code gốc dùng `.cuda()` trực tiếp, không tự động rơi về CPU.)

## Bản gốc

Xem `Params.py`/`Model.py`/`Main.py` để đối chiếu — mọi phần không liên quan Phương án 6 giữ nguyên y
hệt [HKUDS/DiffMM](https://github.com/HKUDS/DiffMM). Trích dẫn paper gốc:

```
@article{jiang2024diffmm,
  title={DiffMM: Multi-Modal Diffusion Model for Recommendation},
  author={Jiang, Yangqin and Xia, Lianghao and Wei, Wei and Luo, Da and Lin, Kangyi and Huang, Chao},
  journal={arXiv preprint arXiv:2406.11781},
  year={2024}
}
```

Ý tưởng điểm neo lấy cảm hứng từ:
```
@inproceedings{luo2025ofm,
  title={Optical Flow Matching: Reframing Optical Flow as Continuous Transport Dynamics},
  author={Luo, Ao and Li, Xin and Yang, Fan and Li, Yuezun and Yuan, Zhaoquan and Zhao, Shan and Su, Bing and Wu, Xiao},
  booktitle={CVPR},
  year={2025}
}
```
