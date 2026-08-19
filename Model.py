import torch
from torch import nn
import torch.nn.functional as F
from Params import args
import numpy as np
import random
import math
from Utils.Utils import *

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class Model(nn.Module):
	def __init__(self, image_embedding, text_embedding, audio_embedding=None):
		super(Model, self).__init__()

		self.uEmbeds = nn.Parameter(init(torch.empty(args.user, args.latdim)))
		self.iEmbeds = nn.Parameter(init(torch.empty(args.item, args.latdim)))
		self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])

		self.edgeDropper = SpAdjDropEdge(args.keepRate)

		if args.trans == 1:
			self.image_trans = nn.Linear(args.image_feat_dim, args.latdim)
			self.text_trans = nn.Linear(args.text_feat_dim, args.latdim)
		elif args.trans == 0:
			self.image_trans = nn.Parameter(init(torch.empty(size=(args.image_feat_dim, args.latdim))))
			self.text_trans = nn.Parameter(init(torch.empty(size=(args.text_feat_dim, args.latdim))))
		else:
			self.image_trans = nn.Parameter(init(torch.empty(size=(args.image_feat_dim, args.latdim))))
			self.text_trans = nn.Linear(args.text_feat_dim, args.latdim)
		if audio_embedding != None:
			if args.trans == 1:
				self.audio_trans = nn.Linear(args.audio_feat_dim, args.latdim)
			else:
				self.audio_trans = nn.Parameter(init(torch.empty(size=(args.audio_feat_dim, args.latdim))))

		self.image_embedding = image_embedding
		self.text_embedding = text_embedding
		if audio_embedding != None:
			self.audio_embedding = audio_embedding
		else:
			self.audio_embedding = None

		if audio_embedding != None:
			self.modal_weight = nn.Parameter(torch.Tensor([0.3333, 0.3333, 0.3333]))
		else:
			self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
		self.softmax = nn.Softmax(dim=0)

		self.dropout = nn.Dropout(p=0.1)

		self.leakyrelu = nn.LeakyReLU(0.2)
				
	def getItemEmbeds(self):
		return self.iEmbeds
	
	def getUserEmbeds(self):
		return self.uEmbeds
	
	def getImageFeats(self):
		if args.trans == 0 or args.trans == 2:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			return image_feats
		else:
			return self.image_trans(self.image_embedding)
	
	def getTextFeats(self):
		if args.trans == 0:
			text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
			return text_feats
		else:
			return self.text_trans(self.text_embedding)

	def getAudioFeats(self):
		if self.audio_embedding == None:
			return None
		else:
			if args.trans == 0:
				audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_trans))
			else:
				audio_feats = self.audio_trans(self.audio_embedding)
		return audio_feats

	def forward_MM(self, adj, image_adj, text_adj, audio_adj=None):
		if args.trans == 0:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
		elif args.trans == 1:
			image_feats = self.image_trans(self.image_embedding)
			text_feats = self.text_trans(self.text_embedding)
		else:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			text_feats = self.text_trans(self.text_embedding)

		if audio_adj != None:
			if args.trans == 0:
				audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_trans))
			else:
				audio_feats = self.audio_trans(self.audio_embedding)

		weight = self.softmax(self.modal_weight)

		embedsImageAdj = torch.concat([self.uEmbeds, self.iEmbeds])
		embedsImageAdj = torch.spmm(image_adj, embedsImageAdj)

		embedsImage = torch.concat([self.uEmbeds, F.normalize(image_feats)])
		embedsImage = torch.spmm(adj, embedsImage)

		embedsImage_ = torch.concat([embedsImage[:args.user], self.iEmbeds])
		embedsImage_ = torch.spmm(adj, embedsImage_)
		embedsImage += embedsImage_
		
		embedsTextAdj = torch.concat([self.uEmbeds, self.iEmbeds])
		embedsTextAdj = torch.spmm(text_adj, embedsTextAdj)

		embedsText = torch.concat([self.uEmbeds, F.normalize(text_feats)])
		embedsText = torch.spmm(adj, embedsText)

		embedsText_ = torch.concat([embedsText[:args.user], self.iEmbeds])
		embedsText_ = torch.spmm(adj, embedsText_)
		embedsText += embedsText_

		if audio_adj != None:
			embedsAudioAdj = torch.concat([self.uEmbeds, self.iEmbeds])
			embedsAudioAdj = torch.spmm(audio_adj, embedsAudioAdj)

			embedsAudio = torch.concat([self.uEmbeds, F.normalize(audio_feats)])
			embedsAudio = torch.spmm(adj, embedsAudio)

			embedsAudio_ = torch.concat([embedsAudio[:args.user], self.iEmbeds])
			embedsAudio_ = torch.spmm(adj, embedsAudio_)
			embedsAudio += embedsAudio_

		embedsImage += args.ris_adj_lambda * embedsImageAdj
		embedsText += args.ris_adj_lambda * embedsTextAdj
		if audio_adj != None:
			embedsAudio += args.ris_adj_lambda * embedsAudioAdj
		if audio_adj == None:
			embedsModal = weight[0] * embedsImage + weight[1] * embedsText
		else:
			embedsModal = weight[0] * embedsImage + weight[1] * embedsText + weight[2] * embedsAudio

		embeds = embedsModal
		embedsLst = [embeds]
		for gcn in self.gcnLayers:
			embeds = gcn(adj, embedsLst[-1])
			embedsLst.append(embeds)
		embeds = sum(embedsLst)

		embeds = embeds + args.ris_lambda * F.normalize(embedsModal)

		return embeds[:args.user], embeds[args.user:]

	def forward_cl_MM(self, adj, image_adj, text_adj, audio_adj=None):
		if args.trans == 0:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
		elif args.trans == 1:
			image_feats = self.image_trans(self.image_embedding)
			text_feats = self.text_trans(self.text_embedding)
		else:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			text_feats = self.text_trans(self.text_embedding)

		if audio_adj != None:
			if args.trans == 0:
				audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_trans))
			else:
				audio_feats = self.audio_trans(self.audio_embedding)

		embedsImage = torch.concat([self.uEmbeds, F.normalize(image_feats)])
		embedsImage = torch.spmm(image_adj, embedsImage)

		embedsText = torch.concat([self.uEmbeds, F.normalize(text_feats)])
		embedsText = torch.spmm(text_adj, embedsText)

		if audio_adj != None:
			embedsAudio = torch.concat([self.uEmbeds, F.normalize(audio_feats)])
			embedsAudio = torch.spmm(audio_adj, embedsAudio)

		embeds1 = embedsImage
		embedsLst1 = [embeds1]
		for gcn in self.gcnLayers:
			embeds1 = gcn(adj, embedsLst1[-1])
			embedsLst1.append(embeds1)
		embeds1 = sum(embedsLst1)

		embeds2 = embedsText
		embedsLst2 = [embeds2]
		for gcn in self.gcnLayers:
			embeds2 = gcn(adj, embedsLst2[-1])
			embedsLst2.append(embeds2)
		embeds2 = sum(embedsLst2)

		if audio_adj != None:
			embeds3 = embedsAudio
			embedsLst3 = [embeds3]
			for gcn in self.gcnLayers:
				embeds3 = gcn(adj, embedsLst3[-1])
				embedsLst3.append(embeds3)
			embeds3 = sum(embedsLst3)

		if audio_adj == None:
			return embeds1[:args.user], embeds1[args.user:], embeds2[:args.user], embeds2[args.user:]
		else:
			return embeds1[:args.user], embeds1[args.user:], embeds2[:args.user], embeds2[args.user:], embeds3[:args.user], embeds3[args.user:]

	def reg_loss(self):
		ret = 0
		ret += self.uEmbeds.norm(2).square()
		ret += self.iEmbeds.norm(2).square()
		return ret

class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()

	def forward(self, adj, embeds):
		return torch.spmm(adj, embeds)

class SpAdjDropEdge(nn.Module):
	def __init__(self, keepRate):
		super(SpAdjDropEdge, self).__init__()
		self.keepRate = keepRate

	def forward(self, adj):
		vals = adj._values()
		idxs = adj._indices()
		edgeNum = vals.size()
		mask = ((torch.rand(edgeNum) + self.keepRate).floor()).type(torch.bool)

		newVals = vals[mask] / self.keepRate
		newIdxs = idxs[:, mask]

		return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)
		
class Denoise(nn.Module):
	def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5):
		super(Denoise, self).__init__()
		self.in_dims = in_dims
		self.out_dims = out_dims
		self.time_emb_dim = emb_size
		self.norm = norm

		self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

		in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]

		out_dims_temp = self.out_dims

		self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
		self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

		self.drop = nn.Dropout(dropout)
		self.init_weights()

	def init_weights(self):
		for layer in self.in_layers:
			size = layer.weight.size()
			std = np.sqrt(2.0 / (size[0] + size[1]))
			layer.weight.data.normal_(0.0, std)
			layer.bias.data.normal_(0.0, 0.001)
		
		for layer in self.out_layers:
			size = layer.weight.size()
			std = np.sqrt(2.0 / (size[0] + size[1]))
			layer.weight.data.normal_(0.0, std)
			layer.bias.data.normal_(0.0, 0.001)

		size = self.emb_layer.weight.size()
		std = np.sqrt(2.0 / (size[0] + size[1]))
		self.emb_layer.weight.data.normal_(0.0, std)
		self.emb_layer.bias.data.normal_(0.0, 0.001)

	def forward(self, x, timesteps, mess_dropout=True):
		freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).cuda()
		temp = timesteps[:, None].float() * freqs[None]
		time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
		if self.time_emb_dim % 2:
			time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
		emb = self.emb_layer(time_emb)
		if self.norm:
			x = F.normalize(x)
		if mess_dropout:
			x = self.drop(x)
		h = torch.cat([x, emb], dim=-1)
		for i, layer in enumerate(self.in_layers):
			h = layer(h)
			h = torch.tanh(h)
		for i, layer in enumerate(self.out_layers):
			h = layer(h)
			if i != len(self.out_layers) - 1:
				h = torch.tanh(h)

		return h

class GaussianDiffusion(nn.Module):
	def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True):
		super(GaussianDiffusion, self).__init__()

		self.noise_scale = noise_scale
		self.noise_min = noise_min
		self.noise_max = noise_max
		self.steps = steps

		if noise_scale != 0:
			self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).cuda()
			if beta_fixed:
				self.betas[0] = 0.0001

			self.calculate_for_diffusion()

	def get_betas(self):
		start = self.noise_scale * self.noise_min
		end = self.noise_scale * self.noise_max
		variance = np.linspace(start, end, self.steps, dtype=np.float64)
		alpha_bar = 1 - variance
		betas = []
		betas.append(1 - alpha_bar[0])
		for i in range(1, self.steps):
			betas.append(min(1 - alpha_bar[i] / alpha_bar[i-1], 0.999))
		return np.array(betas) 

	def calculate_for_diffusion(self):
		alphas = 1.0 - self.betas
		self.alphas_cumprod = torch.cumprod(alphas, axis=0).cuda()
		self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).cuda(), self.alphas_cumprod[:-1]]).cuda()
		self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).cuda()]).cuda()

		self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
		self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
		self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
		self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
		self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

		self.posterior_variance = (
			self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
		)
		self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
		self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
		self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))

	def p_sample(self, model, x_start, steps, sampling_noise=False):
		if steps == 0:
			x_t = x_start
		else:
			t = torch.tensor([steps-1] * x_start.shape[0]).cuda()
			x_t = self.q_sample(x_start, t)
		
		indices = list(range(self.steps))[::-1]

		for i in indices:
			t = torch.tensor([i] * x_t.shape[0]).cuda()
			model_mean, model_log_variance = self.p_mean_variance(model, x_t, t)
			if sampling_noise:
				noise = torch.randn_like(x_t)
				nonzero_mask = ((t!=0).float().view(-1, *([1]*(len(x_t.shape)-1))))
				x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
			else:
				x_t = model_mean
		return x_t

	def q_sample(self, x_start, t, noise=None):
		if noise is None:
			noise = torch.randn_like(x_start)
		return self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

	def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
		arr = arr.cuda()
		res = arr[timesteps].float()
		while len(res.shape) < len(broadcast_shape):
			res = res[..., None]
		return res.expand(broadcast_shape)

	def p_mean_variance(self, model, x, t):
		model_output = model(x, t, False)

		model_variance = self.posterior_variance
		model_log_variance = self.posterior_log_variance_clipped

		model_variance = self._extract_into_tensor(model_variance, t, x.shape)
		model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

		model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * model_output + self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x)
		
		return model_mean, model_log_variance

	def training_losses(self, model, x_start, itmEmbeds, batch_index, model_feats):
		batch_size = x_start.size(0)

		ts = torch.randint(0, self.steps, (batch_size,)).long().cuda()
		noise = torch.randn_like(x_start)
		if self.noise_scale != 0:
			x_t = self.q_sample(x_start, ts, noise)
		else:
			x_t = x_start

		model_output = model(x_t, ts)

		mse = self.mean_flat((x_start - model_output) ** 2)

		weight = self.SNR(ts - 1) - self.SNR(ts)
		weight = torch.where((ts == 0), 1.0, weight)

		diff_loss = weight * mse

		usr_model_embeds = torch.mm(model_output, model_feats)
		usr_id_embeds = torch.mm(x_start, itmEmbeds)

		gc_loss = self.mean_flat((usr_model_embeds - usr_id_embeds) ** 2)

		return diff_loss, gc_loss
		
	def mean_flat(self, tensor):
		return tensor.mean(dim=list(range(1, len(tensor.shape))))
	
	def SNR(self, t):
		self.alphas_cumprod = self.alphas_cumprod.cuda()
		return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])

class GaussianDiffusionOTCFM(GaussianDiffusion):
	"""
	[Phuong an 3 - Phuong_An_3_OT_CFM_KetHop_KeHoachChiTiet.md]
	Ket hop Phuong an 1 (duong di OT-linear, D1) + Phuong an 2 (trong so CFM loss, D2) + rut gon
	so buoc suy luan D4 (generalized-DDIM voi lich trinh cach deu K < T buoc thay vi T buoc tuan
	tu). D1 va cong thuc trong so D2 duoc TAI SU DUNG NGUYEN VEN tu Phuong an 1/2 (da chung minh
	dai so + kiem chung so hoc rang cong thuc w_CFM(t) khong phu thuoc duong di cu the). Phan MOI
	duy nhat la D4 (p_mean_variance/p_sample): da kiem chung bang giai tich rang cong thuc DDIM
	tong quat hoa (dung lai tu Phuong an 1, khong doi cong thuc) dung cho MOI cap chi so
	(t, t_prev) tren cung 1 quy dao, khong bat buoc t_prev = t-1 - nen chi can doi lich trinh lap
	trong p_sample, khong can suy them toan hoc nao.

	Override 4 ham: q_sample (D1, y het Phuong an 1), training_losses (D2, y het Phuong an 2 nhung
	ap len mu/sigma cua duong OT), p_mean_variance + p_sample (D4, MOI - lich trinh rut gon).
	"""

	def __init__(self, sigma_min, steps, w_clip=50.0, num_sample_steps=0):
		nn.Module.__init__(self)  # bo qua __init__ cua GaussianDiffusion (khong can beta kieu VP)
		self.steps = steps
		self.sigma_min = sigma_min
		self.w_clip = w_clip
		self.noise_scale = 1.0  # de tuong thich dieu kien "if self.noise_scale != 0" trong training_losses

		# --- D1: duong di OT-linear (tai su dung nguyen ven Phuong an 1) ---
		t_idx = torch.arange(steps, dtype=torch.float64)
		s = (1.0 - t_idx / (steps - 1)) if steps > 1 else torch.ones_like(t_idx)
		self.mu_coef = s.cuda()
		self.sigma_coef = (1.0 - (1.0 - sigma_min) * s).cuda()

		# --- D2: trong so CFM (tai su dung nguyen ven cong thuc Phuong an 2, ap len mu/sigma cua OT) ---
		self._precompute_cfm_weight()

		# --- D4: lich trinh suy luan rut gon (MOI) ---
		K = num_sample_steps if num_sample_steps and num_sample_steps > 0 else max(1, round(0.6 * steps))
		self.num_sample_steps = K
		self._build_sample_schedule(K)

	def _precompute_cfm_weight(self):
		# Y HET cong thuc cua GaussianDiffusionCFM (Phuong an 2) - khong sua gi, chi mu/sigma dau vao khac
		mu, sigma = self.mu_coef, self.sigma_coef
		mu_prev = torch.cat([mu[:1], mu[:-1]])
		sigma_prev = torch.cat([sigma[:1], sigma[:-1]])
		mu_prime = mu_prev - mu
		sigma_prime = sigma_prev - sigma
		w = (mu_prime - (sigma_prime / sigma.clamp(min=1e-8)) * mu) ** 2
		w[0] = 1.0  # bien t=0: dung dung quy uoc cua Phuong an 1/2
		self.cfm_weight = w.clamp(max=self.w_clip).cuda()

	def _build_sample_schedule(self, K):
		# K chi so cach deu trong {0,...,T-1}, giam dan, luon co ca 2 dau mut
		idx = torch.linspace(self.steps - 1, 0, steps=K).round().long()
		idx = torch.unique_consecutive(idx)  # tranh trung neu K > T hoac lam tron trung nhau
		self.sample_schedule = idx.tolist()
		# next_index_map[t] = t_prev (hoac -1 = "sach hoan toan") - chi dinh nghia cho t trong schedule
		self.next_index_map = {}
		for k, t in enumerate(self.sample_schedule):
			self.next_index_map[t] = self.sample_schedule[k + 1] if k + 1 < len(self.sample_schedule) else -1

	def q_sample(self, x_start, t, noise=None):
		if noise is None:
			noise = torch.randn_like(x_start)
		mu_t = self._extract_into_tensor(self.mu_coef, t, x_start.shape)
		sigma_t = self._extract_into_tensor(self.sigma_coef, t, x_start.shape)
		return mu_t * x_start + sigma_t * noise

	def training_losses(self, model, x_start, itmEmbeds, batch_index, model_feats):
		batch_size = x_start.size(0)

		ts = torch.randint(0, self.steps, (batch_size,)).long().cuda()
		noise = torch.randn_like(x_start)
		if self.noise_scale != 0:
			x_t = self.q_sample(x_start, ts, noise)
		else:
			x_t = x_start

		model_output = model(x_t, ts)

		mse = self.mean_flat((x_start - model_output) ** 2)

		weight = self._extract_into_tensor(self.cfm_weight, ts, mse.shape)
		diff_loss = weight * mse

		usr_model_embeds = torch.mm(model_output, model_feats)
		usr_id_embeds = torch.mm(x_start, itmEmbeds)

		gc_loss = self.mean_flat((usr_model_embeds - usr_id_embeds) ** 2)

		return diff_loss, gc_loss

	def p_mean_variance(self, model, x, t):
		model_output = model(x, t, False)  # du doan alpha_0 (giong het parameterization cua code goc)

		mu_t = self._extract_into_tensor(self.mu_coef, t, x.shape)
		sigma_t = self._extract_into_tensor(self.sigma_coef, t, x.shape)
		noise_pred = (x - mu_t * model_output) / sigma_t.clamp(min=1e-8)

		# ca batch dung chung 1 gia tri t trong vong lap p_sample (xem cach p_sample duoc goi ben duoi)
		t_val = int(t[0].item())
		t_prev_val = self.next_index_map[t_val]
		if t_prev_val == -1:
			mu_prev, sigma_prev = 1.0, 0.0
		else:
			t_prev = torch.full_like(t, t_prev_val)
			mu_prev = self._extract_into_tensor(self.mu_coef, t_prev, x.shape)
			sigma_prev = self._extract_into_tensor(self.sigma_coef, t_prev, x.shape)

		model_mean = mu_prev * model_output + sigma_prev * noise_pred
		model_log_variance = None  # bien the OT-CFM chi ho tro suy luan tat dinh (sampling_noise=False)
		return model_mean, model_log_variance

	def p_sample(self, model, x_start, steps, sampling_noise=False):
		if steps == 0:
			x_t = x_start
		else:
			t = torch.tensor([steps - 1] * x_start.shape[0]).cuda()
			x_t = self.q_sample(x_start, t)

		for i in self.sample_schedule:  # <-- CHI KHAC lop cha: lap qua lich trinh rut gon thay vi range(T)
			t = torch.tensor([i] * x_t.shape[0]).cuda()
			model_mean, _ = self.p_mean_variance(model, x_t, t)
			x_t = model_mean  # tat dinh (sampling_noise khong ho tro, giong Phuong an 1)
		return x_t

class GaussianDiffusionAnchorOT(GaussianDiffusionOTCFM):
	"""
	[Phuong an 6 - Phuong_An_6_Learnable_Anchor_KeHoachChiTiet.md]
	Ke thua GaussianDiffusionOTCFM (Phuong an 3) de tai su dung nguyen ven duong OT (D1) VA trong so
	CFM (D2) - da CHUNG MINH DAI SO + KIEM CHUNG SO HOC rang trong so CFM BAT BIEN voi diem neo (CT-6.7
	trong ban ke hoach chi tiet), nen KHONG can sua _precompute_cfm_weight/cfm_weight chut nao. Chi
	override q_sample (D1) va p_mean_variance/p_sample (D4) de them dung 1 so hang moi: diem neo
	alpha_l * anchor_w, nhan voi he so sigma_coef da co san (CT-6.2, CT-6.4).

	alpha_l (CT-6.1) la 1 ham DONG (khong tham so hoc moi) cua embedding user/item DA CO SAN va DA
	DUOC .detach() tu truoc (giong het cach model_feats/itmEmbeds duoc truyen vao o Phuong an 1-5) -
	khong tao them vong lap phan hoi gradient nao.

	anchor_w=0.0 (mac dinh) lam trung khit tuyet doi Phuong an 3 (CT-6.5, da kiem chung so hoc atol=0).
	"""

	def __init__(self, sigma_min, steps, w_clip=50.0, num_sample_steps=0, anchor_w=0.0):
		super(GaussianDiffusionAnchorOT, self).__init__(sigma_min, steps, w_clip=w_clip, num_sample_steps=num_sample_steps)
		self.anchor_w = anchor_w

	def _compute_anchor(self, uEmbeds_batch, iEmbeds):
		# CT-6.1: khong them tham so hoc moi - chi la ham dong cua embedding da co san
		return torch.sigmoid(torch.mm(uEmbeds_batch, iEmbeds.t()))

	def q_sample(self, x_start, alpha_l, t, noise=None):
		# CT-6.2
		if noise is None:
			noise = torch.randn_like(x_start)
		mu_t = self._extract_into_tensor(self.mu_coef, t, x_start.shape)
		sigma_t = self._extract_into_tensor(self.sigma_coef, t, x_start.shape)
		return mu_t * x_start + sigma_t * self.anchor_w * alpha_l + sigma_t * noise

	def p_mean_variance(self, model, x, alpha_l, t):
		# CT-6.4
		model_output = model(x, t, False)  # du doan alpha_0 (khong doi parameterization)

		mu_t = self._extract_into_tensor(self.mu_coef, t, x.shape)
		sigma_t = self._extract_into_tensor(self.sigma_coef, t, x.shape)
		noise_pred = (x - mu_t * model_output - sigma_t * self.anchor_w * alpha_l) / sigma_t.clamp(min=1e-8)

		t_val = int(t[0].item())
		t_prev_val = self.next_index_map[t_val]
		if t_prev_val == -1:
			mu_prev, sigma_prev = 1.0, 0.0
		else:
			t_prev = torch.full_like(t, t_prev_val)
			mu_prev = self._extract_into_tensor(self.mu_coef, t_prev, x.shape)
			sigma_prev = self._extract_into_tensor(self.sigma_coef, t_prev, x.shape)

		# bien t=0 (mu_prev=1, sigma_prev=0): so hang neo tu triet tieu, dung quy uoc PA1/PA3 (CT-6.4)
		model_mean = mu_prev * model_output + sigma_prev * self.anchor_w * alpha_l + sigma_prev * noise_pred
		model_log_variance = None
		return model_mean, model_log_variance

	def p_sample(self, model, x_start, uEmbeds_batch, iEmbeds, steps, sampling_noise=False):
		if self.anchor_w != 0:
			alpha_l = self._compute_anchor(uEmbeds_batch, iEmbeds)
		else:
			alpha_l = torch.zeros_like(x_start)  # anchor_w=0 -> so hang neo = 0 du alpha_l la gi (CT-6.5)

		if steps == 0:
			x_t = x_start
		else:
			t = torch.tensor([steps - 1] * x_start.shape[0]).cuda()
			x_t = self.q_sample(x_start, alpha_l, t)

		for i in self.sample_schedule:
			t = torch.tensor([i] * x_t.shape[0]).cuda()
			model_mean, _ = self.p_mean_variance(model, x_t, alpha_l, t)
			x_t = model_mean
		return x_t

	def training_losses(self, model, x_start, itmEmbeds, batch_index, model_feats, uEmbeds_batch):
		batch_size = x_start.size(0)

		if self.anchor_w != 0:
			alpha_l = self._compute_anchor(uEmbeds_batch, itmEmbeds)
		else:
			alpha_l = torch.zeros_like(x_start)

		ts = torch.randint(0, self.steps, (batch_size,)).long().cuda()
		noise = torch.randn_like(x_start)
		if self.noise_scale != 0:
			x_t = self.q_sample(x_start, alpha_l, ts, noise)
		else:
			x_t = x_start

		model_output = model(x_t, ts)

		mse = self.mean_flat((x_start - model_output) ** 2)

		weight = self._extract_into_tensor(self.cfm_weight, ts, mse.shape)  # KHONG doi - bat bien voi diem neo (CT-6.7)
		diff_loss = weight * mse

		usr_model_embeds = torch.mm(model_output, model_feats)
		usr_id_embeds = torch.mm(x_start, itmEmbeds)

		gc_loss = self.mean_flat((usr_model_embeds - usr_id_embeds) ** 2)

		return diff_loss, gc_loss

class GaussianDiffusionTVS(GaussianDiffusionAnchorOT):
	"""
	[Phuong an 7 - Phuong_An_7_TVS_KeHoachChiTiet.md]
	Triangle Velocities Synergy cho DiffMM.
	Ke thua GaussianDiffusionAnchorOT (Phuong an 6) de tai su dung nguyen ven duong OT (D1)
	va trong so CFM (D2). Chi override training_losses va p_mean_variance.
	"""
	def __init__(self, sigma_min, steps, w_clip=50.0, num_sample_steps=0, anchor_w=0.0,
				 velocity_mode=False, lambda_x=1.0, lambda_y=1.0, lambda_z=1.0):
		super(GaussianDiffusionTVS, self).__init__(sigma_min, steps, w_clip=w_clip,
												   num_sample_steps=num_sample_steps, anchor_w=anchor_w)
		self.velocity_mode = bool(velocity_mode)
		self.lambda_x = lambda_x
		self.lambda_y = lambda_y
		self.lambda_z = lambda_z

	def p_mean_variance(self, model, x, alpha_l, t):
		model_output = model(x, t, False) # v_pred neu velocity_mode=True, else alpha_0_pred

		mu_t = self._extract_into_tensor(self.mu_coef, t, x.shape)
		sigma_t = self._extract_into_tensor(self.sigma_coef, t, x.shape)

		if self.velocity_mode:
			# CT-7.4 khao sat lai: alpha0_reconstructed = (1 - sigma_min) * x - sigma_t * model_output
			# Day la cong thuc hoi quy chinh xac khong NaN/Inf o moi t, da kiem chung so hoc verify_tvs.py
			alpha0_reconstructed = (1.0 - self.sigma_min) * x - sigma_t * model_output
		else:
			alpha0_reconstructed = model_output

		noise_pred = (x - mu_t * alpha0_reconstructed - sigma_t * self.anchor_w * alpha_l) / sigma_t.clamp(min=1e-8)

		t_val = int(t[0].item())
		t_prev_val = self.next_index_map[t_val]
		if t_prev_val == -1:
			mu_prev, sigma_prev = 1.0, 0.0
		else:
			t_prev = torch.full_like(t, t_prev_val)
			mu_prev = self._extract_into_tensor(self.mu_coef, t_prev, x.shape)
			sigma_prev = self._extract_into_tensor(self.sigma_coef, t_prev, x.shape)

		model_mean = mu_prev * alpha0_reconstructed + sigma_prev * self.anchor_w * alpha_l + sigma_prev * noise_pred
		model_log_variance = None
		return model_mean, model_log_variance

	def training_losses(self, model, x_start, itmEmbeds, batch_index, model_feats, uEmbeds_batch):
		if not self.velocity_mode:
			return super(GaussianDiffusionTVS, self).training_losses(model, x_start, itmEmbeds, batch_index, model_feats, uEmbeds_batch)

		batch_size = x_start.size(0)

		if self.anchor_w != 0:
			alpha_l = self._compute_anchor(uEmbeds_batch, itmEmbeds)
		else:
			alpha_l = torch.zeros_like(x_start)

		ts = torch.randint(0, self.steps, (batch_size,)).long().cuda()
		noise = torch.randn_like(x_start)
		x_t = self.q_sample(x_start, alpha_l, ts, noise)

		v_pred_x = model(x_t, ts)
		v_gt_x = (1.0 - self.sigma_min) * (self.anchor_w * alpha_l + noise) - x_start

		# Build auxiliary trajectories (CT-7.1)
		t_norm = ts.float() / (self.steps - 1) if self.steps > 1 else torch.zeros_like(ts).float()
		t_norm = t_norm.view(-1, 1)

		eps_y = torch.randn_like(x_start)
		y_t = t_norm * alpha_l + (1.0 - t_norm) * x_start + self.sigma_min * eps_y

		eps_z = torch.randn_like(x_start)
		z_t = x_start + self.sigma_min * eps_z

		v_pred_y = model(y_t, ts)
		v_pred_z = model(z_t, ts)

		v_gt_y = alpha_l - x_start
		v_gt_z = torch.zeros_like(x_start)

		mse_x = self.mean_flat((v_pred_x - v_gt_x) ** 2)
		mse_y = self.mean_flat((v_pred_y - v_gt_y) ** 2)
		mse_z = self.mean_flat((v_pred_z - v_gt_z) ** 2)

		mse_flat = self.lambda_x * mse_x + self.lambda_y * mse_y + self.lambda_z * mse_z

		weight = self._extract_into_tensor(self.cfm_weight, ts, mse_flat.shape)
		diff_loss = weight * mse_flat

		# Reconstruct alpha_0 for gc_loss (CT-7.4)
		sigma_t = self._extract_into_tensor(self.sigma_coef, ts, x_t.shape)
		model_output_reconstructed = (1.0 - self.sigma_min) * x_t - sigma_t * v_pred_x

		usr_model_embeds = torch.mm(model_output_reconstructed, model_feats)
		usr_id_embeds = torch.mm(x_start, itmEmbeds)
		gc_loss = self.mean_flat((usr_model_embeds - usr_id_embeds) ** 2)

		return diff_loss, gc_loss
