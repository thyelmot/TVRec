import torch
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import Model, GaussianDiffusionTVS, Denoise
from DataHandler import DataHandler
import numpy as np
from Utils.Utils import *
import os
import scipy.sparse as sp
import random
import tempfile
import setproctitle
from scipy.sparse import coo_matrix

class Coach:
	def __init__(self, handler):
		self.handler = handler

		print('USER', args.user, 'ITEM', args.item)
		print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'Recall', 'NDCG', 'Precision']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Validation' + met] = list()
			self.metrics['Test' + met] = list()

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.5f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		if args.epoch < 1 or args.tstEpoch < 1 or args.patience < 0:
			raise ValueError('epoch and tstEpoch must be positive; patience must be nonnegative')
		self.prepareModel()
		log('Model Prepared')
		os.makedirs(args.checkpoint_dir, exist_ok=True)
		run_dir = tempfile.mkdtemp(prefix=f'{args.data}_seed{args.seed}_', dir=args.checkpoint_dir)
		self.checkpoint_path = os.path.join(run_dir, 'best.pt')
		log(f'Best checkpoint: {self.checkpoint_path}')

		recallMax = -float('inf')
		bestEpoch = -1
		for ep in range(args.epoch):
			valFlag = (ep % args.tstEpoch == 0 or ep == args.epoch - 1)
			reses = self.trainEpoch()
			log(self.makePrint('Train', ep, reses, valFlag))
			if valFlag:
				reses = self.testEpoch(self.handler.valLoader)
				if not all(np.isfinite(value) for value in reses.values()):
					raise ValueError('Non-finite validation metrics; refusing to select a checkpoint')
				if reses['Recall'] > recallMax:
					recallMax = reses['Recall']
					bestEpoch = ep
					self.saveCheckpoint(ep, reses)
				log(self.makePrint('Validation', ep, reses, True))
				if args.patience > 0 and ep - bestEpoch >= args.patience:
					log(f'Early stopping at epoch {ep}; best validation epoch: {bestEpoch}')
					break
			print()

		checkpoint = self.loadCheckpoint()
		log(self.makePrint('Best Validation', checkpoint['epoch'], checkpoint['validation'], False))
		# Test is evaluated once, after restoring the validation-selected state.
		reses = self.testEpoch()
		log(self.makePrint('Test', checkpoint['epoch'], reses, True))
		return reses

	def saveCheckpoint(self, epoch, validation):
		modalities = ['image', 'text', 'audio'] if args.data == 'tiktok' else ['image', 'text']
		checkpoint = {
			'epoch': epoch,
			'validation': {key: float(value) for key, value in validation.items()},
			'args': vars(args).copy(),
			'model': {key: value.detach().cpu() for key, value in self.model.state_dict().items()},
			'denoisers': {},
			'graphs': {},
		}
		for modality in modalities:
			denoiser = getattr(self, 'denoise_model_' + modality)
			checkpoint['denoisers'][modality] = {key: value.detach().cpu() for key, value in denoiser.state_dict().items()}
			# Preserve the exact graph, including the epoch's sampled edge dropout.
			checkpoint['graphs'][modality] = getattr(self, modality + '_UI_matrix').coalesce().cpu()
		torch.save(checkpoint, self.checkpoint_path + '.tmp')
		os.replace(self.checkpoint_path + '.tmp', self.checkpoint_path)

	def loadCheckpoint(self):
		checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=True)
		self.model.load_state_dict(checkpoint['model'])
		device = next(self.model.parameters()).device
		for modality, state in checkpoint['denoisers'].items():
			getattr(self, 'denoise_model_' + modality).load_state_dict(state)
			setattr(self, modality + '_UI_matrix', checkpoint['graphs'][modality].to(device))
		return checkpoint

	def prepareModel(self):
		if args.data == 'tiktok':
			self.model = Model(self.handler.image_feats.detach(), self.handler.text_feats.detach(), self.handler.audio_feats.detach()).cuda()
		else:
			self.model = Model(self.handler.image_feats.detach(), self.handler.text_feats.detach()).cuda()
		self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

		self.diffusion_model = GaussianDiffusionTVS(args.sigma_min, args.steps, w_clip=args.w_clip, num_sample_steps=args.num_sample_steps, anchor_w=args.anchor_w, lambda_x=args.lambda_x, lambda_y=args.lambda_y, lambda_z=args.lambda_z).cuda()
		
		out_dims = eval(args.dims) + [args.item]
		in_dims = out_dims[::-1]
		self.denoise_model_image = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
		self.denoise_opt_image = torch.optim.Adam(self.denoise_model_image.parameters(), lr=args.lr, weight_decay=0)

		out_dims = eval(args.dims) + [args.item]
		in_dims = out_dims[::-1]
		self.denoise_model_text = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
		self.denoise_opt_text = torch.optim.Adam(self.denoise_model_text.parameters(), lr=args.lr, weight_decay=0)

		if args.data == 'tiktok':
			out_dims = eval(args.dims) + [args.item]
			in_dims = out_dims[::-1]
			self.denoise_model_audio = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
			self.denoise_opt_audio = torch.optim.Adam(self.denoise_model_audio.parameters(), lr=args.lr, weight_decay=0)

	def normalizeAdj(self, mat): 
		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

	def buildUIMatrix(self, u_list, i_list, edge_list):
		mat = coo_matrix((edge_list, (u_list, i_list)), shape=(args.user, args.item), dtype=np.float32)

		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((args.item, args.item))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		mat = (mat + sp.eye(mat.shape[0])) * 1.0
		mat = self.normalizeAdj(mat)

		idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = torch.from_numpy(mat.data.astype(np.float32))
		shape = torch.Size(mat.shape)

		return torch.sparse.FloatTensor(idxs, vals, shape).cuda()

	def trainEpoch(self):
		self.model.train()
		trnLoader = self.handler.trnLoader
		trnLoader.dataset.negSampling()
		epLoss, epRecLoss, epClLoss = 0, 0, 0
		epDiLoss = 0
		epDiLoss_image, epDiLoss_text = 0, 0
		if args.data == 'tiktok':
			epDiLoss_audio = 0
		steps = trnLoader.dataset.__len__() // args.batch

		diffusionLoader = self.handler.diffusionLoader

		for i, batch in enumerate(diffusionLoader):
			batch_item, batch_index = batch
			batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

			iEmbeds = self.model.getItemEmbeds().detach()
			uEmbeds = self.model.getUserEmbeds().detach()

			image_feats = self.model.getImageFeats().detach()
			text_feats = self.model.getTextFeats().detach()
			if args.data == 'tiktok':
				audio_feats = self.model.getAudioFeats().detach()

			self.denoise_opt_image.zero_grad()
			self.denoise_opt_text.zero_grad()
			if args.data == 'tiktok':
				self.denoise_opt_audio.zero_grad()

			uEmbeds_batch = uEmbeds[batch_index]  # Diem neo cho cac quy dao TVS.

			diff_loss_image, gc_loss_image = self.diffusion_model.training_losses(self.denoise_model_image, batch_item, iEmbeds, batch_index, image_feats, uEmbeds_batch)
			diff_loss_text, gc_loss_text = self.diffusion_model.training_losses(self.denoise_model_text, batch_item, iEmbeds, batch_index, text_feats, uEmbeds_batch)
			if args.data == 'tiktok':
				diff_loss_audio, gc_loss_audio = self.diffusion_model.training_losses(self.denoise_model_audio, batch_item, iEmbeds, batch_index, audio_feats, uEmbeds_batch)

			loss_image = diff_loss_image.mean() + gc_loss_image.mean() * args.e_loss
			loss_text = diff_loss_text.mean() + gc_loss_text.mean() * args.e_loss
			if args.data == 'tiktok':
				loss_audio = diff_loss_audio.mean() + gc_loss_audio.mean() * args.e_loss

			epDiLoss_image += loss_image.item()
			epDiLoss_text += loss_text.item()
			if args.data == 'tiktok':
				epDiLoss_audio += loss_audio.item()

			if args.data == 'tiktok':
				loss = loss_image + loss_text + loss_audio
			else:
				loss = loss_image + loss_text

			loss.backward()

			self.denoise_opt_image.step()
			self.denoise_opt_text.step()
			if args.data == 'tiktok':
				self.denoise_opt_audio.step()

			log('Diffusion Step %d/%d' % (i, diffusionLoader.dataset.__len__() // args.batch), save=False, oneline=True)

		log('')
		log('Start to re-build UI matrix')

		with torch.no_grad():

			u_list_image = []
			i_list_image = []
			edge_list_image = []

			u_list_text = []
			i_list_text = []
			edge_list_text = []

			if args.data == 'tiktok':
				u_list_audio = []
				i_list_audio = []
				edge_list_audio = []

			for _, batch in enumerate(diffusionLoader):
				batch_item, batch_index = batch
				batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

				# image
				denoised_batch = self.diffusion_model.p_sample(self.denoise_model_image, batch_item, uEmbeds[batch_index], iEmbeds, args.sampling_steps, args.sampling_noise)
				top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)

				for i in range(batch_index.shape[0]):
					for j in range(indices_[i].shape[0]): 
						u_list_image.append(int(batch_index[i].cpu().numpy()))
						i_list_image.append(int(indices_[i][j].cpu().numpy()))
						edge_list_image.append(1.0)

				# text
				denoised_batch = self.diffusion_model.p_sample(self.denoise_model_text, batch_item, uEmbeds[batch_index], iEmbeds, args.sampling_steps, args.sampling_noise)
				top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)

				for i in range(batch_index.shape[0]):
					for j in range(indices_[i].shape[0]): 
						u_list_text.append(int(batch_index[i].cpu().numpy()))
						i_list_text.append(int(indices_[i][j].cpu().numpy()))
						edge_list_text.append(1.0)

				if args.data == 'tiktok':
					# audio
					denoised_batch = self.diffusion_model.p_sample(self.denoise_model_audio, batch_item, uEmbeds[batch_index], iEmbeds, args.sampling_steps, args.sampling_noise)
					top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)

					for i in range(batch_index.shape[0]):
						for j in range(indices_[i].shape[0]): 
							u_list_audio.append(int(batch_index[i].cpu().numpy()))
							i_list_audio.append(int(indices_[i][j].cpu().numpy()))
							edge_list_audio.append(1.0)

			# image
			u_list_image = np.array(u_list_image)
			i_list_image = np.array(i_list_image)
			edge_list_image = np.array(edge_list_image)
			self.image_UI_matrix = self.buildUIMatrix(u_list_image, i_list_image, edge_list_image)
			self.image_UI_matrix = self.model.edgeDropper(self.image_UI_matrix)

			# text
			u_list_text = np.array(u_list_text)
			i_list_text = np.array(i_list_text)
			edge_list_text = np.array(edge_list_text)
			self.text_UI_matrix = self.buildUIMatrix(u_list_text, i_list_text, edge_list_text)
			self.text_UI_matrix = self.model.edgeDropper(self.text_UI_matrix)

			if args.data == 'tiktok':
				# audio
				u_list_audio = np.array(u_list_audio)
				i_list_audio = np.array(i_list_audio)
				edge_list_audio = np.array(edge_list_audio)
				self.audio_UI_matrix = self.buildUIMatrix(u_list_audio, i_list_audio, edge_list_audio)
				self.audio_UI_matrix = self.model.edgeDropper(self.audio_UI_matrix)

		log('UI matrix built!')

		for i, tem in enumerate(trnLoader):
			ancs, poss, negs = tem
			ancs = ancs.long().cuda()
			poss = poss.long().cuda()
			negs = negs.long().cuda()

			self.opt.zero_grad()

			if args.data == 'tiktok':
				usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix, self.audio_UI_matrix)
			else:
				usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix)
			ancEmbeds = usrEmbeds[ancs]
			posEmbeds = itmEmbeds[poss]
			negEmbeds = itmEmbeds[negs]
			scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
			bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch
			regLoss = self.model.reg_loss() * args.reg
			loss = bprLoss + regLoss
			
			epRecLoss += bprLoss.item()
			epLoss += loss.item()

			if args.data == 'tiktok':
				usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2, usrEmbeds3, itmEmbeds3 = self.model.forward_cl_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix, self.audio_UI_matrix)
			else:
				usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2 = self.model.forward_cl_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix)
			if args.data == 'tiktok':
				clLoss = (contrastLoss(usrEmbeds1, usrEmbeds2, ancs, args.temp) + contrastLoss(itmEmbeds1, itmEmbeds2, poss, args.temp)) * args.ssl_reg
				clLoss += (contrastLoss(usrEmbeds1, usrEmbeds3, ancs, args.temp) + contrastLoss(itmEmbeds1, itmEmbeds3, poss, args.temp)) * args.ssl_reg
				clLoss += (contrastLoss(usrEmbeds2, usrEmbeds3, ancs, args.temp) + contrastLoss(itmEmbeds2, itmEmbeds3, poss, args.temp)) * args.ssl_reg
			else:
				clLoss = (contrastLoss(usrEmbeds1, usrEmbeds2, ancs, args.temp) + contrastLoss(itmEmbeds1, itmEmbeds2, poss, args.temp)) * args.ssl_reg

			clLoss1 = (contrastLoss(usrEmbeds, usrEmbeds1, ancs, args.temp) + contrastLoss(itmEmbeds, itmEmbeds1, poss, args.temp)) * args.ssl_reg
			clLoss2 = (contrastLoss(usrEmbeds, usrEmbeds2, ancs, args.temp) + contrastLoss(itmEmbeds, itmEmbeds2, poss, args.temp)) * args.ssl_reg
			if args.data == 'tiktok':
				clLoss3 = (contrastLoss(usrEmbeds, usrEmbeds3, ancs, args.temp) + contrastLoss(itmEmbeds, itmEmbeds3, poss, args.temp)) * args.ssl_reg
				clLoss_ = clLoss1 + clLoss2 + clLoss3
			else:
				clLoss_ = clLoss1 + clLoss2

			if args.cl_method == 1:
				clLoss = clLoss_

			loss += clLoss

			epClLoss += clLoss.item()

			loss.backward()
			self.opt.step()

			log('Step %d/%d: bpr : %.3f ; reg : %.3f ; cl : %.3f ' % (
				i, 
				steps,
				bprLoss.item(),
        regLoss.item(),
				clLoss.item()
				), save=False, oneline=True)

		ret = dict()
		ret['Loss'] = epLoss / steps
		ret['BPR Loss'] = epRecLoss / steps
		ret['CL loss'] = epClLoss / steps
		ret['Di image loss'] = epDiLoss_image / (diffusionLoader.dataset.__len__() // args.batch)
		ret['Di text loss'] = epDiLoss_text / (diffusionLoader.dataset.__len__() // args.batch)
		if args.data == 'tiktok':
			ret['Di audio loss'] = epDiLoss_audio / (diffusionLoader.dataset.__len__() // args.batch)
		return ret

	@torch.no_grad()
	def testEpoch(self, tstLoader=None):
		self.model.eval()
		if tstLoader is None:
			tstLoader = self.handler.tstLoader
		epRecall, epNdcg, epPrecision = [0] * 3
		i = 0
		num = tstLoader.dataset.__len__()
		if num == 0 or not 1 <= args.topk <= args.item:
			raise ValueError('Evaluation needs a nonempty split and 1 <= topk <= item count')
		steps = len(tstLoader)

		if args.data == 'tiktok':
			usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix, self.audio_UI_matrix)
		else:
			usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix)

		for usr, trnMask in tstLoader:
			i += 1
			usr = usr.long().cuda()
			trnMask = trnMask.cuda()
			allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0)).masked_fill(trnMask.bool(), -float('inf'))
			_, topLocs = torch.topk(allPreds, args.topk)
			recall, ndcg, precision = self.calcRes(topLocs.cpu().numpy(), tstLoader.dataset.tstLocs, usr.cpu().tolist())
			epRecall += recall
			epNdcg += ndcg
			epPrecision += precision
			log('Steps %d/%d: recall = %.2f, ndcg = %.2f , precision = %.2f   ' % (i, steps, recall, ndcg, precision), save=False, oneline=True)
		ret = dict()
		ret['Recall'] = epRecall / num
		ret['NDCG'] = epNdcg / num
		ret['Precision'] = epPrecision / num
		return ret

	def calcRes(self, topLocs, tstLocs, batIds):
		assert topLocs.shape[0] == len(batIds)
		allRecall = allNdcg = allPrecision = 0
		for i in range(len(batIds)):
			temTopLocs = list(topLocs[i])
			temTstLocs = tstLocs[batIds[i]]
			tstNum = len(temTstLocs)
			maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
			recall = dcg = precision = 0
			for val in temTstLocs:
				if val in temTopLocs:
					recall += 1
					dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
					precision += 1
			recall = recall / tstNum
			ndcg = dcg / maxDcg
			precision = precision / args.topk
			allRecall += recall
			allNdcg += ndcg
			allPrecision += precision
		return allRecall, allNdcg, allPrecision

def seed_it(seed):
	random.seed(seed)
	os.environ["PYTHONSEED"] = str(seed)
	np.random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True 
	torch.backends.cudnn.enabled = True
	torch.manual_seed(seed)

def self_check():
	"""Small CPU check for TVS and validation-selected checkpoint restoration."""
	from types import SimpleNamespace
	from unittest.mock import patch
	from DataHandler import TstData

	# ponytail: synthetic smoke check; extend here for new failure cases.
	with tempfile.TemporaryDirectory() as directory, \
		patch.dict(vars(args), data='tiktok', user=2, item=4, epoch=3, tstEpoch=1, patience=1, checkpoint_dir=directory), \
		patch.object(torch.Tensor, 'cuda', lambda self, *a, **k: self):
		x = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])
		u, items = torch.ones(2, 3), torch.ones(4, 3)
		diffusion = GaussianDiffusionTVS(0.001, 3, anchor_w=2.)
		anchor = diffusion._compute_anchor(u, items)
		for t in range(3):
			ts = torch.full((2,), t, dtype=torch.long)
			noise = torch.randn_like(x)
			noisy = diffusion.q_sample(x, anchor, ts, noise)
			velocity = (1 - diffusion.sigma_min) * (2 * anchor + noise) - x
			sigma = diffusion._extract_into_tensor(diffusion.sigma_coef, ts, x.shape)
			torch.testing.assert_close((1 - diffusion.sigma_min) * noisy - sigma * velocity, x, rtol=0, atol=2e-6)
		denoiser = Denoise([4, 3], [3, 4], 4)
		losses = diffusion.training_losses(denoiser, x, items, torch.arange(2), items, u)
		assert all(torch.isfinite(loss).all() for loss in losses)
		sum(loss.mean() for loss in losses).backward()
		assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in denoiser.parameters())

		train = sp.coo_matrix(([1.], ([0], [0])), shape=(2, 4))
		val = sp.coo_matrix(([1.], ([0], [1])), shape=(2, 4))
		test = sp.coo_matrix(([1.], ([0], [2])), shape=(2, 4))
		assert np.array_equal(TstData(val, train)[0][1], [1, 0, 0, 0])
		assert np.array_equal(TstData(test, train + val)[0][1], [1, 1, 0, 0])

		class CheckCoach(Coach):
			def prepareModel(self):
				self.model = torch.nn.Linear(1, 1)
				self.trained = self.test_calls = 0
				for modality in ('image', 'text', 'audio'):
					setattr(self, 'denoise_model_' + modality, torch.nn.Linear(1, 1))

			def trainEpoch(self):
				self.trained += 1
				with torch.no_grad():
					self.model.weight.fill_(self.trained)
					for modality in ('image', 'text', 'audio'):
						getattr(self, 'denoise_model_' + modality).weight.fill_(self.trained)
						setattr(self, modality + '_UI_matrix', (torch.eye(2) * self.trained).to_sparse())
				return {'Loss': 0.}

			def testEpoch(self, loader=None):
				if loader is not None:
					assert loader is self.handler.valLoader
					return {'Recall': [0., .5, .2][self.trained - 1], 'NDCG': np.float64(0.)}
				self.test_calls += 1
				assert self.trained == 3 and self.model.weight.item() == 2
				for modality in ('image', 'text', 'audio'):
					assert getattr(self, 'denoise_model_' + modality).weight.item() == 2
					torch.testing.assert_close(getattr(self, modality + '_UI_matrix').to_dense(), torch.eye(2) * 2)
				return {'Recall': 1.}

		coach = CheckCoach(SimpleNamespace(trnLoader=SimpleNamespace(dataset=[1]), valLoader=object()))
		coach.run()
		assert coach.test_calls == 1 and len(coach.metrics['ValidationRecall']) == 3
		assert coach.loadCheckpoint()['epoch'] == 1
	print('PASS: TVS reconstruction/backward, masks, best checkpoint and final-only test.')

if __name__ == '__main__':
	if args.self_check:
		self_check()
		raise SystemExit(0)
	seed_it(args.seed)

	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	logger.saveDefault = True
	
	log('Start')
	handler = DataHandler()
	handler.LoadData()
	log('Load Data')

	coach = Coach(handler)
	coach.run()