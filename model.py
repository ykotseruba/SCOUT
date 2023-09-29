import torch
from torch import nn
import torch.nn.functional as F
import math
import os
from collections import OrderedDict
from video_swin_transformer import SwinTransformer3DBackbone
from einops import rearrange
import time
from utils import get_task_attribute_dict

class SCOUT_task(nn.Module):
	def __init__(self, 
				  num_encoder_layers=4,
				  use_task=False,
				  task_attributes=None,
				  train_backbone=False,
				  pretrained_backbone=False,
				  transformer_params=None,
				  **kwargs
				):

		super(SCOUT_task, self).__init__()

		self.backbone = SwinTransformer3DBackbone(pretrained=pretrained_backbone,
												train_backbone=train_backbone)
		self.task_attributes = get_task_attribute_dict(task_attributes)
		self.num_encoder_layers = num_encoder_layers
		if num_encoder_layers in [1, 2, 3, 4]:
			self.decoder = DecoderSwin(num_encoder_layers)
		else:
			raise ValueError(f'ERROR: unsupported num_encoder_layers={num_encoder_layers}, \
							should be between 1 and 4.')

		self.use_task = use_task

		self.add_and_norm = transformer_params['add_and_norm']
		self.fuse_idx = transformer_params['fuse_idx']
		self.num_att_heads = transformer_params['num_att_heads']

		if self.use_task:
			embed_dims = [768, 384, 192, 96]

			self.task_context_model = TaskContextEncoder(clip_size=16, 
									task_attributes=self.task_attributes,
									fuse_idx=self.fuse_idx)

			multihead_attn_layers = [None, None, None, None]
			norm_layers = [None, None, None, None]
			
			for idx in self.fuse_idx:
				multihead_attn_layers[idx] = nn.MultiheadAttention(embed_dim=embed_dims[idx],
														num_heads=self.num_att_heads[idx], 
														bias=True)
				norm_layers[idx] = nn.LayerNorm(embed_dims[idx])


			self.multihead_attn = nn.ModuleList(multihead_attn_layers)
			self.norm = nn.ModuleList(norm_layers)



	def forward(self, x, task_input):

		b_out = self.backbone(x)
		b_s = [b.shape for b in b_out]

		if self.use_task:
			task_enc = self.task_context_model(task_input)
			for idx, b in enumerate(b_out):
				if idx in self.fuse_idx:
					task_enc[idx] = task_enc[idx].flatten(2).permute((2, 0, 1))
					b = b.flatten(2).permute((2, 0, 1))

					fused_out, _ = self.multihead_attn[idx](task_enc[idx], b, b)
				
					if self.add_and_norm:
						fused_out += task_enc[idx]
						fused_out = self.norm[idx](fused_out)

					fused_out = fused_out.permute((1, 2, 0))
					b_out[idx] = fused_out.view(*b_s[idx])
		
		return self.decoder(b_out[:self.num_encoder_layers])


class SCOUT_map_v1(nn.Module):
	def __init__(self, 
				  num_encoder_layers=4,
				  use_map=True,
				  map_params=None,
				  train_backbone=False,
				  pretrained_backbone=False,
				  transformer_params=None,
				  **kwargs
				):

		super(SCOUT_map_v1, self).__init__()

		self.backbone_3d = SwinTransformer3DBackbone(pretrained=pretrained_backbone,
												train_backbone=train_backbone)

		self.num_encoder_layers = num_encoder_layers

		if num_encoder_layers in [1, 2, 3, 4]:
			self.decoder = DecoderSwin(num_encoder_layers)
		else:
			raise ValueError(f'ERROR: unsupported num_encoder_layers={num_encoder_layers}, \
							should be between 1 and 4.')

		self.use_map = use_map

		self.add_and_norm = transformer_params['add_and_norm']
		self.fuse_idx = transformer_params['fuse_idx']
		self.num_att_heads = transformer_params['num_att_heads']

		if self.use_map:
			num_channels = 1 + len([x for x in map_params.keys() if isinstance(map_params[x], bool) and map_params[x]])

			self.map_encoder = MapEncoder(input_size=(num_channels, *map_params['img_size']), fuse_idx=self.fuse_idx)


			multihead_attn_layers = [None, None, None, None]
			norm_layers = [None, None, None, None]

			embed_dims = [768, 384, 192, 96]
			
			for idx in self.fuse_idx:
				multihead_attn_layers[idx] = nn.MultiheadAttention(embed_dim=embed_dims[idx],
														num_heads=self.num_att_heads[idx], 
														bias=True)
				norm_layers[idx] = nn.LayerNorm(embed_dims[idx])


			self.multihead_attn = nn.ModuleList(multihead_attn_layers)
			self.norm = nn.ModuleList(norm_layers)

	def forward(self, x, map_input):

		b_out = self.backbone_3d(x)
		b_s = [b.shape for b in b_out]

		if self.use_map:
			map_enc = self.map_encoder(map_input)

			for idx, b in enumerate(b_out):
				if idx in self.fuse_idx:
					
					#print('map_enc', idx, map_enc[idx].shape)
					#print('b', idx, b.shape)

					map_enc[idx] = map_enc[idx].flatten(2).permute((2, 0, 1))					
					b = b.flatten(2).permute((2, 0, 1))

					fused_out, _ = self.multihead_attn[idx](map_enc[idx], b, b)
				
					if self.add_and_norm:
						fused_out += map_enc[idx]
						fused_out = self.norm[idx](fused_out)

					fused_out = fused_out.permute((1, 2, 0))
					b_out[idx] = fused_out.view(*b_s[idx])
		
		return self.decoder(b_out[:self.num_encoder_layers])


# model with only map input
class SCOUT_map_v2(nn.Module):
	def __init__(self, 
				  num_encoder_layers=4,
				  use_map=True,
				  map_params=None,
				  train_backbone=False,
				  pretrained_backbone=False,
				  transformer_params=None,
				  **kwargs
				):

		super(SCOUT_map_v2, self).__init__()

		#self.backbone_3d = SwinTransformer3DBackbone(pretrained=pretrained_backbone,
		#											train_backbone=train_backbone)

		self.num_encoder_layers = num_encoder_layers

		if num_encoder_layers in [1, 2, 3, 4]:
			self.decoder = DecoderSwin(num_encoder_layers)
		else:
			raise ValueError(f'ERROR: unsupported num_encoder_layers={num_encoder_layers}, \
							should be between 1 and 4.')

		self.use_map = use_map

		self.add_and_norm = transformer_params['add_and_norm']
		self.fuse_idx = transformer_params['fuse_idx']
		self.num_att_heads = transformer_params['num_att_heads']

		num_channels = 1 + map_params['obs_traj'] + map_params['coords'] + map_params['dist']

		self.map_encoder = MapEncoder(input_size=(num_channels, *map_params['img_size']), fuse_idx=self.fuse_idx)

	def forward(self, x, map_input):

		b_out = self.map_encoder(map_input)		
		return self.decoder(b_out[:self.num_encoder_layers]) 


class TaskContextEncoder(nn.Module):
	def __init__(self,
				 clip_size=16,
				 dict_len=30,
				 fuse_idx=(0, 1, 2, 3),
				 task_attributes=None):
		super(TaskContextEncoder, self).__init__()
		self.embedding = nn.Embedding(dict_len, clip_size//4)
		self.relu = nn.ReLU()
		self.task_attributes = get_task_attribute_dict(task_attributes)
		self.fuse_idx = fuse_idx
		self.emb_dims = (768, 384, 192, 96)
		self.repl_dims = (49, 196, 784, 3136)
		num_features = len([k for k, v in self.task_attributes.items() if v]) # number of task and context features
		if num_features == 0:
			raise ValueError('ERROR: no task attributes provided')

		dense_layers = [None, None, None, None]

		for idx in fuse_idx:
			dense_layers[idx] = nn.Linear(num_features, self.emb_dims[idx]) 
		self.dense = nn.ModuleList(dense_layers)


	def forward(self, task_context):

		for k in task_context.keys():	
			if len(task_context[k].shape) == 2:
				task_context[k] = self.embedding(task_context[k])
			else:
				task_context[k] = task_context[k][:, :, ::4]
		#print(task_context.keys())
		task_context_enc = torch.stack([v for k,v in task_context.items()], dim=1)
		#print('task_context_enc', task_context_enc.shape)

		task_context_enc = task_context_enc.permute((0, 3, 1, 2)).flatten(2)
		#print('task_context_enc', task_context_enc.shape)
		task = [None, None, None, None]
		for idx in self.fuse_idx:
			task[idx] = self.relu(self.dense[idx](task_context_enc)).permute((0, 2, 1))
			task[idx] = task[idx][:,:,:, None]
			task[idx] = task[idx].repeat(1, 1, 1, self.repl_dims[idx])
			#print('task', idx, task[idx].shape)
		return task

# Map encoder modified from
# https://github.com/StanfordASL/Trajectron-plus-plus/blob/1031c7bd1a444273af378c1ec1dcca907ba59830/trajectron/model/components/map_encoder.py
class MapEncoder(nn.Module):
	def __init__(self,
				 input_size=(1, 128, 128),
				 fuse_idx=(0, 1, 2, 3)):
		super(MapEncoder, self).__init__()

		hidden_channels = [10, 20, 10, 1]
		kernel_size = (5, 3, 3, 1)
		strides = (2, 2, 1, 1)
		output_size = (56, 56)

		self.fuse_idx = fuse_idx
		self.repl_dims = ([1, 768, 4, 1, 1], [1, 384, 4, 1, 1], [1, 192, 4, 1, 1], [1, 96, 4, 1, 1])

		self.convs = nn.ModuleList()
		self.post = nn.ModuleList()

		x_dummy = torch.ones(input_size).unsqueeze(0) * torch.tensor(float('nan'))

		for i, hidden_size in enumerate(hidden_channels):
			self.convs.append(nn.Conv2d(input_size[0] if i == 0 else hidden_channels[i-1],
										hidden_channels[i], kernel_size[i],
										stride=strides[i]))
			x_dummy = self.convs[i](x_dummy)
			#print(x_dummy.shape)
		
		self.post.append(nn.AvgPool2d(kernel_size=1, stride=4))
		self.post.append(nn.AvgPool2d(kernel_size=1, stride=2))
		self.post.append(nn.Identity())
		self.post.append(nn.Upsample(size=output_size))


	def forward(self, x):
		for conv in self.convs:
			x = F.leaky_relu(conv(x), 0.2)
		
		map_enc = [None, None, None, None]
		for i in range(len(self.post)):
			if i in self.fuse_idx:
				#print('x', i, x.shape)
				map_enc[i] = self.post[i](x)[:,:,None,:,:].repeat(self.repl_dims[i])
				#print('map_enc', i, map_enc[i].shape)
		return map_enc

class DecoderSwin(nn.Module):
	def __init__(self, num_layers=4):
		super(DecoderSwin, self).__init__()
		
		self.upsampling = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False)
		
		self.convtsp1 = nn.Sequential(
			nn.Conv3d(768, 384, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling
		)

		x = 1 if num_layers == 1 else 3

		self.convtsp2 = nn.Sequential(
			nn.Conv3d(384, 192, kernel_size=(x, 3, 3), stride=(x, 1, 1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling
		)

		x = 1 if num_layers < 4 else 5

		self.convtsp3 = nn.Sequential(
			nn.Conv3d(192, 96, kernel_size=(x,3,3), stride=(x,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling
		)

		x = 1 if num_layers < 3 else 5
		layers = [('conv3_1', nn.Conv3d(96, 64, kernel_size=(x,3,3), stride=(x,1,1), padding=(0,1,1), bias=False)),
							  ('relu_1', nn.ReLU()),
							  ('up_1', self.upsampling),
							  ('conv3_2', nn.Conv3d(64, 32, kernel_size=(1,3,3), stride=(2,1,1), padding=(0,1,1), bias=False)),
							  ('relu_2', nn.ReLU()),
							  ('up_2', self.upsampling)
							  
				]
		if num_layers == 1:
			layers.append(('conv3_3', nn.Conv3d(32, 1, kernel_size=(1,1,1), stride=(2,1,1), bias=True)))
		else:
			layers.append(('conv3_3', nn.Conv3d(32, 1, kernel_size=(1,1,1), stride=(1,1,1), bias=True)))

		layers.append(('sigm', nn.Sigmoid()))

		self.convtsp4 = nn.Sequential(OrderedDict(layers))

	def forward(self, y):
		if not isinstance(y, list):
			raise ValueError(f'ERROR: input to decoder should be a list!')

		if len(y) >= 1:
			z = self.convtsp1(y[0])

		if len(y) >= 2:
			z = torch.cat((z,y[1]), 2)
		
		z = self.convtsp2(z)

		if len(y) >= 3:
			z = torch.cat((z,y[2]), 2)
		
		z = self.convtsp3(z)

		if len(y) == 4:
			z = torch.cat((z,y[3]), 2)
		
		z = self.convtsp4(z)
		
		z = z.view(z.size(0), z.size(3), z.size(4))
		return z

