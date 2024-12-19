import torch
import torch.nn as nn
from model import recons_video_ori
from model import swinir
from util import utils
from model import SearchTransfer
import torch.nn.functional as F
from model.edgeinformation import *
import warnings
warnings.filterwarnings("ignore")
def activation_statistics(features):
    features_np = features.detach().cpu().numpy().reshape(-1)
    mean = np.mean(features_np)
    std = np.std(features_np)
    return mean, std

def make_model(args):
    device = 'cpu' if args.cpu else 'cuda'
    load_recons_net = False
    recons_pretrain_fn = ''
    is_mask_filter = True
    return SPEINet(in_channels=args.n_colors, n_sequence=args.n_sequence, out_channels=args.n_colors,
                    n_resblock=args.n_resblock, n_feat=args.n_feat,
                    load_recons_net=load_recons_net, recons_pretrain_fn=recons_pretrain_fn,
                    is_mask_filter=is_mask_filter, device=device, args=args)

class SPEINet(nn.Module):
    def __init__(self, in_channels=3, n_sequence=3, out_channels=3, n_resblock=3, n_feat=32,
                 load_flow_net=False, load_recons_net=False, flow_pretrain_fn='', recons_pretrain_fn='',
                 is_mask_filter=False, device='cuda', args=None):
        super(SPEINet, self).__init__()
        print("Creating SPEINet...")

        self.n_sequence = n_sequence
        self.device = device
        self.is_mask_filter = is_mask_filter
        extra_channels = 0
        print('Select mask mode: concat, num_mask={}'.format(extra_channels))

        self.swin = swinir.SwinIR(upscale=1,
                                          in_chans=n_feat * 4,
                                          img_size=args.patch_size // 4,
                                          window_size=args.window_size,
                                          img_range=args.rgb_range,
                                          depths=args.depths,
                                          embed_dim=args.embed_dim,
                                          num_heads=args.num_heads,
                                          mlp_ratio=args.mlp_ratio,
                                          resi_connection=args.resi_connection)
        self.recons_net = recons_video_ori.RECONS_VIDEO(in_channels=in_channels, n_sequence=5, out_channels=out_channels,
                                                        n_resblock=n_resblock, n_feat=n_feat,
                                                        extra_channels=extra_channels)
        self.SearchTransfer = SearchTransfer.SearchTransfer()
        self.SelfTransfer = SearchTransfer.SelfTransfer()
        self.conv_lv1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1, stride=1, padding=0)
        self.conv_lv2 = nn.Conv2d(n_feat * 2 * 2, n_feat * 2, kernel_size=1, stride=1, padding=0)
        self.conv_lv3 = nn.Conv2d(n_feat * 4 * 2, n_feat * 4, kernel_size=1, stride=1, padding=0)
        self.fusion = nn.Conv2d(n_feat * 4 * n_sequence, n_feat * 4, kernel_size=1, stride=1, padding=0)
        self.connect = nn.Conv2d(n_feat * 4 * 2, n_feat * 4, kernel_size=1, stride=1, padding=0)
        self.search3 = nn.Conv2d(n_feat * 2, n_feat * 2, kernel_size=3, stride=1, padding=1)
        self.search2 = nn.Conv2d(n_feat * 4, n_feat * 2, kernel_size=1, stride=1, padding=0)
        self.search1 = nn.Conv2d(n_feat * 4, n_feat * 2, kernel_size=1, stride=1, padding=0)
        self.search43 = nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1)
        self.search33 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=3, stride=1, padding=1)
        self.search23 = nn.Conv2d(n_feat * 4, n_feat, kernel_size=1, stride=1, padding=0)
        self.search13 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1, stride=1, padding=0)
        if load_recons_net:
            self.recons_net.load_state_dict(torch.load(recons_pretrain_fn))
            print('Loading reconstruction pretrain model from {}'.format(recons_pretrain_fn))

    def _forwardx(self, x):
        batch_size, S, channels, height, width = x.shape
        batch_4_all_zeros = torch.all(x[:, 3, :, :, :] == 0, dim=1).all(dim=1).all(dim=1)
        batch_5_all_zeros = torch.all(x[:, 4, :, :, :] == 0, dim=1).all(dim=1).all(dim=1)
        
        return batch_4_all_zeros, batch_5_all_zeros

    def _process(self, frame_list, f_mid):
        blur_kernel = create_blur_kernel().to(self.device)
        f_fusion = f_mid
        for i in range(self.n_sequence):
            if i == self.n_sequence // 2:
                continue
            deblurred_tensor = r_l_per_channel(frame_list[i], blur_kernel, 1, 0.01)
            features = self.recons_net.encoder_second(self.recons_net.encoder_first(self.recons_net.inBlock(frame_list[i])))
            fea_deblurred = self.recons_net.encoder_second(self.recons_net.encoder_first(self.recons_net.inBlock(deblurred_tensor)))
            f_features = features + fea_deblurred
            f_trans = self.swin(f_mid, f_features)
            f_fusion = torch.cat((f_fusion, f_trans), dim=1)
        if self.n_sequence == 1:
            f_trans = self.swin(f_mid, f_mid)
            f_fusion = f_fusion + f_trans
        return f_fusion

    def _decode(self, fusion, weight_S, sharp_T_lv3, sharp_T_lv2, sharp_T_lv1):
        # sharp_v3 = self.conv_lv3(torch.cat((fusion, sharp_T_lv3), dim=1)) * weight_S
        # fusion_lv3 = fusion + sharp_v3
        decoder_v2 = self.recons_net.decoder_second(fusion)
        # soft_v2 = self.conv_lv2(torch.cat((decoder_v2, sharp_T_lv2), dim=1)) * F.interpolate(weight_S, scale_factor=2, mode='bicubic')
        # soft_lv2 = decoder_v2 + soft_v2

        # search_1 = F.interpolate(fusion_lv3, scale_factor=2, mode='bicubic')
        # search_1 = F.relu(self.search1(search_1))
        # search_2 = F.relu(self.search3(soft_lv2))
        # search_11 = F.relu(self.search2(torch.cat((decoder_v2, search_1), dim=1)))
        # search_22 = F.relu(self.search2(torch.cat((soft_lv2, search_2), dim=1)))
        # soft_lv3 = decoder_v2 + search_11
        # soft_lv2 = soft_lv2 + search_22

        decoder_v1 = self.recons_net.decoder_first(decoder_v2)
        # soft_v1 = self.conv_lv1(torch.cat((decoder_v1, sharp_T_lv1), dim=1)) * F.interpolate(weight_S, scale_factor=4, mode='bicubic')
        # soft_lv1 = decoder_v1 + soft_v1

        # search_13 = F.interpolate(soft_lv3, scale_factor=2, mode='bicubic')
        # search_13 = F.relu(self.search13(search_13))
        # search_23 = F.interpolate(soft_lv2, scale_factor=2, mode='bicubic')
        # search_23 = F.relu(self.search33(search_23))
        # search_33 = F.relu(self.search43(soft_lv1))
        # search_113 = F.relu(self.search33(torch.cat((search_13, search_23), dim=1)))
        # search_223 = F.relu(self.search33(torch.cat((search_13, search_33), dim=1)))
        # search_323 = F.relu(self.search33(torch.cat((search_23, search_33), dim=1)))
        # soft_lv1 = soft_lv1 + search_113 + search_223 + search_323
        return self.recons_net.outBlock(decoder_v1)

    def _forwardbs(self, x):
        frame_list = [x[:, i, :, :, :] for i in range(self.n_sequence)]
        sharp_frame = x[:, self.n_sequence + 1, :, :, :]
        sharp_lv1 = self.recons_net.inBlock(sharp_frame)
        sharp_lv2 = self.recons_net.encoder_first(sharp_lv1)
        sharp_lv3 = self.recons_net.encoder_second(sharp_lv2)
        # blur_kernel = create_blur_kernel().to(self.device)
        # deblurred_tensor = r_l_per_channel(frame_list[self.n_sequence // 2], blur_kernel, 1, 0.01)
        feature_mid1 = self.recons_net.encoder_second(self.recons_net.encoder_first(self.recons_net.inBlock(frame_list[self.n_sequence // 2])))
        # feature_mid2 = self.recons_net.encoder_second(self.recons_net.encoder_first(self.recons_net.inBlock(deblurred_tensor)))
        feature_mid = feature_mid1# + feature_mid2
        fusion = self._process(frame_list, feature_mid)
        fusion = self.fusion(fusion)
        weight_S, sharp_T_lv3, sharp_T_lv2, sharp_T_lv1 = self.SearchTransfer(fusion, sharp_lv3, sharp_lv1, sharp_lv2, sharp_lv3)
        return self._decode(fusion, weight_S, sharp_T_lv3, sharp_T_lv2, sharp_T_lv1)

    def _forwardb(self, x):
        frame_list = [x[:, i, :, :, :] for i in range(self.n_sequence)]
        # blur_kernel = create_blur_kernel().to(self.device)
        # deblurred_tensor = r_l_per_channel(frame_list[self.n_sequence // 2], blur_kernel, 1, 0.01)
        feature_mid1 = self.recons_net.encoder_second(self.recons_net.encoder_first(self.recons_net.inBlock(frame_list[self.n_sequence // 2])))
        # feature_mid2 = self.recons_net.encoder_second(self.recons_net.encoder_first(self.recons_net.inBlock(deblurred_tensor)))
        feature_mid = feature_mid1 #+ feature_mid2
        fusion = self._process(frame_list, feature_mid)
        fusion = self.fusion(fusion)
        weight_S, sharp_T_lv3, sharp_T_lv2, sharp_T_lv1 = self.SelfTransfer(fusion)
        return self._decode(fusion, weight_S, sharp_T_lv3, sharp_T_lv2, sharp_T_lv1)

    def forward(self, x):
        batch_4_all_zeros, batch_5_all_zeros = self._forwardx(x)
        final_output = torch.empty((x.shape[0], x.shape[2], x.shape[3], x.shape[4]), device=x.device)
    
        # 执行模型 blur
        if batch_4_all_zeros.any():
            x_a = x[batch_4_all_zeros]
            # print(f"Executing Model A for batches: {torch.where(batch_4_all_zeros)[0].tolist()}")
            outputs_a = self._forwardb(x_a)
            final_output[batch_4_all_zeros] = outputs_a
    
        # 执行模型 blurs
        if (~batch_4_all_zeros).any():
            x_b = x[~batch_4_all_zeros]
            # print(f"Executing Model B for batches: {torch.where(~batch_4_all_zeros)[0].tolist()}")
            outputs_b = self._forwardbs(x_b)
            final_output[~batch_4_all_zeros] = outputs_b
        # import pdb
        # pdb.set_trace()
        return final_output
        # model_a_indices = torch.where(batch_4_all_zeros | (batch_4_all_zeros & batch_5_all_zeros))[0]
        # model_b_indices = torch.where(batch_5_all_zeros & ~batch_4_all_zeros | ~batch_4_all_zeros & ~batch_5_all_zeros)[0]
        # # final_output = torch.empty_like(x)
        # if model_a_indices.numel() > 0:
        #     new_x_b = x[model_a_indices]
        #     outputs_a = self._forwardb(new_x_b)
        #     # final_output[model_a_indices] = outputs_a
        # if model_b_indices.numel() > 0:
        #     new_x_s = x[model_b_indices]
        #     outputs_b = self._forwardbs(new_x_s)
        #     # final_output[model_b_indices] = outputs_b

        # import pdb
        # pdb.set_trace()
        # return out
