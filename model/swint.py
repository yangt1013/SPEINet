import torch
import torch.nn as nn
from model import recons_video_ori
from model import swinir
from util import utils


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
        print('Is meanfilter image when process mask:', 'True' if is_mask_filter else 'False')
        extra_channels = 0
        print('Select mask mode: concat, num_mask={}'.format(extra_channels))
        self.swin = swinir.SwinIR(upscale=1,
                   in_chans=n_feat * 4,
                   img_size=args.patch_size//4,
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
        self.conv = nn.Conv2d(n_feat*4*n_sequence, n_feat*4, kernel_size=1, stride=1, padding=0)
        if load_recons_net:
            self.recons_net.load_state_dict(torch.load(recons_pretrain_fn))
            print('Loading reconstruction pretrain model from {}'.format(recons_pretrain_fn))

    def forward(self, x):  
        frame_list = [x[:, i, :, :, :] for i in range(self.n_sequence)]
        f_mid = self.recons_net.encoder_second(self.recons_net.encoder_first(self.recons_net.inBlock(frame_list[self.n_sequence//2])))
        f_fusion = f_mid
        for i in range(self.n_sequence):
            if i == self.n_sequence//2:
                continue
            feature = self.recons_net.encoder_second(self.recons_net.encoder_first(self.recons_net.inBlock(frame_list[i])))
            f_trans = self.swin(f_mid, feature)  
            f_fusion = torch.cat((f_fusion, f_trans), dim=1)
        if self.n_sequence == 1:
            f_trans = self.swin(f_mid, f_mid) 
            f_fusion = f_fusion + f_trans
        f_fusion = self.conv(f_fusion)
        output = self.recons_net.outBlock(self.recons_net.decoder_first(self.recons_net.decoder_second(f_fusion)))

        return output
