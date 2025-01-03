def set_template(args):
    if args.template == 'SPEINet':
        args.task = "VideoDeblur"
        args.model = "SPEINet"
        args.n_sequence = 3
        args.patch_size = 200
        args.n_frames_per_video = 200
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 4
        args.loss = '1*L1+2*HEM'
        args.lr = 1e-4
        args.lr_decay = 150 #150
        args.window_size = 5
        args.depths = [6, 6, 6, 6, 6, 6]
        args.embed_dim = 256
        args.num_heads = [8, 8, 8, 8, 8, 8]
        args.mlp_ratio = 2
        args.resi_connection = "1conv"
        args.data_train = 'DVD_NFS'
        args.data_test = 'DVD_NFS'
        args.batch_size = 20
        args.pre_train = '../experiment/SPEINet/model/model_best.pt'
    elif args.template == 'SPEINet_REDS':
        args.task = "VideoDeblur"
        args.model = "SPEINet"
        args.n_sequence = 3
        args.patch_size = 200
        args.n_frames_per_video = 200
        args.n_feat = 32
        args.n_resblock = 3
        args.size_must_mode = 4
        args.loss = '1*L1+2*HEM'
        args.lr = 5e-5
        args.lr_decay = 200
        args.window_size = 5
        args.depths = [6, 6, 6, 6, 6, 6]
        args.embed_dim = 256
        args.num_heads = [8, 8, 8, 8, 8, 8]
        args.mlp_ratio = 2
        args.resi_connection = "1conv"
        args.data_train = 'DVD_NFS'
        args.data_test = 'DVD_NFS'
        args.batch_size = 20
        args.pre_train = '../experiment/swint_hsa_nsf/model/model_best.pt'
        args.dir_data = '../dataset/REDS/train'
        args.dir_data_test = '../dataset/REDS/val'
    else:
        raise NotImplementedError('Template [{:s}] is not found'.format(args.template))
