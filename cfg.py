import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-baseline', type=str, default='unet', help='baseline net type')
    parser.add_argument('-tsth', type=int, default=500, help='threshold of ts')
    parser.add_argument('-pickwin', type=int, default=256, help='picked window')
    parser.add_argument('-mode', type=str, required=True, help='mode')
    parser.add_argument('-vis', type=bool, default=False, help='visualization')
    parser.add_argument('-val_freq',type=int,default=5,help='interval between each validation')
    parser.add_argument('-cuda', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-gpu_device', type=int, default=0, help='use which gpu')
    parser.add_argument('-epoch_ini', type=int, default=1, help='start epoch')
    parser.add_argument('-hidch', type=int, default=16, help='dim_size')
    parser.add_argument('-layers', type=int, default=8, help='depth')
    parser.add_argument('-heads', type=int, default=1, help='heads number')
    parser.add_argument('-mlp_dim', type=int, default=1024, help='mlp_dim')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-imp_ep', type=int, default=256, help='implicit learning rate')
    parser.add_argument('-weights', type=str, default = 0, help='the weights file you want to test')
    parser.add_argument('-distributed', default='0,1,2,3' ,type=str,help='multi GPU ids to use')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type =int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=256,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument(
    '-data_path',
    type=str,
    default='../dataset',
    help='The path of store data')
    parser.add_argument(
    '-out_path',
    type=str,
    default='../dataset',
    help='The path of store data')
    opt = parser.parse_args()

    return opt
