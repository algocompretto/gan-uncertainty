import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Pytorch implementation of WGAN model")
    parser.add_argument('--model', type=str, default='WGAN-GP')
    parser.add_argument('--epochs', type=int, default=50, help='The number of epochs to run')
    parser.add_argument('--channels', type=int, default=1, help='The number of channels')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--cuda', type=str, default='False', help='Availability of CUDA')
    parser.add_argument('--generator_iters', type=int, default=10000, help='The number of iterations for generator in '
                                                                        'WGAN-GP model.')
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    return check_args(parser.parse_args())


# Checking arguments
def check_args(args):
    try:
        assert args.epochs >= 1
    except AssertionError:
        print('[INFO] Number of epochs must be larger than or equal to one')

    try:
        assert args.batch_size >= 1
    except AssertionError:
        print('[INFO] Batch size must be larger than or equal to one')

    try:
        assert args.channels == 1
    except AssertionError:
        print('[INFO] Channels for dataset should be 1')
    args.cuda = True if args.cuda == 'True' else False
    if args.cuda:
        print('[INFO] CUDA enabled!')
    return args
