import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    """
    parser = argparse.ArgumentParser(description="Run.")

    parser.add_argument('--model_name', type=str, default='HSFA', help='Model name')
    parser.add_argument('--data_name', type=str, default='NYC', help='Dataset name')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed')
    parser.add_argument('--batch', type=int, default=128, help='Batch size')
    parser.add_argument('--epoch', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--save', action='store_true', help='Save model and results')
    parser.add_argument('--log', action='store_true', help='Log information')
    parser.add_argument('--save_args', action='store_true', help='Save arguments')
    parser.add_argument('--cuda', default=False, help='Use CUDA')
    parser.add_argument('--gpu_id', type=int, default=1, help='GPU ID to use')

    parser.add_argument('--num_region', type=int, default=40, help='Number of regions')
    parser.add_argument('--min_len', type=int, default=3, help='Minimum trajectory length')
    parser.add_argument('--max_len', type=int, default=101, help='Maximum trajectory length')
    parser.add_argument('--scale', type=float, default=0.1, help='Scale for similarity score')
    parser.add_argument(
        '--pre_hsfa_num_bands',
        type=int,
        default=None,
        help='Override pre_hsfa_num_bands in config (K).'
    )
    parser.add_argument(
        '--pre_hsfa_low_band_count',
        type=int,
        default=None,
        help='Override low-frequency band count c in config. Valid range: 1..K-1.'
    )
    
    args = parser.parse_args()

    return args
