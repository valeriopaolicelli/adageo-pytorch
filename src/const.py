
runsPath = 'runs'

num_clusters = 64
margin = 0.1

num_samples = 10


def add_arguments(parser):
    parser.add_argument('--batchSize', type=int, default=4,
                        help='Number of triplets (query, pos, negs). Each triplet consists of 12 images.')
    parser.add_argument('--cacheBatchSize', type=int, default=24, help='Batch size for caching and testing')
    parser.add_argument('--cacheRefreshRate', type=int, default=1000,
                        help='How often to refresh cache, in number of queries. 0 for off')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning Rate.')
    parser.add_argument('--weightDecay', type=float, default=0.0, help='Weight Decay')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to load checkpoint from, for resuming training or testing.')
    parser.add_argument('--ckpt', type=str, default='latest',
                        help='Resume from latest or best checkpoint.', choices=['latest', 'best'])
    parser.add_argument('--pretrain', type=str, default='places', help='basenetwork to use', 
                        choices=['imagenet', 'places'])
    parser.add_argument('--wait', type=int, default=-1, help='PID of process to wait before start.')
    parser.add_argument('--arch', type=str, default='resnet18', help='basenetwork to use', 
                        choices=['resnet18', 'resnet101', 'resnet152', 'resnet50'])
    parser.add_argument('--train_part', action='store_false', help='Train partial network')
    parser.add_argument('--optim', type=str, default='adam',
                        help='optimizer to use', choices=['sgd', 'adam'])
    
    parser.add_argument('--attention', action='store_true', help='Use attention mechanism')
    parser.add_argument('--atten_type', type=str, default='cam',
                        help='Type of attention to apply', choices=['cam', 'channel_cam'])

    
    parser.add_argument('--save_htm', action='store_true', help='Save heatmap during evaluation')
    parser.add_argument('--load_pos', action='store_true', help='Load positives and predictions during evaluation')
    parser.add_argument('--save_pos', action='store_true', help='Save positives and predictions during evaluation')
    parser.add_argument('--save_preds', action='store_true', help='Save predictions')
    parser.add_argument('--scenario', type=int, default=0, help='Scenario su cui testare.')
    
    parser.add_argument('--isDebug', action="store_true", help='Se vuoi fare solo una prova, utile per debuggare')
    parser.add_argument('--grl', action="store_true", help='Usa il GRL')
    parser.add_argument('--atten_grl', action="store_true", help='Usa il GRL con attention features')
    
    parser.add_argument('--epochDivider', type=int, default=5, help='Per dividere un epoca in N parti')
    parser.add_argument('--num_workers', type=int, default=4, help='Per dividere un epoca in N parti')
    parser.add_argument('--grlBatchSize', type=int, default=8, help='Batch size for GRL')
    parser.add_argument('--grlLoss', type=float, default=0.1, help='Moltiplicatore della loss GRL')
    # PATHS dei DATASETS
    parser.add_argument('--allDatasetsPath', type=str, default='/home/valeriop/datasets', 
                        help='Path con tutti i dataset')
    parser.add_argument('--rootPath', type=str, default='oxford60k/image', help='Root del dataset')
    parser.add_argument('--trainG', type=str, default='train/gallery', help='Path train gallery')
    parser.add_argument('--trainQ', type=str, default='train/queries', help='Path train query')
    parser.add_argument('--valG', type=str, default='val/gallery', help='Path val gallery')
    parser.add_argument('--valQ', type=str, default='val/queries', help='Path val query')
    parser.add_argument('--testG', type=str, default='test/gallery', help='Path test gallery')
    parser.add_argument('--testQ', type=str, default='test/queries', help='Path test query')
    parser.add_argument('--grlDatasets', type=str,
                        default='train/gallery+train/queries_all',
                        help='Paths per il grl, separati da +')
    parser.add_argument('--expName', type=str, default='default',
                        help='Folder name of the current run (saved in runsPath).')

