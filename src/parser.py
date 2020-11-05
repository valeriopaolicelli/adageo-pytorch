
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="AdAGeo", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of triplets (query, pos, negs). Each triplet consists of 12 images.")
    parser.add_argument("--cache_batch_size", type=int, default=24, help="Batch size for caching and testing")
    parser.add_argument("--cache_refresh_rate", type=int, default=1000,
                        help="How often to refresh cache, in number of queries. 0 for off")
    parser.add_argument("--num_clusters", type=int, default=64,
                        help="Number of clusters for NetVLAD layer.")
    parser.add_argument("--margin", type=float, default=0.1,
                        help="margin for the triplet loss.")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs to train for")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N",
                        help="manual epoch number (useful on restarts)")
    parser.add_argument("--lr", type=float, default=0.00001, help="Learning Rate.")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to load checkpoint from, for resuming training or testing.")
    parser.add_argument("--ckpt", type=str, default="latest",
                        help="Resume from latest or best checkpoint.", choices=["latest", "best"])
    parser.add_argument("--pretrain", type=str, default="places", help="basenetwork to use", 
                        choices=["imagenet", "places"])
    parser.add_argument("--train_part", action="store_false", help="Train partial network")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--attention", action="store_true", help="Use attention mechanism")
    parser.add_argument("--scenario", type=int, default=0, help="Scenario su cui testare.")
    parser.add_argument("--is_debug", action="store_true", help="Se vuoi fare solo una prova, utile per debuggare")
    parser.add_argument("--grl", action="store_true", help="Usa il GRL")
    parser.add_argument("--epoch_divider", type=int, default=5, help="Per dividere un epoca in N parti")
    parser.add_argument("--num_workers", type=int, default=4, help="Per dividere un epoca in N parti")
    parser.add_argument("--grl_batch_size", type=int, default=8, help="Batch size for GRL")
    parser.add_argument("--grl_loss", type=float, default=0.1, help="Moltiplicatore della loss GRL")
    
    parser.add_argument("--output_path", type=str, default="runs", help="Root del dataset")
    # PATHS dei DATASETS
    parser.add_argument("--all_datasets_path", type=str, default="/home/valeriop/datasets", 
                        help="Path con tutti i dataset")
    parser.add_argument("--root_path", type=str, default="oxford60k/image", help="Root del dataset")
    parser.add_argument("--train_g", type=str, default="train/gallery", help="Path train gallery")
    parser.add_argument("--train_q", type=str, default="train/queries", help="Path train query")
    parser.add_argument("--val_g", type=str, default="val/gallery", help="Path val gallery")
    parser.add_argument("--val_q", type=str, default="val/queries", help="Path val query")
    parser.add_argument("--test_g", type=str, default="test/gallery", help="Path test gallery")
    parser.add_argument("--test_q", type=str, default="test/queries", help="Path test query")
    parser.add_argument("--grl_datasets", type=str,
                        default="train/gallery+train/queries_all",
                        help="Paths per il grl, separati da +")
    parser.add_argument("--exp_name", type=str, default="default",
                        help="Folder name of the current run (saved in output_path).")
    return parser.parse_args()

