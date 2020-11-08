
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="AdAGeo", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of triplets (query, pos, negs). Each triplet consists of 12 images")
    parser.add_argument("--cache_batch_size", type=int, default=24, 
                        help="Batch size for inference (caching and testing)")
    parser.add_argument("--cache_refresh_rate", type=int, default=1000,
                        help="How often to refresh cache, in number of queries")
    parser.add_argument("--num_clusters", type=int, default=64,
                        help="Number of clusters for NetVLAD layer")
    parser.add_argument("--margin", type=float, default=0.1,
                        help="margin for the triplet loss")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=100, 
                        help="number of epochs to train for")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.00001, 
                        help="Learning Rate.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to load checkpoint from, for resuming training or testing.")
    parser.add_argument("--pretrain", type=str, default="places", choices=["imagenet", "places"],
                        help="pretrained to use")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--attention", action="store_true", help="Use attention mechanism")
    parser.add_argument("--epoch_divider", type=int, default=5, 
                        help="Divide a train epoch in epoch_divider parts. Useful with huge datasets")
    parser.add_argument("--num_workers", type=int, default=6, help="num_workers for all dataloaders")
    parser.add_argument("--grl", action="store_true", help="Use Gradient Reversal Layer (GRL)")
    parser.add_argument("--grl_batch_size", type=int, default=8, help="Batch size for GRL")
    parser.add_argument("--grl_loss_weight", type=float, default=0.1, help="Weight for GRL loss")
    # PATHS
    parser.add_argument("--all_datasets_path", type=str, default="/home/valerio/datasets", 
                        help="Path containg all datasets")
    parser.add_argument("--root_path", type=str, default="oxford60k/image", help="Root of the dataset")
    parser.add_argument("--train_g", type=str, default="train/gallery", help="Path train gallery")
    parser.add_argument("--train_q", type=str, default="train/queries", help="Path train query")
    parser.add_argument("--val_g", type=str, default="val/gallery", help="Path val gallery")
    parser.add_argument("--val_q", type=str, default="val/queries", help="Path val query")
    parser.add_argument("--test_g", type=str, default="test/gallery", help="Path test gallery")
    parser.add_argument("--test_q", type=str, default="test/queries_5", help="Path test query")
    parser.add_argument("--grl_datasets", type=str,
                        default="train/queries+test/queries_5",
                        help="Paths for GRL datasets, linked by +")
    parser.add_argument("--output_path", type=str, default="runs", help="Folder with all outputs")
    parser.add_argument("--exp_name", type=str, default="default",
                        help="Folder name of the current run (saved in output_path)")
    return parser.parse_args()

