import test
import util
import logging
import datasets
import commons

if __name__ == "__main__":
    args = parser.parse_arguments()
    commons.setup_logging(args.output_folder)
    # commons.make_deterministic(args.seed)
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {args.output_folder}")

    # Test set
    whole_test_set = datasets.WholeDataset(args.dataset_root, args.test_g, args.test_q)
    logging.info(f"Test set: {whole_test_set}")

    # Trained Model
    model = util.build_model(args)
    best_model_state_dict = torch.load(f"{args.output_folder}/best_model.pth")["state_dict"]
    model.load_state_dict(best_model_state_dict)

    # Inference
    recalls, recalls_str  = test.test(args, whole_test_set, model)
    logging.info(f"Recalls on {whole_test_set}: {recalls_str}")