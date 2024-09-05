import pickle
import argparse

from natural_niches_fn import run_natural_niches
from map_elites_fn import run_map_elites
from cma_es_fn import run_cma_es


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Description of your script")

    # Add arguments
    parser.add_argument('--pop_size', type=int, default=20)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--store_train_results', action='store_true', default=False)
    parser.add_argument('--use_crossover', action='store_true', default=True)
    parser.add_argument('--use_splitpoint', action='store_true', default=True)
    parser.add_argument('--use_matchmaker', action='store_true', default=True)
    parser.add_argument('--method', type=str, default="natural_niches",
                        choices=["natural_niches", "ga", "map_elites", "cma_es"])
    parser.add_argument('--total_forward_passes', type=int, default=50000)

    return parser.parse_args()


def main():
    args = parse_arguments()
    args_dict = vars(args)
    method = args_dict.pop('method')
    file_name = method

    if method == "natural_niches":
        if args.alpha != 1.0:
            file_name += f"_alpha_{args.alpha}"
        if not args.use_crossover:
            file_name += "_no_crossover"
        if not args.use_splitpoint and args.use_crossover:
            file_name += "_no_splitpoint"
        if not args.use_matchmaker and args.use_crossover:
            file_name += "_no_matchmaker"
        results = run_natural_niches(**args_dict)
    elif method == "ga":
        assert not args.use_matchmaker, "GA does not use matchmaker"
        results = run_natural_niches(**args_dict, alpha=0.0)
    elif method == "map_elites":
        results = run_map_elites(args.runs, args.total_forward_passes, args.store_train_results)
    elif method == "cma_es":
        results = run_cma_es(args.runs, args.pop_size, args.total_forward_passes, args.store_train_results)
    else:
        raise NotImplementedError("Method not implemented")
    
    with open(f"results/{file_name}.pkl", "wb") as f:
            pickle.dump(results, f) 


if __name__ == "__main__":
    main()