import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="calopic-mltrainer",
        description="Calopic ML Trainer - minimal CLI for classical ML models",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit", help="train and evaluate a model")
    fit.add_argument("--model", required=True, choices=["logreg", "rf"])
    fit.add_argument("--data", required=True)
    fit.add_argument("--target", required=True)
    fit.add_argument("--test-size", type=float, default=0.2)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--save", default=None)

    args = parser.parse_args()

    if args.command == "fit":
        from .data import load_tabular
        from .train import train_and_evaluate
        import joblib

        X_train, X_test, y_train, y_test, _ = load_tabular(
            args.data, args.target, test_size=args.test_size, seed=args.seed
        )
        model, metrics = train_and_evaluate(
            args.model, X_train, y_train, X_test, y_test
        )
        print(f"metrics: {metrics}")

        if args.save:
            joblib.dump(model, args.save)
            print(f"saved model to {args.save}")


if __name__ == "__main__":
    main()