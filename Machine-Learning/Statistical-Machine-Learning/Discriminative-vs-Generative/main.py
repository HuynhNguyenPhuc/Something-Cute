from model.discriminative import OneVsOne, OneVsTheRest, MultipleClass, Fisher
from model.generative import Bayesian

def test_models(data_dir):
    models = [
        OneVsOne(data_dir),
        OneVsTheRest(data_dir),
        MultipleClass(data_dir),
        Fisher(data_dir),
        Bayesian(data_dir)
    ]

    for model in models:
        print(f"Training {model.__class__.__name__}...")
        model.train()
        print(f"Evaluating {model.__class__.__name__}...")
        model.print_evaluation()
        print()

def main():
    data_dir = "data/iris.csv"
    test_models(data_dir)

if __name__ == "__main__":
    main()