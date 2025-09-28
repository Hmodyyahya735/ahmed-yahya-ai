import argparse
from ahmed_yahya_ai.algorithms import linear_regression, logistic_regression_predict

def main():
    parser = argparse.ArgumentParser(description="Ahmed Yahya AI Algorithms CLI")
    parser.add_argument('--linear', nargs=2, metavar=('X', 'y'), help="Run linear regression. Example: --linear \"1,2,3\" \"2,4,6\"")
    parser.add_argument('--logistic', nargs=3, metavar=('X', 'w', 'b'), help="Run logistic regression. Example: --logistic \"1,2,3\" 1.0 0.0")
    args = parser.parse_args()

    if args.linear:
        X = [float(i) for i in args.linear[0].split(',')]
        y = [float(i) for i in args.linear[1].split(',')]
        result = linear_regression(X, y)
        print("Linear Regression Result:", result)

    if args.logistic:
        X = [float(i) for i in args.logistic[0].split(',')]
        w = float(args.logistic[1])
        b = float(args.logistic[2])
        result = logistic_regression_predict(X, w, b)
        print("Logistic Regression Prediction:", result)

if __name__ == "__main__":
    main()
