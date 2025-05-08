import argparse
import pandas as pd
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='Path to the metrics.csv file')
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.path):
        print(f"Error: Path {args.path} does not exist")
        return

    if args.dataset == 'md17':
        pd.set_option('display.max_rows', None)
        df = pd.read_csv(args.path)
        # test_e_mae = df.set_index('epoch')['test_e_MAE'].dropna()
        test_f_mae = df.set_index('epoch')['test_f_MAE'].dropna()
        # print(test_e_mae)
        print(test_f_mae)
    
    else:
        df = pd.read_csv(args.path)
        test_mae = df.set_index('epoch')['test_MAE'].dropna()
        print(test_mae)
        last_10_avg = test_mae.iloc[-10:].mean()
        print(f"average: {last_10_avg}")

if __name__ == "__main__":
    main()
