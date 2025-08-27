import joblib
import argparse
import pandas as pd


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', type=str, required=True)
    return parser.parse_args()


def main():
    args = arg_parser()
    result = joblib.load(args.result_file)

    num_of_cumulateds = sorted(result.keys())
    num_of_cross_versions = sorted(result[1].keys())

    # initialize dataframe
    d = {}
    for num_of_cross_version in num_of_cross_versions:
        if num_of_cross_version == 0:
            continue
        d[num_of_cross_version] = [None for _ in num_of_cumulateds]
    df = pd.DataFrame(data=d, index=num_of_cumulateds)

    # fill dataframe
    for num_of_cumulated in result.keys():
        for num_of_cross_version in sorted(result[num_of_cumulated].keys()):
            if num_of_cross_version == 0:
                continue

            df.loc[num_of_cumulated, num_of_cross_version] = \
                format((result[num_of_cumulated][num_of_cross_version] - result[num_of_cumulated][0]) * 100, '.2f')

    print(df)

    print('Average')
    for colname, data in df.items():
        valid_data = [float(i) for i in data if i is not None]
        print(colname, format(sum(valid_data) / len(valid_data), '.2f'))


if __name__ == '__main__':
    main()
