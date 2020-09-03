import argparse
import os

import numpy as np
import pandas as pd


def _main(args):
    submissions = []

    for i, submission in enumerate(args.submissions):
        submission = pd.read_csv(submission)

        if args.weights:
            submission["Label"] *= args.weights[i]

        submissions.append(submission)

    submissions = pd.concat(submissions)
    submissions = submissions.groupby("Id")

    if args.weights:
        submissions = submissions["Label"].sum()
    else:
        submissions = submissions["Label"].mean()

    submissions = submissions.reset_index()
    submissions[["Id", "Label"]].to_csv(args.output, index=False)


def serialize(text):
    text = text.replace("[", "")
    text = text.replace("]", "")
    text = [float(x) for x in text.split(" ") if x]
    return np.array(text)


def rearrange(df, size=2500):
    probabilities = np.array([x.tolist() for x in df.probabilities])
    labels = probabilities.argmax(1)
    df_n = df[labels == 0]
    df_p = df[labels != 0]

    df_p["Label"] = df_p.probabilities.apply(lambda x: sum(x[1:]))
    df_n["negative"] = df_n.probabilities.apply(lambda x: x[0])
    df_n = df_n.sort_values("negative", ascending=False)

    print(len(df_n))

    df_buffer = df_n[size:]
    df_n = df_n[:size]

    df_buffer["Label"] = df_buffer.probabilities.apply(lambda x: sum(x[1:]))
    df_n["Label"] = df_n.probabilities.apply(lambda x: 1 - x[0])
    df_p = df_p.sort_values("Label")

    eps = df_n.Label.max() - df_p.Label.min() + 1e-3 
    print(eps)
    df_p.Label += eps

    data = pd.concat(
        [
            df_n[["Id", "Label"]],
            df_p[["Id", "Label"]],
            df_buffer[["Id", "Label"]]
        ]
    )

    data = data.sort_index()

    return data


def main(args):
    submissions = []

    for i, submission in enumerate(args.submissions):
        submission = pd.read_csv(submission)
        submission.probabilities = submission.probabilities.apply(lambda x: serialize(x))
        if args.weights:
            submission.probabilities = submission.probabilities * args.weights[i] / sum(args.weights)
        submissions.append(submission)

    submissions = pd.concat(submissions)
    # submissions.probabilities = submissions.probabilities.apply(lambda x: serialize(x))
    submissions = submissions.groupby("Id")

    if args.weights:
        func = lambda x: np.sum(x)
    else:
        func = lambda x: np.mean(x)

    submissions = submissions.probabilities.apply(func).reset_index()

    if args.size:
        submissions = rearrange(submissions, size=args.size)
        submissions.to_csv(args.output, index=False)
    else:
        probabilities = np.array([x.tolist() for x in submissions.probabilities])
        labels = probabilities.argmax(1)

        aggregated = np.zeros((len(probabilities),))

        aggregated[labels != 0] = probabilities[labels != 0, 1:].sum(1)
        aggregated[labels == 0] = 1 - probabilities[labels == 0, 0]

        predictions = pd.DataFrame.from_dict({"Id": submissions.Id, "Label": aggregated})

        predictions[["Id", "Label"]].to_csv(args.output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trace code")
    parser.add_argument("--submissions", nargs="+", type=str, required=True)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument("--size", type=int, default=2500)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    main(args)
