from pathlib import Path

import pandas as pd

folders = {"Cover": 0, "JMiPOD": 1, "JUNIWARD": 2, "UERD": 3}


def main():
    dataset = []

    for folder in folders:
        folder = Path(folder)

        for item in folder.iterdir():
            dataset.append(
                {"Image": str(item), "Label": folders[str(folder)]}
            )

    dataset = pd.DataFrame(dataset)
    dataset.to_csv("train.csv", index=None)


if __name__ == "__main__":
    main()
