import pandas as pd
import click


@click.command()
@click.option("--input", default='train.csv',
              help="Original training set or corrected labels for training")
@click.option("--output", default='less_noisy_train.csv',
              help="Training set with less noisy labels")
@click.option("--corrected_labels",
              default='Airspace_Opacity.csv,Cardiomegaly.csv,Fracture.csv,Lung_Lesion.csv,'
                      'Pleural_Effusion.csv,Pneumothorax.csv',
              help="Training set with less noisy labels")
@click.option("--pre-process", default=False,
              help="Do pre-process steps like fill not available values, convert values to one-hot")
def correct_noisy_labels(input, output, corrected_labels, pre_process):
    train_df = pd.read_csv(input)

    if pre_process:
        train_df = train_df.fillna(0)
        train_df[list(train_df)[1:]] = 1 * (train_df[list(train_df)[1:]] >= 0.5)

    corrected_labels = corrected_labels.split(',')
    for file in corrected_labels:
        noise_labels = dict()
        with open(file, 'r') as f:
            cnt = 0
            for line in f.readlines():
                cnt += 1
                if cnt == 1:
                    continue

                line = line.strip()
                items = line.split(',')
                noise_labels[items[0]] = int(items[1])

        for i, row in train_df.iterrows():
            if row['Images'] in noise_labels:
                train_df.at[i, file.split('.')[0]] = 1 - noise_labels[row['Images']]

    train_df.to_csv(output, index=False)


if __name__ == "__main__":
    correct_noisy_labels()
