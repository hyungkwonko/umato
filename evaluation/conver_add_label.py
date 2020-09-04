
import pandas as pd


if __name__ == "__main__":
    x = pd.read_csv('results/spheres/atsne.csv', float_precision='round_trip')
    df = pd.DataFrame(x)
    y = pd.read_csv('results/spheres/pca.csv', float_precision='round_trip')
    df2 = pd.DataFrame(y)
    df['label'] = df2['label']

    df.to_csv("atsne_spheres.csv", index=False)

    print(df.head(3))
