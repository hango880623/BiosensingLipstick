import pandas as pd
import ast

if __name__ == '__main__':
    # Read the data
    # df = pd.read_csv('./UMAP/PixelClass.csv')
    # # Display the first 5 rows of the data
    # df['pH'] = df['Filename'].str.split('_').str[2]
    # df['L'] = df['Lips_Color'].apply(lambda x: ast.literal_eval(x)[0])
    # df['A'] = df['Lips_Color'].apply(lambda x: ast.literal_eval(x)[1])
    # df['B'] = df['Lips_Color'].apply(lambda x: ast.literal_eval(x)[2])
    # df.drop(columns=['Monk_Skin_Tone'], inplace=True)
    # df.drop(columns=['Monk_Skin_Tone_Color'], inplace=True)
    # df.drop(columns=['Lips_Color'], inplace=True)
    # df.drop(columns=['Skin_Color'], inplace=True)
    # df.drop(columns=['Lips_Color_RGB'], inplace=True)
    # df.drop(columns=['Skin_Color_RGB'], inplace=True)
    # df = df.rename(columns={'Lips_Color_Copy': 'First_Lips_Color'})
    # print(df.head())
    # df.to_csv('./UMAP/pixel.csv', index=False)

    df = pd.read_csv('./UMAP/mixtest.csv')
    df['pH'] = ' ' + df['pH'].astype(str)
    df = df.sort_values(by='Filename')
    print(df.head())
    df.to_csv('./UMAP/mixtestString.csv', index=False)
