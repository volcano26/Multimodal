import os
import pandas as pd
import chardet
from sklearn.model_selection import train_test_split

def read_and_process_data(txt_filepath, output_csv_filepath):
    df_txt = pd.read_csv(txt_filepath, header=None, names=['guid', 'tag'], dtype={'guid': str, 'tag': str})

    df_result = pd.DataFrame(columns=['guid', 'text', 'tag'])

    for index, row in df_txt.iterrows():
        guid = row['guid']
        txt_filename = f"{guid}.txt"
        txt_filepath = os.path.join('data', txt_filename)

        if os.path.exists(txt_filepath):

            with open(txt_filepath, 'rb') as txt_file:
                raw_data = txt_file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']

            with open(txt_filepath, 'r', encoding=encoding, errors='replace') as txt_file:
                text = txt_file.read().strip(",")

            df_result = df_result.append({'guid': guid, 'text': text, 'tag': row['tag']}, ignore_index=True)
        else:
            print(f"Warning: Text file not found for guid {guid}")

    df_result['img_path'] = df_result['guid'].apply(lambda guid: f'data/{guid}.jpg')

    df_result.to_csv(output_csv_filepath, index=False)

train_txt_filepath = 'train.txt'
train_csv_filepath = 'train.csv'
read_and_process_data(train_txt_filepath, train_csv_filepath)

train_df = pd.read_csv(train_csv_filepath)

train_set, val_set = train_test_split(train_df, test_size=0.2, random_state=42)

print(f'Training set shape: {train_set.shape}')
print(f'Validation set shape: {val_set.shape}')


train_train_csv_filepath = 'train_train.csv'
train_val_csv_filepath = 'train_val.csv'
train_set.to_csv(train_train_csv_filepath, index=False)
val_set.to_csv(train_val_csv_filepath, index=False)

test_txt_filepath = 'test_without_label.txt'
test_csv_filepath = 'test.csv'
read_and_process_data(test_txt_filepath, test_csv_filepath, is_test=True)
