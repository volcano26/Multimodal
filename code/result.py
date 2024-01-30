import pandas as pd

csv_filepath = 'results/result.csv'
df = pd.read_csv(csv_filepath)

# 选择两列数据（示例中选择第一列和第二列）
selected_columns = df[['guid', 'tag']]

# 将选定的数据保存为文本文件
txt_filepath = 'results/result.txt'
selected_columns.to_csv(txt_filepath, header=None, index=None, sep=',', mode='a')

# 读取生成的文本文件
with open(txt_filepath, 'r') as txt_file:
    content = txt_file.read()
    print(content)
