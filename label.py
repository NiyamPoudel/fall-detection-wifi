import pandas as pd
import numpy as np
csv_input = pd.read_csv('/home/nimai/workspace/fall-detection-wifi/csv/exp67.csv')
df = pd.DataFrame(csv_input)
# print(df)
# last_two = ['00', '01', '02', '10', '11', '12','20', '21', '22','30', '31', '32','40', '41', '42','50', '51', '52']
last_two = ['00', '01', '02', '20', '21', '22', '40', '41', '42']
label_data = []
for i, row in df.iterrows():
    # print (row['timestamp'][-2:])
    if (str(row['timestamp'][-2:]) in last_two):
        label_data.append("fall")
    else:
        label_data.append("nofall")
# print(label_data)
df['label'] = label_data
print(df)
df.to_csv('/home/nimai/workspace/fall-detection-wifi/csv/exp67_l.csv')
    # ifor_val = something
    # if <condition>:
    #     ifor_val = something_else
    # df.at[i,'ifor'] = ifor_val

# dd = ((csv_input['timestamp']).astype(str))
# for val in dd:
#     print(val[-2:])

# csv_input['label'] = np.where(str(csv_input['timestamp'])[-2:] == '36', True, False)
# print(csv_input)
# csv_input['Berries'] = csv_input['Name']
# csv_input.to_csv('output.csv', index=False)