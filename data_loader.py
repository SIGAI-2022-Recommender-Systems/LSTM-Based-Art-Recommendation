import numpy as np
import pandas as pd
import json

f = open("Behance_appreciate_1M","rb")
raw_data = [i.decode('utf8').rstrip("\n") for i in f.readlines()]
f.close()
data = np.array([list(map(int, i.split())) for i in raw_data])

df = pd.DataFrame(data)
df.columns = ["userid","itemid","timestamp"]

organized_data = df.groupby('userid')['itemid'].apply(np.array)

organized_data_df = pd.DataFrame(organized_data)
keyli = organized_data.keys().tolist()
keyli = [int(i) for i in keyli]
organized_data = np.array(organized_data)

timestamps = np.array(df.groupby('userid')['timestamp'].apply(np.array))

new_array = []

for i in range(len(timestamps)):
    # the int casts you see are just to comply with json regulations, apparently tolist() isnt enough
    idx = np.argsort(timestamps[i])
    x = (organized_data[i][idx]).tolist()
    x = [int(i) for i in x]
    y = timestamps.tolist()[i]
    y = [int(i) for i in y]
    ele = list(zip(x,y))
    ele = [list(i) for i in ele]
    new_array.append(ele)

final_json = dict(zip(keyli,new_array))

with open("data.json","w") as outfile:
    json.dump(final_json,outfile)