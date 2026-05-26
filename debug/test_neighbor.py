import pandas as pd
import numpy as np

df = pd.read_pickle('e:/Msc/new/code/simpl_new/SIMPL/data_av2/features/train/0925ba29-4dd0-43c7-9b5c-ae067c7fb7c0.pkl')
row = df.iloc[0]
trajs_pos = row['TRAJS']['trajs_pos']
trajs_cat = row['TRAJS']['trajs_cat']

print('Num agents:', len(trajs_cat))
for i in range(5):
    print(f'Agent {i} ({trajs_cat[i]}):')
    print('   t=0:', trajs_pos[i, 0, :])
    print('   t=49:', trajs_pos[i, 49, :])
