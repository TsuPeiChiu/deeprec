import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

file_Mg_true = "../atf4/data/methylated_10mer_with_Mg.txt"
file_Mg_pred = "../atf4/output/test_predictions_meth.tsv"
file_CG_true = "../atf4/data/unmethylated_10mer_with_CG.txt"
file_CG_pred = "../atf4/output/test_predictions_unmeth.tsv"

meth_true = pd.read_csv(file_Mg_true, names=['seq_meth', 'meth_true'], sep='\t')
meth_pred = pd.read_csv(file_Mg_pred, names=['seq_meth_pred', 'meth_pred'], sep='\t')
unmeth_true = pd.read_csv(file_CG_true, names=['seq_unmeth', 'unmeth_true'], sep='\t')
unmeth_pred = pd.read_csv(file_CG_pred, names=['seq_unmeth_pred', 'unmeth_pred'], sep='\t')

df = pd.concat([meth_true, meth_pred, unmeth_true, unmeth_pred], axis=1)
df['left_3_4'] = df.seq_meth.str[1:3]
df['right_3_4'] = df.seq_meth.str[7:9]
df['center'] = df.seq_meth.str[4:6] 


conditions = [
        (df['left_3_4']=='Mg'),
        (df['right_3_4']=='Mg'),
        (df['center']=='Mg'),
        ((df['left_3_4']!='Mg') &  (df['right_3_4']!='Mg') & (df['center']!='Mg'))        
        ]
values = ['flank', 'flank', 'center', 'others']


df['group'] = np.select(conditions, values)



sns.jointplot(x="unmeth_pred", y="meth_pred", data=df,
                  kind="reg", truncate=False,
                  xlim=(0, 0.25), ylim=(0, 0.25),
                  hue='group')







"""
print(data)

print(data['meth_pred'].corr(data['meth_true']))
print(data['unmeth_pred'].corr(data['unmeth_true']))
print(data['meth_pred'].corr(data['unmeth_true']))
print(data['unmeth_pred'].corr(data['meth_true']))

print(data['meth_true'].corr(data['unmeth_true']))
print(data['meth_pred'].corr(data['unmeth_pred']))


plt.scatter(x=data['meth_pred'], y=data['meth_true'])

plt.scatter(x=data['unmeth_pred'], y=data['meth_pred '])
"""







#print(data['unmeth_true'].corr(data['meth_pred']))







#pt.scatter(x=a['meth_true'], y=a['meth_pred'])
#pt.scatter(x=a['meth_true'], y=a['unmeth_true'])
#pt.scatter(x=a['unmeth_true'], y=a['meth_pred'])


#pt.xlabel("unmeth_true")
#pt.ylabel("meth_pred")
#pt.xlim(0, 0.5)
#pt.ylim(0, 0.5)


#pt.show()


