"""Preprocessing PTB-XL (100 Hz) -> numpy arrays + label superclass (multi-label)."""
import ast, glob, os
import numpy as np, pandas as pd, wfdb

ROOT = glob.glob('data/raw/*')[0] + '/'
df = pd.read_csv(ROOT + 'ptbxl_database.csv', index_col='ecg_id')
df.scp_codes = df.scp_codes.apply(ast.literal_eval)
scp = pd.read_csv(ROOT + 'scp_statements.csv', index_col=0)
scp = scp[scp.diagnostic == 1]

CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

def superclass(codes):
    return list({scp.loc[c, 'diagnostic_class'] for c in codes if c in scp.index})

df['super'] = df.scp_codes.apply(superclass)
df = df[df.super.map(len) > 0]  # buang rekaman tanpa label diagnostik
Y = np.array([[c in s for c in CLASSES] for s in df.super], dtype=np.float32)

X = np.stack([wfdb.rdsamp(ROOT + f)[0] for f in df.filename_lr]).astype(np.float32)  # (N, 1000, 12)
np.savez_compressed('data/ptbxl100.npz', X=X, Y=Y, fold=df.strat_fold.values,
                    age=df.age.values, sex=df.sex.values, ecg_id=df.index.values)
print(X.shape, Y.sum(0), dict(zip(CLASSES, Y.mean(0).round(3))))
