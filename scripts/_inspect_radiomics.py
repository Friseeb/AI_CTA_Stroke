import pandas as pd
df = pd.read_csv(r'C:\Users\spost\Desktop\CT_image\SLAAOBIDS\derivatives\radiomics_ts_laa.csv')
df = df.dropna(subset=['ratio_laa_la_median'])
print('Rows after NaN drop:', len(df))
print('ct_type counts:'); print(df['ct_type'].value_counts())
cols = ['ct_type','laa_original_firstorder_Median','la_original_firstorder_Median',
        'aorta_original_firstorder_Median','ratio_laa_la_median','ratio_laa_aorta_median','ratio_la_aorta_median']
print(df[cols].describe().to_string())
