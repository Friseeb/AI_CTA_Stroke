import pandas as pd
df = pd.read_csv(r'C:\Users\spost\Desktop\CT_image\SLAAOBIDS\derivatives\radiomics\radiomics_ibsi_all.csv')
print('ct_type unique:', df['ct_type'].unique())
print('shape:', df.shape)
cols = [
    'laa_roi_voxels','la_roi_voxels','aorta_roi_voxels',
    'laa_diagnostics_Mask-interpolated_VoxelNum',
    'la_diagnostics_Mask-interpolated_VoxelNum',
    'aorta_diagnostics_Mask-interpolated_VoxelNum',
    'laa_diagnostics_Mask-original_VoxelNum',
    'la_diagnostics_Mask-original_VoxelNum',
    'aorta_diagnostics_Mask-original_VoxelNum',
    'laa_original_firstorder_Median',
    'la_original_firstorder_Median',
    'aorta_original_firstorder_Median',
    'ratio_la_aorta_median','ratio_laa_la_median','ratio_laa_aorta_median',
]
for col in cols:
    if col in df.columns:
        print(f'{col}: non-null={df[col].notna().sum()}, sample={df[col].dropna().head(2).tolist()}')
    else:
        print(f'{col}: NOT FOUND')
