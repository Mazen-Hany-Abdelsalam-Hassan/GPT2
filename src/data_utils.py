import pandas as pd
from config import PARENT_DIR ,DATA_PATH , SEED ,test_df_dir , val_df_dir ,train_df_dir
def split_data(data:pd.DataFrame, train_split:int = .8 , val_split:int= .10  ):
    
    os.makedirs(PARENT_DIR,exist_ok = True)
    os.makedirs(DATA_PATH, exist_ok = True )
    data = data.sample(frac = 1,random_state = SEED). reset_index(drop=True)
    data_size  = len(data)
    train_size = int(train_split * data_size)
    val_size   = int((train_split+val_split) * data_size)
    train_df   = data[:train_size]
    val_df     = data[train_size : val_size]
    test_df     = data[val_size:]
    train_df.iloc[::,1] = train_df.iloc[::,1].map(LABEL_DICTIONARY)
    val_df.iloc[::, 1]  = val_df.iloc[:: , 1].map(LABEL_DICTIONARY)
    test_df.iloc[::,1]  = test_df.iloc[:: , 1].map(LABEL_DICTIONARY)
    train_df.to_csv(train_df_dir, index = False)
    val_df.to_csv(val_df_dir , index = False)
    test_df.to_csv(test_df_dir, index = False)
