import os
import tiktoken

ALLOWED_SEQ_LENGTH = 1000
SEED = 123
LABEL_DICTIONARY = {"positive": 1 , "negative":0}
PARENT_DIR= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_df_dir = os.path.join(PARENT_DIR ,"data", "train.csv")
val_df_dir = os.path.join(PARENT_DIR ,"data", "val.csv")
test_df_dir = os.path.join(PARENT_DIR ,"data", "test.csv")
DATA_PATH = os.path.join(PARENT_DIR ,'data')
TOKENIZER = tiktoken.encoding_for_model('gpt-2')
PADDINGTEXT = "<|endoftext|>"
PADDINGTOKEN = TOKENIZER.encode(PADDINGTEXT,allowed_special = 'all')[0]
MODEL_DIR = "gpt2-small-124M.pth"   #for downloading model
BASE_MODELS_DIR = os.path.join(PARENT_DIR, 'BASEModel')
CLASSIFICATION_MODEL_DIR = os.path.join(PARENT_DIR, 'CLASSIFICATION')





