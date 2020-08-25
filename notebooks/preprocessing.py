import pandas as pd 


class Preprocessing(object):
    def __init__(self, train_datefile, test_datafile):
        self.train_df, self.test_df = pd.read_csv('../data/train.csv'), pd.read_csv('../data/test.csv')
        self.train_df_dis = self.train_df[(self.train_df['target']==1)]
        self.train_df_nodis = self.train_df[(self.train_df['target']==0)]
        
    def null_treatment(self):
        train_df_keyword=self.train_df['keyword'].fillna('uns_keyword')
        self.train_df['modified_keyword'] = train_df_keyword
        train_df_loc=self.train_df['location'].fillna('uns_location')
        self.train_df['modified_location'] = train_df_loc
        self.train_df = self.train_df.drop(['location', 'keyword'], axis=1)
        return self.train_df 