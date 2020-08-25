class Sig(object):
    def __init__(self, df, null_col, test_col):
        # split test column based on null_col: one split will have all rows with null null_col and vice versa
        # check if two resulting series are coming from the same distribution

        df_test = df[[null_col, test_col]]
        
        ch_nonull = df_test[df_test[null_col].notnull()]
        ch_null = df_test[df_test[null_col].isnull()]
        self.sample1 = ch_nonull[test_col]
        self.sample2 = ch_null[test_col]
    
    def testing(self, test):
        return test(self.sample1, self.sample2)
    
    def statistics(self):
        import numpy as np
        
        return {'sample-1': (np.mean(self.sample1),np.std(self.sample1)), 
                'sample-2': (np.mean(self.sample2),np.std(self.sample2))}