import os
import pandas as pd
from sqlalchemy import create_engine
import time


# you need to first create a database called "ucl_final_year_project" pgAdmin
engine = create_engine('postgresql://{}:{}@localhost:5432/{}'.format('postgres', 'since2014',
                                                                     'ucl_final_year_project'))

"""
csv_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'files', 'senti_matrix.csv')
pd.read_csv(csv_path).to_sql('sentiment_scored', engine, index=False, if_exists='replace')
"""

folder_path = os.path.join(os.path.dirname(os.getcwd()), 'twitter_data', 'parts')
files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file[11:14] == 'all']
t1 = time.time()
df_list = [pd.read_csv(file) for file in files]
t2 = time.time()
print("Step 1: {:.2f} seconds (load all CSVs into a list of DataFrames)".format(t2-t1))
df = pd.concat(df_list, ignore_index=True)
t3 = time.time()
print("Step 2: {:.2f} seconds (merge all DataFrames into one large DataFrame)".format(t3-t2))
df.to_sql('tweets', engine, index=False, if_exists='replace')
print("Step 3: {:.2f} seconds (export the DataFrame to PostgreSQL)".format(time.time()-t3))
