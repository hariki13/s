# import pandas as pd


# # DataFrame creation
# data = {
#     'Name': ['Alice', 'Bob', 'Charlie', 'David'],
#     'Age': [25, 30, 35, 40],
#     'City': ['New York', 'Los Angeles', 'Chicago', 'Houston'],
#     'Salary': [70000, 80000, 90000, 100000],
# }
# df = pd.DataFrame(data, index=['manager1', 'supervisor', 'leader', 'staff'])
# print(df.iloc[1:3, 0:2])  # Slicing rows and columns using iloc
# print(df.loc['supervisor':'leader', 'Name':'Age'])  # Slicing rows and columns using loc

import os 
import kagglehub


#list datasets by keyword
os.system("kaggle datasets list -s coffee")
kaggle datasets download - d halaturkialotaibi/coffee-bean-sales-dataset




