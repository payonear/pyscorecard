"""
@author: payonear
"""
# import libraries file
from functions import *

# import Data Set (put the right directory)
df = pd.read_csv('PATH.../UCI_Credit_Card.csv')
# rename column with default
df = df.rename({'default.payment.next.month': 'DEF'}, axis = 1)
# rename columns with PAY_
df = df.rename({'PAY_2': 'PAY_1', 'PAY_3': 'PAY_2', 'PAY_4': 'PAY_3', 'PAY_5':'PAY_4', 'PAY_6':'PAY_5'}, axis = 1)

# initial data review
summary = df.describe()
# check data types
df.dtypes

# change type od LIMIT_BAL
df['LIMIT_BAL'] = df['LIMIT_BAL'].astype('int64')

# example of kurtosis and skewness calculation for variables you need
print(skew(df['LIMIT_BAL']))
print(kurtosis(df['LIMIT_BAL'])) 

# check for nulls
df.isnull().sum(axis=0)
# there are no null values in this dataset, which is very rare case

# Variable's value modification
df['SEXF'] =df['SEX'].replace({1:'male', 2:'female'}) 


# 0, 4,5,6 modified and combined to one level - 0

df['EDUCATION'] = df['EDUCATION'].replace([0,4,5,6],0)
df['EDUCATIONF'] = df['EDUCATION'].replace({0:'others', 1:'graduate school', 2:'university', 3:'high school'})

# marriage variable modification 
df['MARRIAGEF'] = df['MARRIAGE'].replace({0:'others', 1:'married', 2:'single', 3:'divorce'})

# PAY_ variables modification
for i in range(0,6):
    etykieta = 'PAY_' + str(i)
    etykietaf = 'PAY_' + str(i) + 'F'
    df[etykietaf] = df[etykieta].replace({-2:'no consumption', -1:'fully paid', 0:'revolving',\
                                          1:'1d', 2:'2d', 3:'3d', 4:'4d', 5:'5d', 6:'6d', 7:'7d', 8:'8d', 9:'9d'})

# Initial data analyses (Education example)
table = pd.crosstab(df['EDUCATIONF'], df['DEF'], margins = True)
table1 = pd.crosstab(df['EDUCATIONF'], df['DEF'], normalize = 'index')
table1['Total'] = table1[0] + table1[1]
pd.crosstab(df['EDUCATIONF'], df['DEF'], normalize = 'index').round(2)

# correlation table
corr = df.corr()


# independence Chi-square test Education example (critical value = 0.352 in this case)
print(chi2_contingency(table)[0])

 # 3-way crosstabs
table = pd.crosstab(index = df['EDUCATIONF'], columns = df['SEXF'], values = df['DEF'],\
                    aggfunc = lambda x: round(np.sum(x)/len(x),2))

# Default rate plot
fig =plt.figure(figsize=(6, 6)) 
sns.set_color_codes("pastel")
sns.barplot(x=table1.iloc[:,:-1].sort_values(by = 1).index, y= 'Total', data=table1.sort_values(by = 1),
            label="Total", color="b", orient = 'v')
sns.set_color_codes("muted")
sns.barplot(x=table1.iloc[:,:-1].sort_values(by = 1).index, y= 1, data=table1.sort_values(by = 1),
label="default", color="b", orient = 'v')
plt.show()


# basic dataset creation
base = df[["SEXF","EDUCATIONF","MARRIAGEF","LIMIT_BAL","PAY_0F","PAY_2F","PAY_3F",\
           "PAY_4F","PAY_5F","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5",\
           "BILL_AMT6","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]]

base.to_csv('base.csv', index = None, header = True)
df.to_csv('df.csv', index = None, header = True)