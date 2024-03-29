
"""
@author: payonear
"""
from functions import *

# import Data Sets (put the right directory)
train = pd.read_csv('PATH.../train.csv')
test = pd.read_csv('PATH.../test.csv')
coarse_r = pd.read_csv('PATH.../coarse_r.csv')

# separate numerics and factors in training sample for coarse classing
columns = list(train.columns.drop(['def']))
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
columns_n = list(train[columns].select_dtypes(include=numerics).columns)
columns_f = list(train[columns].select_dtypes(exclude=numerics).columns)

# create a dictionary with cut-points generated by smbinning
coarse_r['cuts'] = coarse_r['cuts'].map(lambda x: str(x)[2:-1])
coarse = coarse_r.set_index('VAR')['cuts'].str.split(', ').to_dict()
for i in list(coarse.keys()):
    coarse[i] = list(map(int, coarse[i]))

# cut-points were calculated and fitted based on training sample
# we use the same cut-points for testing sample
coarse_classing(train, coarse, columns_n)
coarse_classing(test, coarse, columns_n)
# calculation of WOE and IV now for coarse classes
bins_coarse = woe_iv(train, list(train.filter(like='_coarse').columns), y = 'def')
bins_coarse_test = woe_iv(test, [s + '_coarse' for s in columns_n], y = 'def')

# create pdf and save analitical plots for each coarse classed numerical variable
pdf = PdfPages('WOE_coarse.pdf')

for i in columns_n:
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    
    d = sns.barplot(data = train, x = str(i)+'_coarse', y = str(i), estimator = lambda y: round(100*len(y)/train.shape[0],2),
                      orient = 'v', palette = sns.cubehelix_palette(8),ax = ax[0][0])
    d.set_xticklabels(d.get_xticklabels(),rotation=45, horizontalalignment='right')
    for p in d.patches:
        d.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
    ax[0][0].set_title('Distribution \n ' + str(i) + ' train')
    ###############################################################################
    
    a = sns.barplot(data = test, x = str(i)+'_coarse', y = str(i), estimator = lambda y: round(100*len(y)/test.shape[0],2),
                      orient = 'v', palette = sns.cubehelix_palette(8),ax = ax[0][1])
    a.set_xticklabels(a.get_xticklabels(),rotation=45, horizontalalignment='right')
    for p in a.patches:
        a.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
    ax[0][1].set_title('Distribution \n ' + str(i) + ' test')
    
    
    ###############################################################################
    
    b = sns.barplot(data = train, x = str(i)+'_coarse', y = 'def', estimator = lambda y: round(100*y.sum()/len(y),2),
                    orient = 'v', palette = sns.cubehelix_palette(8),errwidth   = 0.0,ax = ax[1][0])
    b.set_xticklabels(b.get_xticklabels(),rotation=45, horizontalalignment='right')
    for p in b.patches:
        b.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
    ax[1][0].set_title('Bad rate \n ' + str(i))
    
    ###############################################################################
    
    c = sns.barplot(data = bins_coarse[str(i)+'_coarse'][bins_coarse[str(i)+'_coarse']['WOE'].notnull()],
                                x = 'Bin', y = 'WOE', palette = sns.cubehelix_palette(8),ax = ax[1][1])
    c.set_xticklabels(c.get_xticklabels(),rotation=45, horizontalalignment='right')
    for p in c.patches:
        c.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
    ax[1][1].set_title('WOE \n ' + str(i))
    
    plt.subplots_adjust(hspace=0.8)
    pdf.savefig()
    plt.close()
    
pdf.close()

# coarse classing factorial variables (grouping by default rate and consolidating non-numerous fine classed groups with huge ones)
train['EDUCATIONF_coarse'] = train['EDUCATIONF'].replace({'university': 'univ/hs', 'high school': 'univ/hs'})
test['EDUCATIONF_coarse'] = test['EDUCATIONF'].replace({'university': 'univ/hs', 'high school': 'univ/hs'})     

train['PAY_0F_coarse'] = train['PAY_0F'].replace({'2d': '2d+', '3d': '2d+', '4d': '2d+', '5d': '2d+', '6d': '2d+', '7d': '2d+'})
test['PAY_0F_coarse'] = test['PAY_0F'].replace({'2d': '2d+', '3d': '2d+', '4d': '2d+', '5d': '2d+', '6d': '2d+', '7d': '2d+'})

train['PAY_2F_coarse'] = train['PAY_2F'].replace({ '1d': '1d+', '2d': '1d+', '3d': '1d+', '4d': '1d+', '5d': '1d+', '6d': '1d+', '7d': '1d+'})
test['PAY_2F_coarse'] = test['PAY_2F'].replace({ '1d': '1d+', '2d': '1d+', '3d': '1d+', '4d': '1d+', '5d': '1d+', '6d': '1d+', '7d': '1d+'})

train['PAY_5F_coarse'] = train['PAY_5F'].replace({'2d': '2d+', '3d': '2d+', '4d': '2d+', '5d': '2d+', '6d': '2d+', '7d': '2d+'})
test['PAY_5F_coarse'] = test['PAY_5F'].replace({'2d': '2d+', '3d': '2d+', '4d': '2d+', '5d': '2d+', '6d': '2d+', '7d': '2d+'})

# calculation of WOE and IV now for coarse classes
bins_f_coarse = woe_iv(train, ['PAY_0F_coarse','PAY_2F_coarse','PAY_5F_coarse','EDUCATIONF_coarse'], y = 'def')
bins_f_coarse_test = woe_iv(test, ['PAY_0F_coarse','PAY_2F_coarse','PAY_5F_coarse','EDUCATIONF_coarse'], y = 'def')

# create pdf and save analitical plots for each coarse classed factor variable
pdf = PdfPages('WOE_coarse_fact.pdf')

for i in columns_f:
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    
    d = sns.barplot(data = train, x = str(i)+'_coarse', y = str(i), estimator = lambda y: round(100*len(y)/train.shape[0],2),
                      orient = 'v', palette = sns.cubehelix_palette(8),ax = ax[0][0])
    d.set_xticklabels(d.get_xticklabels(),rotation=45, horizontalalignment='right')
    for p in d.patches:
        d.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
    ax[0][0].set_title('Distribution \n ' + str(i) + ' train')
    ###############################################################################
    
    a = sns.barplot(data = test, x = str(i)+'_coarse', y = str(i), estimator = lambda y: round(100*len(y)/test.shape[0],2),
                      orient = 'v', palette = sns.cubehelix_palette(8),ax = ax[0][1])
    a.set_xticklabels(a.get_xticklabels(),rotation=45, horizontalalignment='right')
    for p in a.patches:
        a.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
    ax[0][1].set_title('Distribution \n ' + str(i) + ' test')
    
    
    ###############################################################################
    
    b = sns.barplot(data = train, x = str(i)+'_coarse', y = 'def', estimator = lambda y: round(100*y.sum()/len(y),2),
                    orient = 'v', palette = sns.cubehelix_palette(8),errwidth   = 0.0,ax = ax[1][0])
    b.set_xticklabels(b.get_xticklabels(),rotation=45, horizontalalignment='right')
    for p in b.patches:
        b.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
    ax[1][0].set_title('Bad rate \n ' + str(i))
    
    ###############################################################################
    
    c = sns.barplot(data = bins_f_coarse[str(i)+'_coarse'][bins_f_coarse[str(i)+'_coarse']['WOE'].notnull()],
                                x = 'Bin', y = 'WOE', palette = sns.cubehelix_palette(8),ax = ax[1][1])
    c.set_xticklabels(c.get_xticklabels(),rotation=45, horizontalalignment='right')
    for p in c.patches:
        c.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
    ax[1][1].set_title('WOE \n ' + str(i))
    
    plt.subplots_adjust(hspace=0.8)
    pdf.savefig()
    plt.close()
    
pdf.close()

# create WOE variables numerics
for i in columns_n:
    train[str(i) + '_woe'] = train[str(i) + '_coarse'].replace(bins_coarse[str(i) + '_coarse'][['Bin', 'WOE']][0:-1].set_index('Bin')['WOE'].to_dict())

for i in columns_n:
    test[str(i) + '_woe'] = test[str(i) + '_coarse'].replace(bins_coarse[str(i) + '_coarse'][['Bin', 'WOE']][0:-1].set_index('Bin')['WOE'].to_dict())


# create WOE variables factor
for i in columns_f:
    train[str(i)+'_woe'] = train[str(i) + '_coarse'].replace(bins_f_coarse[str(i) + '_coarse'][['Bin', 'WOE']][0:-1].set_index('Bin')['WOE'].to_dict())
    
for i in columns_f:
    test[str(i)+'_woe'] = test[str(i) + '_coarse'].replace(bins_f_coarse[str(i) + '_coarse'][['Bin', 'WOE']][0:-1].set_index('Bin')['WOE'].to_dict())

# create summary table      
stats_coarse = pd.concat([bins_coarse['IV'],bins_f_coarse['IV']]).sort_values(by = 'IV', ascending = False)
stats_coarse = stats_coarse.reset_index(drop = True)
stats_coarse['Gini'] = 0.0
stats_coarse['miss'] = 0.0
for i,j in enumerate(list(stats_coarse['VAR'].values)):
    stats_coarse['Gini'][i] = gini(train['def'].values, train[j.replace('_coarse', '_woe')].values)
    stats_coarse['miss'][i] = train[j].isnull().sum(axis=0)

stats_coarse_test = pd.concat([bins_coarse_test['IV'],bins_f_coarse_test['IV']]).sort_values(by = 'IV', ascending = False)
stats_coarse_test = stats_coarse_test.reset_index(drop = True)
stats_coarse_test['Gini_test'] = 0.0
stats_coarse_test['miss_test'] = 0.0
for i,j in enumerate(list(stats_coarse_test['VAR'].values)):
    stats_coarse_test['Gini_test'][i] = gini(test['def'].values, test[j.replace('_coarse', '_woe')].values)
    stats_coarse_test['miss_test'][i] = test[j].isnull().sum(axis=0)
stats_coarse_test = stats_coarse_test.rename(columns = {'IV': 'IV_test'})

stats_coarse = stats_coarse.merge(stats_coarse_test, left_on = 'VAR', right_on = 'VAR')
stats_coarse.to_csv('results_coarse_classing.csv', index = None, header = True)
###############################################################################
# save Data Sets for modelling
cols = list(train.filter(like = '_woe').columns)
data = train[['def'] + cols]
data_test = test[['def'] + cols]
data.to_csv('data.csv', index = None, header = True)
data_test.to_csv('data_test.csv', index = None, header = True)