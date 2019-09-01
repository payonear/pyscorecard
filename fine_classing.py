
"""
@author: payonear
"""
# import python file with data analyses and processing
from functions import *
# import 'base' Data Set (put the right directory)
base = pd.read_csv('PATH.../base.csv')
df = pd.read_csv('PATH.../df.csv')
# filer numeric variables
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
base_n = base.select_dtypes(include=numerics)
# factor variable selection
base_f = base.select_dtypes(exclude=numerics)

# less than 10 unique values filtering
numeric = base_n.columns[base_n.apply(lambda x: x.nunique()>10, axis=0)]
num_as_fact = base_n.columns[base_n.apply(lambda x: x.nunique()<10, axis=0)] # there are no such variables

# fine classing 10 groups (numeric variables)
# perform fine classing
fine_classing(base_n, numeric)


# default flag appending
base_n['def'] =  df['DEF']
base_f['def'] =  df['DEF']

# woe and informational value calculation
bins = woe_iv(base_n, list(base_n.filter(like='_fine').columns))


# plotting and variable analyses
# define pdf file directory
os.chdir('PATH.../')
# check whether it's ok
os.getcwd()

# create pdf and save analitical plots for each fine classed numeric variable
pdf = PdfPages('WOE_numeric.pdf')

for i in list(base_n.columns.drop(list(base_n.filter(like='_fine').columns)).drop('def')):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    
    d = sns.boxplot(data = base_n, x = i, y = 'def', orient = 'h', palette = sns.cubehelix_palette(8), ax = ax[0][0])
    d.set_xticklabels(d.get_xticklabels(),rotation=45, horizontalalignment='right')
    ax[0][0].set_title('Distribution \n ' + str(i))
    
    ###############################################################################
    
    a = sns.barplot(data = base_n, x = str(i)+'_fine', y = str(i), estimator = lambda y: round(100*len(y)/base_n.shape[0],2),
                      orient = 'v', palette = sns.cubehelix_palette(8),ax = ax[0][1])
    a.set_xticklabels(a.get_xticklabels(),rotation=45, horizontalalignment='right')
    for p in a.patches:
        a.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
    ax[0][1].set_title('Distribution \n ' + str(i))
    
    
    ###############################################################################
    
    b = sns.barplot(data = base_n, x = str(i)+'_fine', y = 'def', estimator = lambda y: round(100*y.sum()/len(y),2),
                    orient = 'v', palette = sns.cubehelix_palette(8),errwidth   = 0.0,ax = ax[1][0])
    b.set_xticklabels(b.get_xticklabels(),rotation=45, horizontalalignment='right')
    for p in b.patches:
        b.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
    ax[1][0].set_title('Bad rate \n ' + str(i))
    
    ###############################################################################
    
    c = sns.barplot(data = bins[str(i)+'_fine'][bins[str(i)+'_fine']['WOE'].notnull()],
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

# woe and iv calculation for categorical variables (mentioned as factors)
base_f.columns
# check number of unique values
for i in list(base_f.columns):
    print(i, len(base_f[i].unique()))

# look for levels with 100% or 0% Default Rate.
for i in list(base_f.columns.drop('def')):
    temp = base_f[[i, 'def']]
    temp = pd.pivot_table(data = temp, index = i, values = 'def', aggfunc = lambda x: sum(x)/len(x))
    for j in list(temp.index):
        if temp['def'][j] == 1 or temp['def'][j] == 0:
            print (i, j, temp['def'][j])

#PAY_4F 8d 1.0
#PAY_5F 8d 1.0
# Too few appearances

# check for appearances of each level and concatenate some levels to one
for i in base_f.columns.drop('def'):
    base_f[i] = base_f[i].replace(['8d','9d'], '7d')
    
# binning
bins_f = woe_iv(base_f, list(base_f.columns.drop('def')))
# creating one table with all IV by variables
stats = pd.concat([bins['IV'],bins_f['IV']]).sort_values(by = 'IV', ascending = False)
stats = stats.reset_index(drop = True)

# create pdf and save analitical plots for each factor varable
pdf = PdfPages('WOE_factor.pdf')

for i in list(base_f.columns.drop('def')):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    
    a = sns.barplot(data = base_f, x = str(i), y = str(i), estimator = lambda y: round(100*len(y)/base_f.shape[0],2),
                      orient = 'v', palette = sns.cubehelix_palette(8),ax = ax[0][0])
    a.set_xticklabels(a.get_xticklabels(),rotation=45, horizontalalignment='right')
    for p in a.patches:
        a.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
    ax[0][0].set_title('Distribution \n ' + str(i))
    
    
    ###############################################################################
    
    b = sns.barplot(data = base_f, x = str(i), y = 'def', estimator = lambda y: round(100*y.sum()/len(y),2),
                    orient = 'v', palette = sns.cubehelix_palette(8),errwidth   = 0.0,ax = ax[0][1])
    b.set_xticklabels(b.get_xticklabels(),rotation=45, horizontalalignment='right')
    for p in b.patches:
        b.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
    ax[0][1].set_title('Bad rate \n ' + str(i))
    
    ###############################################################################
    
    c = sns.barplot(data = bins_f[str(i)][bins_f[str(i)]['WOE'].notnull()],
                                x = 'Bin', y = 'WOE', palette = sns.cubehelix_palette(8),ax = ax[1][0])
    c.set_xticklabels(c.get_xticklabels(),rotation=45, horizontalalignment='right')
    for p in c.patches:
        c.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
    ax[1][0].set_title('WOE \n ' + str(i))
    
    plt.subplots_adjust(hspace=0.8)
    pdf.savefig()
    plt.close()
    
pdf.close()

# concatanation of created variables
base_1 = pd.concat([base_n.drop(['def'], axis = 1), base_f], axis = 1)
###############################################################################
###############################################################################
###############################################################################

# append WOE variables numerics
for i in list(base_n.filter(like='_fine').columns):
    base_1[str(i.replace('_fine', '_woe'))] = base_1[i].replace(bins[i][['Bin', 'WOE']][0:-1].set_index('Bin')['WOE'].to_dict())

# append WOE variables factor
for i in list(base_f.columns.drop('def')):
    base_1[str(i)+'_woe'] = base_1[i].replace(bins_f[i][['Bin', 'WOE']][0:-1].set_index('Bin')['WOE'].to_dict())
###############################################################################
# Gini and missings calculation
stats['Gini'] = 0.0
stats['miss'] = 0.0

for i,j in enumerate(list(stats['VAR'].values)):
    stats['Gini'][i] = gini(base_1['def'].values, base_1[j.replace('_fine', '_woe').replace('F', 'F_woe')].values)
    stats['miss'][i] = base_1[j].isnull().sum(axis=0)
###############################################################################
var_analyse = base_1[base_1.filter(like = '_woe').columns]
var_analyse['def'] = base_1['def']
# correlation calculations
corr = var_analyse.corr(method ='pearson')
corr_kendalltau = var_analyse.corr(method ='kendall')

stats['VAR'] = stats['VAR'].replace({'_fine': ''}, regex = True) 
stats['VARW'] = stats['VAR'].astype(str) + '_woe'
stats = stats.merge(corr_kendalltau, left_on = 'VARW', right_index = True)
stats = stats.sort_values(by = ['Gini'], ascending = False)
stats['VAR']

temp_k = pd.concat([stats['VARW'],stats[list(stats['VARW'])]], axis = 1)

# as in temp_k variables were sorted by Gini, we dropped correlated variables with lower Gini than those which left
results_fine_classing = stats[['VARW', 'IV', 'Gini']].merge(correl_var(temp_k), left_on = 'VARW', right_index = True)
# choose variables with Gini higher than 0.08
results_fine_classing = results_fine_classing[results_fine_classing['Gini'] > 0.08]
results_fine_classing.to_csv('results_fine_classing.csv', index = None, header = True)
# create Data Set for coarse classing
base_coarse = base_1[list(results_fine_classing['VARW'].replace({'_woe': ''}, regex = True)) + ['def']]
base_coarse.to_csv('base_coarse.csv', index = None, header = True)