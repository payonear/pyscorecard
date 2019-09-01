
"""
@author: payonear
"""
from libraries import *

def fine_classing(data, var, bins = 10):
    """
    A function that looks for percentiles of numeric variables to cut it on equally distributed parts.
    In other words, it adds columns with classified numeric variables (categorical variables). 
    The label presents an appropriate interval.
    data - pandas DataFrame to be modified;
    var - columns' names for classification;
    bins - number of bins to create.    
    """
    percentile = data.apply(lambda x: np.round(np.quantile(x, q = np.linspace(1/bins,1,bins)),2),axis=0)

    if len(str(list(var)).split()) > 1:
        
        for i in var:
            data[str(i)+'_fine'] = pd.cut(   x = data[i], 
                                               bins = [- math.inf] + list(np.unique(percentile[i].values)),
                                               labels = ["<=" + str(i) for i in np.unique(percentile[i].values)],
                                               duplicates = 'drop')
    else:
            data[str(var)+'_fine'] = pd.cut(   x = data[var], 
                                               bins = [- math.inf] + list(np.unique(percentile[var].values)),
                                               labels = ["<=" + str(var) for var in np.unique(percentile[var].values)],
                                               duplicates = 'drop') 


def woe_iv(data, colnames, y = 'def'):
    """
    Function creates dictionary with calculated Weight of Evidence for each bin of given variables
    and calculates the Informational Value of each variable.
    data - pandas DataFrame to be analysed;
    colnames - columns' names of variables to analyse;
    y - default_flag variable.
    """
    woe_iv = {}
    for col in colnames:
        data = data.sort_values(by = [col])
        df1 = pd.DataFrame(columns = ['Bin', 'Non-events', 'Events', '% of Non-events',
                                      '% of Events', '% of Non-events/% of Events','WOE', 'IV'], dtype = 'float64')
        df1['Bin'] = data[col].unique()
        for i,j in enumerate(list(df1['Bin'])):
            df1['Non-events'][i] =  data.loc[(data[col] == j) & (data[y] == 0)][col].count()
            df1['Events'][i] =  data.loc[(data[col] == j) & (data[y] == 1)][col].count()
        df1['% of Non-events'] = df1['Non-events']/df1['Non-events'].sum()
        df1['% of Events'] = df1['Events']/df1['Events'].sum()
        df1['% of Non-events/% of Events'] = df1['% of Non-events']/df1['% of Events']
        df1['WOE'] = np.where(df1['% of Events'] == 0,0,np.where(df1['% of Non-events'] == 0,
                                   float('NaN'),(np.log(df1['% of Non-events/% of Events']))))
        df1['IV'] = (df1['% of Non-events'] - df1['% of Events'])*df1['WOE']
        df1 = df1.append({'Bin': 'Total', 'IV': df1['IV'].sum()}, ignore_index=True)
        woe_iv.update({col: df1})
        del df1
    iv = pd.DataFrame(columns = ['VAR', 'IV'])
    iv['VAR'] = list(woe_iv.keys())
    for i,j in enumerate(list(woe_iv.keys())):
        iv['IV'][i] = woe_iv[j]['IV'][len(woe_iv[j]['IV'])-1]
    woe_iv.update({'IV': iv})
    return woe_iv


def correl_var(df, threshold = 0.5):
    """
    Function creates a pandas DataFrame. It checks the correlation coefficient of each variable with all variables, 
    which are above on the list (due to the sequence given by inputed df). If the correlation is higher than threshold,
    the variable is droped and added in the next column in front of varible, it's correlated with.
    df - pandas DataFrame with correlation coefficients (2 dimensional matrix);
    threshold - threshold of correlation coefficient.
    """
    df_copy = df.copy()
    cor_var = {}
    out_tuple = ()
    while len(df_copy) > 0:
        if list(df_copy['VARW'])[0] in out_tuple:
            df_copy = df_copy.drop(df_copy.index[:1])
        else:
            var = list(df_copy['VARW'])[0]
            cor = df_copy.iloc[0,1:].loc[lambda x: x> threshold].index.drop(var).drop(list(out_tuple),errors ='ignore')
            out_tuple += tuple(cor)
            cor_var[var] = list(cor)
            df_copy = df_copy.drop(df_copy.index[:1])
    cor_df = pd.DataFrame.from_dict(cor_var, orient = 'index')
    return cor_df

def coarse_classing(data, coarse_dict, var):
    """
    Function creates categorical variable from numerical, which is cut by cut-points given by coarse_dict
    and ascribes appropriate interval for each record. 
    Adds additional column to given DataFrame with categorical variable.
    data - pandas DataFrame to be modified;
    coarse_dict - dictionary with cut-points for each given var;
    var - variables to be classified.    
    """
    if len(str(list(var)).split()) > 1:
        
        for i in var:
            data[str(i)+'_coarse'] = pd.cut(   x = data[i], 
                                               bins = [- math.inf] + list(np.unique(coarse_dict[i])),
                                               labels = ["<=" + str(i) for i in np.unique(coarse_dict[i])],
                                               duplicates = 'drop')
            data[str(i)+'_coarse'] = data[str(i)+'_coarse'].cat.add_categories(['>' + str(list(np.unique(coarse_dict[i]))[-1])])
            data[str(i)+'_coarse'] = data[str(i)+'_coarse'].fillna('>' + str(list(np.unique(coarse_dict[i]))[-1]))
    else:
            data[str(var)+'_coarse'] = pd.cut(   x = data[var], 
                                               bins = [- math.inf] + list(np.unique(coarse_dict[var].values)),
                                               labels = ["<=" + str(var) for var in np.unique(coarse_dict[var])],
                                               duplicates = 'drop')
            data[str(i)+'_coarse'] = data[str(i)+'_coarse'].cat.add_categories(['>' + str(list(np.unique(coarse_dict[i]))[-1])])
            data[str(i)+'_coarse'] = data[str(i)+'_coarse'].fillna('>' + str(list(np.unique(coarse_dict[i]))[-1]))
            

def gini(fpr, tpr):
    """
    Function calculates Gini coefficient.
    fpr - the vector of defaults;
    tpr - the vector of variable values.
    """
    return -(2 * sm.roc_auc_score(fpr, tpr) - 1)

