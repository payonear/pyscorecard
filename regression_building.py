
"""
@author: payonear
"""
from coarse_classing import *

# import Data Sets (put the right directory)
data = pd.read_csv('PATH.../data.csv')
data_test = pd.read_csv('PATH.../data_test.csv')

X = data[data.filter(like = '_woe').columns]
y = data['def']
X_test = data_test[data_test.filter(like = '_woe').columns]
y_test = data_test['def']

# first model without regularization and any feature selection models
# sklearn.LR does not allow the lack of penalty, but we can minimilize the impact by setting C parameter large
lr = LogisticRegression(penalty = 'l2', C = 10**40, fit_intercept = True, solver='lbfgs')
lr.fit(X,y)
y_pred_train = lr.predict(X)
y_pred = lr.predict(X_test)

sm.accuracy_score(y,y_pred_train) # 0.8201428571428572
sm.accuracy_score(y_test,y_pred) # 0.8174444444444444
print(sm.classification_report(y_test, y_pred))

# AUC_ROC plot
logit_roc_auc = sm.roc_auc_score(y_test, y_pred)
fpr, tpr, thresholds = sm.roc_curve(y_test, lr.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC_model_1')
plt.show()

# let's try LASSO and l2
grid={"C":np.logspace(-3,3,7), 'penalty':['l1','l2']}
lr_r = LogisticRegression()
lr_r_cv = GridSearchCV(lr_r,grid, cv = 5)
lr_r_cv.fit(X,y)
lr_r_cv.cv_results_ 
lr_r_cv.best_params_

# let's perform a regression
lr_r = LogisticRegression(C = 0.01, penalty = 'l1', fit_intercept = True)
lr_r.fit(X,y)
y_pred_train_2 = lr_r.predict(X)
y_pred_2 = lr_r.predict(X_test)

sm.accuracy_score(y,y_pred_train_2) # 0.8204285714285714 (better on train sample)
sm.accuracy_score(y_test,y_pred_2) # 0.8174444444444444 (but the same result as previously for test sample)
print(sm.classification_report(y_test, y_pred_2))

# Let's implement feature selection
# Recursive Feature Elimination
rfe_1 = RFE(estimator=lr, n_features_to_select = 6, step=1)
rfe_1.fit(X, y)

rfe_2 = RFE(estimator=lr_r, n_features_to_select = 6, step=1) # shows the best effect
rfe_2.fit(X, y)

ranking_1 = pd.DataFrame(list(zip(list(X.columns), list(rfe_1.ranking_), lr.coef_[0].tolist())), columns = ['VAR' , 'Rank', 'Coef']).sort_values(by = 'Rank')
ranking_2 = pd.DataFrame(list(zip(list(X.columns), list(rfe_2.ranking_), lr_r.coef_[0].tolist())), columns = ['VAR' , 'Rank', 'Coef']).sort_values(by = 'Rank')
 
lr.fit(X[list(ranking_1[ranking_1['Rank'] == 1]['VAR'])], y)
y_pred_train = lr.predict(X[list(ranking_1[ranking_1['Rank'] == 1]['VAR'])])
y_pred = lr.predict(X_test[list(ranking_1[ranking_1['Rank'] == 1]['VAR'])])

sm.accuracy_score(y,y_pred_train) # 0.8186190476190476
sm.accuracy_score(y_test,y_pred) # 0.8181111111111111
print(sm.classification_report(y_test, y_pred))

lr_r.fit(X[list(ranking_2[ranking_2['Rank'] == 1]['VAR'])], y)
y_pred_train_2 = lr_r.predict(X[list(ranking_2[ranking_2['Rank'] == 1]['VAR'])])
y_pred_2 = lr_r.predict(X_test[list(ranking_2[ranking_2['Rank'] == 1]['VAR'])])


sm.accuracy_score(y,y_pred_train_2) # 0.8203809523809524
sm.accuracy_score(y_test,y_pred_2) # 0.8178888888888889
print(sm.classification_report(y_test, y_pred_2))

sm.roc_auc_score(y_test, y_pred_2)
fpr, tpr, thresholds = sm.roc_curve(y_test, lr_r.predict_proba(X_test[list(ranking_2[ranking_2['Rank'] == 1]['VAR'])])[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC_model_2')
plt.show()

# create data to score
data_score_train = data[list(ranking_2[ranking_2['Rank'] == 1]['VAR']) + ['def']]
data_score_test = data_test[list(ranking_2[ranking_2['Rank'] == 1]['VAR']) + ['def']]

# calculate score
data_score_train['LIMIT_BAL_param'] = lr_r.coef_[0].tolist()[0]
data_score_train['PAY_AMT1_param'] = lr_r.coef_[0].tolist()[1]
data_score_train['PAY_AMT2_param'] = lr_r.coef_[0].tolist()[2]
data_score_train['PAY_0F_param'] = lr_r.coef_[0].tolist()[3]
data_score_train['PAY_2F_param'] = lr_r.coef_[0].tolist()[4]
data_score_train['PAY_5F_param'] = lr_r.coef_[0].tolist()[5]

data_score_train['score'] = -100 * ( data_score_train['LIMIT_BAL_param']*data_score_train['LIMIT_BAL_woe'] + 
                                            data_score_train['PAY_AMT1_param']*data_score_train['PAY_AMT1_woe'] +
                                            data_score_train['PAY_AMT2_param']*data_score_train['PAY_AMT2_woe'] +
                                            data_score_train['PAY_0F_param']*data_score_train['PAY_0F_woe'] +
                                            data_score_train['PAY_2F_param']*data_score_train['PAY_2F_woe'] +
                                            data_score_train['PAY_5F_param']*data_score_train['PAY_5F_woe'])

data_score_test['LIMIT_BAL_param'] = lr_r.coef_[0].tolist()[0]
data_score_test['PAY_AMT1_param'] = lr_r.coef_[0].tolist()[1]
data_score_test['PAY_AMT2_param'] = lr_r.coef_[0].tolist()[2]
data_score_test['PAY_0F_param'] = lr_r.coef_[0].tolist()[3]
data_score_test['PAY_2F_param'] = lr_r.coef_[0].tolist()[4]
data_score_test['PAY_5F_param'] = lr_r.coef_[0].tolist()[5]

data_score_test['score'] = -100 * ( data_score_test['LIMIT_BAL_param']*data_score_test['LIMIT_BAL_woe'] + 
                                            data_score_test['PAY_AMT1_param']*data_score_test['PAY_AMT1_woe'] +
                                            data_score_test['PAY_AMT2_param']*data_score_test['PAY_AMT2_woe'] +
                                            data_score_test['PAY_0F_param']*data_score_test['PAY_0F_woe'] +
                                            data_score_test['PAY_2F_param']*data_score_test['PAY_2F_woe'] +
                                            data_score_test['PAY_5F_param']*data_score_test['PAY_5F_woe'])

# Gini
gini(data_score_train['def'].values, data_score_train['score'].values) # 0.5342862925292858
gini(data_score_test['def'].values, data_score_test['score'].values) # 0.5293276872477727
# Let's calculate Grades
percentile = data_score_train.apply(lambda x: np.round(np.quantile(x, q = np.linspace(1/10,1,10)),2),axis=0)

data_score_train['Grade'] = pd.cut( x = data_score_train['score'], bins = [- math.inf] + list(np.unique(percentile['score'].values)),
                                    labels = [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5],
                                    duplicates = 'drop')
data_score_test['Grade'] = pd.cut( x = data_score_test['score'], bins = [- math.inf] + list(np.unique(percentile['score'].values)),
                                    labels = [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5],
                                    duplicates = 'drop')


fig, ax = plt.subplots(2, 2, figsize=(12, 8))
a = sns.barplot( data = data_score_train, x = 'Grade', y = 'def', estimator = lambda y: round(100*y.sum()/len(y),2),
                orient = 'v', palette = sns.cubehelix_palette(8),errwidth   = 0.0, ax = ax[0][0])
a.set_xticklabels(a.get_xticklabels(),rotation=45, horizontalalignment='right')
for p in a.patches:
    a.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),\
                   ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
ax[0][0].set_title('Bad rate train')
###############################################################################
b = sns.barplot( data = data_score_test, x = 'Grade', y = 'def', estimator = lambda y: round(100*y.sum()/len(y),2),
                orient = 'v', palette = sns.cubehelix_palette(8),errwidth   = 0.0, ax = ax[1][0])
b.set_xticklabels(b.get_xticklabels(),rotation=45, horizontalalignment='right')
for p in b.patches:
    b.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),\
                   ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
ax[1][0].set_title('Bad rate test')
###############################################################################
e = sns.barplot( data = data_score_train, x = 'Grade', y = 'def', estimator = lambda y: round(100*len(y)/data_score_train.shape[0],2),
                orient = 'v', palette = sns.cubehelix_palette(8),ax = ax[0][1])
e.set_xticklabels(e.get_xticklabels(),rotation=45, horizontalalignment='right')
for p in e.patches:
    e.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),\
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
ax[0][1].set_title('Distribution URG train')
###############################################################################
d = sns.barplot( data = data_score_test, x = 'Grade', y = 'def', estimator = lambda y: round(100*len(y)/data_score_test.shape[0],2),
                orient = 'v', palette = sns.cubehelix_palette(8),ax = ax[1][1])
d.set_xticklabels(d.get_xticklabels(),rotation=45, horizontalalignment='right')
for p in d.patches:
    d.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),\
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
ax[1][1].set_title('Distribution URG test')

plt.subplots_adjust(hspace=0.8)
plt.savefig('Scorecard_grades')
plt.show()

# Final Scorecard
Scorecard = {}
for i, j in enumerate(list(ranking_2[ranking_2['Rank'] == 1]['VAR'])):
    if str(j.replace('_woe', '_coarse'))  in list(bins_coarse.keys()):
        a = bins_coarse[j.replace('_woe', '_coarse')][['Bin', 'WOE']][:-1]
        a['param'] = list(lr_r.coef_[0])[i]
        a['Score'] = -100*a['WOE']*a['param']
        a = a[['Bin', 'Score']]
        Scorecard[str(j.replace('_woe', ''))] = a
    else:
        a = bins_f_coarse[j.replace('_woe', '_coarse')][['Bin', 'WOE']][:-1]
        a['param'] = list(lr_r.coef_[0])[i]
        a['Score'] = -100*a['WOE']*a['param']
        a = a[['Bin', 'Score']]
        Scorecard[str(j.replace('_woe', ''))] = a
    del(a)