import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, PowerTransformer,MinMaxScaler, RobustScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, recall_score, precision_recall_curve, f1_score
'''Requires clean up'''
np.random.seed(0)
random.seed(0)

pd.set_option('mode.chained_assignment', None)

df = pd.read_csv('bank-full.csv', sep = ';')
print(len(df))
yes_no = {'yes':1, 'no':0}

df['default'] = df['default'].map(yes_no).astype('int')
#df['loan'] =df['loan'].map(yes_no).astype('int')
df['housing'] = df['housing'].map(yes_no).astype('int')
#df['y'] = df['y'].map(yes_no).astype('int')

df.drop([29182], axis = 0, inplace = True)

df_X = df.drop(['y'], axis = 1)
df_Y = df[['y']]

X_tr, X_te, Y_tr, Y_te = train_test_split(df_X, df_Y, test_size = 0.20, random_state = 0)

'''Subscription distribution'''

sub = Y_tr['y'].value_counts()[1]
no_sub = Y_tr['y'].value_counts()[0]

percent_sub = (sub/Y_tr.shape[0])*100
percent_no = (no_sub/Y_tr.shape[0])*100

plt.figure()
sns.countplot(Y_tr['y'])
plt.xlabel('Subscribed')
plt.ylabel('Count')
plt.xticks((0,1), ['Yes ({0:.2f}%)'.format(percent_sub), 'No ({0:.2f}%)'.format(percent_no)])
plt.title('Training set term subscription distribution')
plt.show()

'''EDA - continuos variables'''
#For the purposes of EDA the X_tr and Y_tr variables are joined so that distribution can more easily be plotted.
df_train =pd.concat([X_tr, Y_tr], axis =1)
df_train.reset_index(inplace = True, drop = True)
subscribed = df_train['y']  == 'yes'
no_subscribed = df_train['y'] == 'no'
print(len(df_train))
print(len(df))


corr = df_train.corr(method = 'pearson')
sns.heatmap(corr, annot = True)
plt.show()


sns.set_style('whitegrid')
df1 = df_train.groupby('')
dfg1 = df1['y'].value_counts(normalize = True)
print(dfg1.unstack())
dfg1.unstack().plot(kind = 'bar')
plt.xticks(rotation = 45)
plt.ylabel('Count', fontsize = 12)
plt.xlabel('Marital', fontsize = 12)
plt.title('Distribution of marital status')
plt.tight_layout()
plt.show()
'''
'''
#multivariate distribution of the continuos feature and the target
sns.set_style('whitegrid')
sns.distplot(df_train[no_subscribed]['duration'], label='Did not subscribe', hist=True, kde = False, color='#e74c3c')
#sns.distplot(df_train[subscribed]['duration'], label='Did subscribe', hist=True,kde = False,  color='#2ecc71')
plt.xlabel('Duration', fontsize = 15)
plt.legend(loc='upper right', prop = {'size':13})
plt.title('Duration distribution by previous subscription outcome')
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)
plt.xlim(-1,4000)
plt.show()


sns.set_style('whitegrid')
df_train.groupby('job').mean()['balance'].plot(kind = 'bar')
#dfg1 = df1['y'].value_counts(normalize = True)
#print(dfg1.unstack())
#dfg1.unstack().plot(kind = 'bar')
plt.xticks(rotation = 45)
plt.ylabel('Mean balance', fontsize = 17)
plt.xlabel('Job', fontsize = 17)
plt.title('Mean balance by job', fontsize = 17)
plt.tick_params(axis='x', labelsize=15)
plt.tight_layout()
plt.show()



sns.set_style('whitegrid')
df_train['marital'].value_counts().sort_index().plot(kind = 'bar')
plt.xticks(rotation = 45)
plt.ylabel('Count', fontsize = 12)
plt.xlabel('Marital', fontsize = 12)
plt.title('Distribution of marital status')
plt.tight_layout()
plt.show()

df1 = df_train.groupby('marital')
dfg1 = df1['y'].value_counts(normalize = True)
print(dfg1.unstack())
dfg1.unstack().plot(kind = 'bar')
plt.xticks(rotation = 45)
plt.ylabel('Percentage', fontsize = 12)
plt.xlabel('Marital', fontsize = 12)
plt.title('Marital status distribution by term subscription outcome')
plt.tight_layout()
plt.show()

plt.figure()
sns.set_style('whitegrid')
sns.boxplot(x = df_train['job'], y = df_train['balance'], dodge = True)
plt.title('Boxplot of the clients balance against their  marital status', fontsize = 20)
plt.show()
sns.boxplot(x = df_train['job'], y = df_train['balance'], hue = df_train['y'], dodge = True)
plt.title('Boxplot of the clients balance against their  marital status', fontsize = 20)
plt.ylabel('Age', fontsize = 15)
plt.tick_params('y', labelsize = 15)
plt.xlabel('Jobs', fontsize = 15)
plt.show()

plt.figure()
sns.set_style('whitegrid')
sns.boxplot(x = df_train['poutcome'], y = df_train['pdays'], dodge = True)
plt.title('Boxplot of the number of days passed against success in a previous campaign', fontsize = 20)
plt.ylabel('Days passed', fontsize = 12)
plt.tick_params('y', labelsize = 12)
plt.xlabel('Outcome of a previous campaign', fontsize = 12)
plt.show()

sns.boxplot(x = df_train['poutcome'], y = df_train['pdays'], hue = df_train['y'], dodge = True)
plt.title('Boxplot of the number of days passed against success in a previous campaign', fontsize = 20)
plt.ylabel('Days passed', fontsize = 12)
plt.tick_params('y', labelsize = 12)
plt.xlabel('Outcome of a previous campaign', fontsize = 12)
plt.show()

avg_dur = df_train['duration'].mean()
sub_abv_dur = len(df_train.loc[(df_train['duration'] > avg_dur) & (df_train['y'] == 'yes')])
no_sub_abv_dur = len(df_train.loc[(df_train['duration'] > avg_dur) & (df_train['y'] == 'no')])

total_abv = sub_abv_dur + no_sub_abv_dur

sub_below_dur = len(df_train.loc[(df_train['duration'] <= avg_dur) & (df_train['y'] == 'yes')])
no_sub_below_dur = len(df_train.loc[(df_train['duration'] <= avg_dur) & (df_train['y'] == 'no')])

total_below = sub_below_dur + no_sub_below_dur

percent_yes_abv = (sub_abv_dur/total_abv)*100
percent_no_abv = (no_sub_abv_dur/total_abv)*100

percent_yes_bel = (sub_below_dur/total_below)*100
percent_no_bel = (no_sub_below_dur/total_below)*100

print('The perecntage who subscribed and were spoken to above the average duration was {0:.3f}%, while those who declined and were spoken to above the avergae was {1:.3f}%'.format(percent_yes_abv,percent_no_abv))
print('The perecntage who subscribed and were spoken to below the average duration was {0:.3f}%, while those who declined and were spoken to below the avergae was {1:.3f}%'.format(percent_yes_bel,percent_no_bel))


avg_day = df_train['pdays'].mean()
sub_abv_day = len(df_train.loc[(df_train['pdays'] > avg_day) & (df_train['y'] == 'yes')])
no_sub_abv_day = len(df_train.loc[(df_train['pdays'] > avg_day) & (df_train['y'] == 'no')])

total_abv_day = sub_abv_day + no_sub_abv_day

sub_below_day = len(df_train.loc[(df_train['pdays'] <= avg_day) & (df_train['y'] == 'yes')])
no_sub_below_day = len(df_train.loc[(df_train['pdays'] <= avg_day) & (df_train['y'] == 'no')])

total_below_day = sub_below_day + no_sub_below_day

percent_yes_abv_day = (sub_abv_day/total_abv_day)*100
percent_no_abv_day = (no_sub_abv_day/total_abv_day)*100

percent_yes_bel_day = (sub_below_day/total_below_day)*100
percent_no_bel_day = (no_sub_below_day/total_below_day)*100

print('The perecntage who subscribed and were spoken to above the average number of days was {0:.3f}%, while those who declined and were spoken to above the avergae was {1:.3f}%'.format(percent_yes_abv_day,percent_no_abv_day))
print('The perecntage who subscribed and were spoken to below the average number of days was {0:.3f}%, while those who declined and were spoken to below the avergae number of days was {1:.3f}%'.format(percent_yes_bel_day,percent_no_bel_day))

sub_abv_70 = len(df_train.loc[(df_train['age'] >= 70) & (df_train['y'] == 'yes')])
no_sub_abv_70 = len(df_train.loc[(df_train['age'] >= 70) & (df_train['y'] == 'no')])
total_abv_70 = sub_abv_70 + no_sub_abv_70
sub_below_25 = len(df_train.loc[(df_train['age'] <= 25) & (df_train['y'] == 'yes')])
no_sub_below_25 = len(df_train.loc[(df_train['age'] <= 25) & (df_train['y'] == 'no')])
total_below_25 = sub_below_25 + no_sub_below_25
percent_yes_abv_70 = (sub_abv_70/total_abv_70)*100
percent_no_abv_70 = (no_sub_abv_70/total_abv_70)*100
percent_yes_bel_25 = (sub_below_25/total_below_25)*100
percent_no_bel_25 = (no_sub_below_25/total_below_25)*100

print('The perecntage who subscribed and were spoken to above the age of 70 was {0:.3f}%, while those who declined and were above the age of 70 was {1:.3f}%'.format(percent_yes_abv_70,percent_no_abv_70))
print('The perecntage who subscribed and were spoken to below the age of 25 was {0:.3f}%, while those who declined and were spoken to below the age of 25 was {1:.3f}%'.format(percent_yes_bel_25,percent_no_bel_25))


df_day = pd.DataFrame([[percent_no_bel_day, percent_yes_bel_day],[percent_no_abv_day, percent_yes_abv_day]], index = ('Above avergae', 'Below average'), columns = ('No', 'Yes'))

sns.set_style('whitegrid')
df_day.plot(kind = 'bar')
plt.xticks(rotation = 45)
plt.ylabel('Percentage', fontsize = 12)
plt.xlabel('Number of days spoken to', fontsize = 12)
plt.title('Change in term deposit subscription with varying number of days passed')
plt.tight_layout()
plt.show()

avg_bal = df_train['balance'].mean()
sub_abv_bal = len(df_train.loc[(df_train['balance'] > avg_bal) & (df_train['loan'] == 'yes')])
no_sub_abv_bal = len(df_train.loc[(df_train['balance'] > avg_bal) & (df_train['loan'] == 'no')])

total_abv_bal = sub_abv_bal + no_sub_abv_bal

sub_below_bal = len(df_train.loc[(df_train['balance'] <= avg_bal) & (df_train['loan'] == 'yes')])
no_sub_below_bal = len(df_train.loc[(df_train['age'] <= avg_bal) & (df_train['loan'] == 'no')])

total_below_bal = sub_below_bal + no_sub_below_bal

percent_yes_abv_bal = (sub_abv_bal/total_abv_bal)*100
percent_no_abv_bal = (no_sub_abv_bal/total_abv_bal)*100

percent_yes_bel_bal = (sub_below_bal/total_below_bal)*100
percent_no_bel_bal = (no_sub_below_bal/total_below_bal)*100

print('The perecntage who recieveded a loan and were above the average yearly balance  was {0:.3f}%, while those who did not have a loan and were above the yearly balance was {1:.3f}%'.format(percent_yes_abv_bal,percent_no_abv_bal))
print('The perecntage who had a loan and were below the yearly income was {0:.3f}%, while those who did not have a loan and were below the average was {1:.3f}%'.format(percent_yes_bel_bal,percent_no_bel_bal))

df_bal = pd.DataFrame([[percent_no_bel_bal, percent_yes_bel_bal],[percent_no_abv_bal, percent_yes_abv_bal]], index = ('Below average', 'Above average'), columns = ('No', 'Yes'))

avg_day = df_train['pdays'].mean()
sub_abv_day_suc = len(df_train.loc[(df_train['pdays'] > avg_day) & (df_train['y'] == 'yes') & (df_train['poutcome'] == 'success')])
no_sub_abv_day_suc = len(df_train.loc[(df_train['pdays'] > avg_day) & (df_train['y'] == 'no') & (df_train['poutcome'] == 'success')])
total_abv_day_suc = sub_abv_day_suc + no_sub_abv_day_suc
sub_below_day_suc = len(df_train.loc[(df_train['pdays'] <= avg_day) & (df_train['y'] == 'yes') & (df_train['poutcome'] == 'success')])
no_sub_below_day_suc = len(df_train.loc[(df_train['pdays'] <= avg_day) & (df_train['y'] == 'no') & (df_train['poutcome'] == 'success')])
total_below_day_suc = sub_below_day_suc + no_sub_below_day_suc
percent_yes_abv_day_suc = (sub_abv_day_suc/total_abv_day_suc)*100
percent_no_abv_day_suc = (no_sub_abv_day_suc/total_abv_day_suc)*100
percent_yes_bel_day_suc = (sub_below_day_suc/total_below_day_suc)*100
percent_no_bel_day_suc = (no_sub_below_day_suc/total_below_day_suc)*100

sub_abv_day_fl = len(df_train.loc[(df_train['pdays'] > avg_day) & (df_train['y'] == 'yes') & (df_train['poutcome'] == 'failure')])
no_sub_abv_day_fl = len(df_train.loc[(df_train['pdays'] > avg_day) & (df_train['y'] == 'no') & (df_train['poutcome'] == 'failure')])
total_abv_day_fl = sub_abv_day_fl + no_sub_abv_day_fl
sub_below_day_fl = len(df_train.loc[(df_train['pdays'] <= avg_day) & (df_train['y'] == 'yes') & (df_train['poutcome'] == 'failure')])
no_sub_below_day_fl = len(df_train.loc[(df_train['pdays'] <= avg_day) & (df_train['y'] == 'no') & (df_train['poutcome'] == 'failure')])
total_below_day_fl = sub_below_day_fl + no_sub_below_day_fl
percent_yes_abv_day_fl = (sub_abv_day_fl/total_abv_day_fl)*100
percent_no_abv_day_fl = (no_sub_abv_day_fl/total_abv_day_fl)*100
percent_yes_bel_day_fl = (sub_below_day_fl/total_below_day_fl)*100
percent_no_bel_day_fl = (no_sub_below_day_fl/total_below_day_fl)*100
print(percent_no_abv_day_suc)
print(percent_yes_abv_day_suc)


df_day_sf = pd.DataFrame([[percent_no_abv_day_suc, percent_yes_abv_day_suc],[percent_no_abv_day_fl, percent_yes_abv_day_fl],[percent_no_bel_day_suc, percent_yes_bel_day_suc], [percent_no_bel_day_fl,percent_yes_bel_day_fl]], index = ('Above average \n & success', 'Above average \n & fail', 'Below average \n & success', 'Below average \n & fail'), columns = ('No', 'Yes'))
sns.set_style('whitegrid')
df_day_sf.plot(kind = 'bar')
plt.xlabel('Number of days since last campaign', fontsize = 11)
plt.ylabel('Percentage', fontsize = 12)
plt.xticks(rotation = 45)
plt.tick_params('x', labelsize = 11)
plt.title('Below and above average days since last campaign \n distributed by the previous outcome', fontsize = 12)
plt.tight_layout()
plt.show()

sns.set_style('whitegrid')
df_day.plot(kind = 'bar')
plt.xticks(rotation = 45)
plt.tick_params('x', labelsize = 13)
plt.ylabel('Percentage', fontsize = 15)
plt.xlabel('Balance', fontsize = 15)
plt.title('Change in personal loan with varying balance above and below the average', fontsize = 15)
plt.tight_layout()
plt.show()

print(avg_bal)
print(avg_dur)
print(avg_day)

'''Transformation and scaling of data'''
print(df_train.loc[(df_train['previous'] >= 250)])
scaler = PowerTransformer()
scale = df_train[['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']]
scaled = scaler.fit_transform(scale)
scale_df = pd.DataFrame(scaled, columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'])
print(scale_df.describe().unstack())

sns.distplot(df_train['age'], hist = False, kde = True)
sns.distplot(df_train['balance'], hist = False, kde = True)
sns.distplot(df_train['day'], hist = False, kde = True)
sns.distplot(df_train['duration'], hist = False, kde = True)
sns.distplot(df_train['campaign'], hist = False, kde = True)
sns.distplot(df_train['pdays'], hist = False, kde = True)
sns.distplot(df_train['previous'], hist = False, kde = True)
plt.xlabel('')
plt.ylabel('Density')
plt.show()

sns.distplot(scale_df['age'], hist = False, kde = True)
sns.distplot(scale_df['balance'], hist = False, kde = True)
sns.distplot(scale_df['day'], hist = False, kde = True)
sns.distplot(scale_df['duration'], hist = False, kde = True)
sns.distplot(scale_df['campaign'], hist = False, kde = True)
sns.distplot(scale_df['pdays'], hist = False, kde = True)
sns.distplot(scale_df['previous'], hist = False, kde = True)
plt.xlabel('')
plt.ylabel('Density')
plt.show()


'''Set up CV pipeline'''
scale_feats = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
OHE_feats = ['education', 'marital', 'contact', 'month', 'poutcome', 'job']

'''Pipeline'''
preprocessor = ColumnTransformer(
        transformers=[
        ('OHE', OneHotEncoder(sparse = False, handle_unknown='ignore'), OHE_feats),
        ('scale', PowerTransformer(), scale_feats)],
        remainder='passthrough')



model_pipeline = Pipeline(steps = [
    ('preprocessing', preprocessor),
    ('clf', XGBClassifier(learning_rate = 0.05, max_depth = 11, min_child_weight =4,gamma = 0.6, objective = 'binary:logistic'))])

param_test = {
        'clf__subsample':[i/100.0 for i in range(10, 110, 10)],
        'clf__colsample_bytree':[i/100.0 for i in range(10,110,10)]
            #'clf__gamma':[i/10.0 for i in range(0,11)]
            }



gsearch = GridSearchCV(model_pipeline, param_grid = param_test, scoring = 'f1', n_jobs=-1,iid=False, cv=5, verbose = 5,refit=True)

print(Y_tr.values.ravel())


gsearch.fit(X_tr, Y_tr.values.ravel())
print(gsearch.best_score_)
print(gsearch.best_params_)



models = [('Logistic Regression': LogisticRegression(C = 100, penalty = 'l2', random_state = 0, n_jobs = -1)), ('XGBoost': XGBClassifier(learning_rate = 0.05, min_child_weights = 1, n_estimators = 1000, max_depth = 8, gamma = 0.8, scale_pos_weights = 1, seed = 0, objective = 'binary:logistic', reg_alpha = 0.00031, reg_lambda = 0.7))]


'''Test and evaluation'''

def plot_confusion_matrix(df_confusion, title='Confusion matrix'):
    plt.matshow(df_confusion, cmap='RdBu') # imshow
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(0,len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name, fontsize = 12)
    plt.xlabel(df_confusion.columns.name, fontsize =12)
    plt.tick_params('x', labelsize = 15)
    plt.tick_params('y', labelsize = 15)
    for i in range(len(df_confusion.index)):
        for j in range(len(df_confusion.columns)):
            plt.text(j,i,str(df_confusion.iloc[i,j]))
    plt.show()
    plt.show()

power = PowerTransformer()
X_tr[scale_feats] = power.fit_transform(X_tr[scale_feats])
enc = OneHotEncoder(sparse = False, handle_unknown='ignore')

Ohe = pd.DataFrame(enc.fit_transform(X_tr[OHE_feats]), columns = enc.get_feature_names())
X_tr.drop(OHE_feats, inplace = True,  axis = 1)
X_tr = X_tr.reset_index(drop = True)
Train =  pd.concat([X_tr, Ohe], axis  = 1)
X_te[scale_feats] = power.transform(X_te[scale_feats])
Ohe_test = pd.DataFrame(enc.transform(X_te[OHE_feats]), columns = enc.get_feature_names())
X_te.drop(OHE_feats, inplace = True, axis = 1)
X_te = X_te.reset_index(drop = True)
Test = pd.concat([X_te, Ohe_test], axis = 1)

#print(Train.shape)
#print(Y_tr)

for name, model in models:
    print(name)
    #X_tr = preprocessor.fit_transform(X_tr)
    #X_te1 = preprocessor.transform(X_te)
    model.fit(Train, Y_tr.values.ravel())
    y_prob = model.predict_proba(Test)
    # predict probabilities
    # keep probabilities for the positive outcome only
    y_probs = y_prob[:, 1]
    # predict class values
    yhat = np.round(y_probs)
    y_precision, y_recall, _ = precision_recall_curve(Y_te, y_probs)
    y_f1, y_auc, recall1 = f1_score(Y_te, yhat), average_precision_score(Y_te, y_probs), recall_score(Y_te, yhat, average = 'micro')
    # summarize scores
    print('{0}: F1 = {1:.3f}, AP = {2:.3f}, Recall = {3:.3f}'.format(name, y_f1, y_auc, recall1))
    # plot the precision-recall curves
    no_skill = len(Y_te[Y_te['y']==1]) / len(Y_te)
    print(no_skill)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(y_recall, y_precision, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision - Recall curve: {}'.format(name))
    # show the legend
    plt.legend()
    # show the plot
    plt.show()

    y_actu = pd.Series(Y_te.values.ravel(), name='Actual')
    y_pred = pd.Series(np.round(yhat.ravel()), name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred)
    print(df_confusion)
    title = 'Confusion matrix: {}'.format(name)
    plot_confusion_matrix(df_confusion, title)

    if name == 'XGBoost':
        feat_imp = pd.Series(model.get_booster().get_fscore(), index = Train.columns).sort_values(ascending=False)
        #feat_imp = pd.Series(model.feature_importances_, index = Train.columns)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        plt.xticks(rotation=45)
        plt.tick_params('x', labelsize = 11)
        plt.show()
        print(model.get_booster().get_fscore())
'''
