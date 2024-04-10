
# Written by Narmin Ghaffari Laleh - Nov 2020
# narminghaffari23@gmail.com

#%%

import pandas as pd 
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib
from sklearn import metrics
import os

#%%
input_path = '###.xlsx'
output_path = '###'
sheet_name = '###'

data = pd.read_excel(open(input_path, 'rb'), sheet_name = sheet_name) 

x_labels = ['Fever (yes/no)', 'Headache (yes/no)', 'LDH >300 U/dL (yes/no)', 'AKI as Crea >=0,3 mg/dL ULN on DOA (yes/no)', 'Platelets <150/nL on DOA (yes/no)']
y_label = ['Hanta-positive (yes/no)']

filtered_data = data[x_labels] 

imp = SimpleImputer(strategy="most_frequent")
x = np.array(imp.fit_transform(filtered_data))

y = np.array(data[y_label])
hantaVirusCount = np.sum(y)

#%%
classifier = LogisticRegression()
classifier.fit(x, y)
print(classifier.coef_)

points = []
for item in classifier.coef_[0]:
    if item>0:
        item = np.floor(item) 
    else:
        item = np.ceil(item) 
    points.append(abs(item))
print(points)

#%%
scoreList = []

# Assuming data is a pandas DataFrame, x_labels is a list of column names, and points is a list of corresponding scores
for i in range(len(data)):
    score = 0
    for j, label in enumerate(x_labels):
        if data.iloc[i][label] == 1:
            score += points[j]
    scoreList.append(score)

data['scores'] = scoreList

#%%
# Calculate prediction o the model based on the scores. 
y_preds = [[0] * len(y) for _ in range(8)]
for i, score in enumerate(data['scores']):
    for j in range(8):
        if score >= j:
            y_preds[j][i] = 1
#%%
def CalculateConfidenceInterval(scoreList):
    nsamples = 1000
    probabilityList = []
    for b in range(nsamples):
        probability = 0
        data_temp = data.loc[data['scores'].isin(scoreList)]
        idx = np.random.randint(len(data_temp), size = len(data_temp))        
        data_temp = data_temp.iloc[idx]
        scoreList_temp = list(data_temp['scores'])
        if len(scoreList)>1:
            score = scoreList_temp.count(scoreList[0]) + scoreList_temp.count(scoreList[1])
            for i in range(len(data_temp)):
                if (data_temp.iloc[i]['scores'] == scoreList[0] or data_temp.iloc[i]['scores'] == scoreList[1]) and data_temp.iloc[i][y_label[0]] == 1:
                    probability += 1
        else:
            score = scoreList_temp.count(scoreList[0])
            for i in range(len(data_temp)):
                if data_temp.iloc[i]['scores'] == scoreList[0] and data_temp.iloc[i][y_label[0]] == 1:
                    probability += 1
        probabilityList.append(np.round((probability / score) * 100))
    probabilityList.sort()
    confidence_lower = probabilityList[int(0.025 * len(probabilityList))]
    confidence_upper = probabilityList[int(0.975 * len(probabilityList))]
    return confidence_lower, confidence_upper

#%%
class_names = ['Healthy', 'Infected']
def plot_confusion_matrix_m(cm, classes,normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    
    matplotlib.rcParams.update({'font.size': 25})
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 25)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes, rotation=90)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label\n')
    plt.xlabel('\nPredicted label')

#%% 
def CalculateParameters(cm):
    total = sum(sum(cm))
    accuracy = (cm[0,0]+cm[1,1])/total
    accuracy = np.round(accuracy, 4)*100
    sensitivity = cm[1,1]/(cm[1,1]+cm[1,0])
    sensitivity = np.round(sensitivity, 4)*100
    specificity = cm[0,0]/(cm[0,0]+cm[0,1])
    specificity = np.round(specificity, 4)*100
    return accuracy, sensitivity, specificity
    
#%%  Initialize dictionaries to store probabilities and score counts
probabilities = {f'probability_{i}': 0 for i in range(0, 8)}
score_counts = {f'score_{i}': scoreList.count(i) for i in range(8)}
results = {f'score_{i}': scoreList.count(i) for i in range(8)}

# Update probabilities based on the conditions
for i in range(8):
    for _, (score, hanta_positive) in enumerate(zip(data['scores'], data[y_label[0]])):
        if int(score) == i and hanta_positive == 1:
            key = f'probability_{i}'
            probabilities[key] += 1    
                
for i in range(8):
    score_key = f'score_{i}'
    prob_key = f'probability_{i}'
    if score_counts[score_key] != 0:  # Avoiding division by zero and using i > 0 to skip score_0
        probability = np.round((probabilities[prob_key] / score_counts[score_key]) * 100)
        confidence_lower, confidence_upper = CalculateConfidenceInterval([i])
        plt.figure()
        cm = confusion_matrix(y, y_preds[i])
        plot_confusion_matrix_m(cm, classes=class_names, title=f'Score >= {i}')
        plt.savefig(os.path.join(output_path, f'confusion_matrix_score_{i}.svg'), format='svg')
        accuracy, sensitivity, specificity = CalculateParameters(cm)
        results[score_key] = [probability, confidence_lower, confidence_upper, accuracy, sensitivity, specificity]

# Convert dictionary to DataFrame and transpose it
results_df = pd.DataFrame(results).T

# Assigning new column names
results_df.columns = ["probability(%)", "confidence_lower(%)", "onfidence_upper(%)", "accuracy(%)", "sensitivity(%)", "specificity(%)"]

# Save the DataFrame to an Excel file
results_df.to_excel(os.path.join(output_path, 'results.xlsx'), index_label='Results')

#%%
# 5 fold cross validation
results_auc = {}
results_auc['tpr'] = {}
results_auc['fpr'] = {}
results_auc['auc'] = {}

randomState = 23
matplotlib.rcParams.update({'font.size': 20})
fpr_temp = np.linspace(0, 1, 100)
for i in range(5):
    classifier = LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,  random_state = randomState)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)
    results_auc['tpr'][i] = np.interp(fpr_temp, fpr, tpr)
    results_auc['tpr'][i][0] = 0.0
    results_auc['fpr'][i] = fpr_temp
    results_auc['auc'][i] = auc
    randomState += 10

tpr_mean5 = [(a+b+c+d+e) / 5 for a, b, c, d, e in zip(list(results_auc['tpr'][0]), list(results_auc['tpr'][1]), list(results_auc['tpr'][2]), list(results_auc['tpr'][3]), list(results_auc['tpr'][4]))]
fpr_mean5 = [(a+b+c+d+e) / 5 for a, b, c, d, e in zip(list(results_auc['fpr'][0]), list(results_auc['fpr'][1]), list(results_auc['fpr'][2]), list(results_auc['fpr'][3]), list(results_auc['fpr'][4]))]
auc_mean5 = metrics.auc(fpr_mean5, tpr_mean5)

aucs = np.array(list(results_auc['auc'].values()))
aucs.sort()
confidence_lower_mean = aucs[int(0.025 * len(aucs))]
confidence_upper_mean = aucs[int(0.975 * len(aucs))]

# Single Fold
random_state = 40    
classifier = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,  random_state = randomState)
classifier.fit(X_train, y_train)
y_pred = classifier.predict_proba(X_test)[::,1]
fpr_single, tpr_single, _ = metrics.roc_curve(y_test,  y_pred)
auc_single = metrics.roc_auc_score(y_test, y_pred)
auc_CI = []
nsamples = 1000
for b in range(nsamples):
    idx = np.random.randint(X_test.shape[0], size = X_test.shape[0])
    if len(np.unique(X_test[idx])) < 2 or np.sum(y_test[idx]) == 0 or np.sum(y_test[idx]) == len(y_test[idx]):
        continue   
    y_pred_proba = classifier.predict_proba(X_test[idx])[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test[idx],  y_pred_proba)
    auc_temp = metrics.roc_auc_score(y_test[idx], y_pred_proba)
    auc_CI.append(auc_temp)
    
auc_CI = np.array(auc_CI)
auc_CI.sort()
confidence_lower_single = auc_CI[int(0.025 * len(auc_CI))]
confidence_upper_single = auc_CI[int(0.975 * len(auc_CI))]
#Plot
fig, ax = plt.subplots(figsize=(10, 10))   
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', alpha=.8)
ax.plot(fpr_single, tpr_single, color = 'b', label = r'AUC = ' + str(round(auc_single,2)) + '(' + str(round(confidence_lower_single, 2)) + ' - ' + str(round(confidence_upper_single, 2)) + ')', lw = 2, alpha = .8)
ax.plot(fpr_mean5, tpr_mean5, color = 'r', label = 'Mean AUC = ' + str(round(auc_mean5,2)) + '(' + str(round(confidence_lower_mean, 2)) + ' - ' + str(round(confidence_upper_mean, 2)) + ')', lw =4, alpha = .8)

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
ax.legend(loc="lower right", fontsize='large')
ax.set_xlabel('1 - Specificity', fontsize='large', fontweight='bold')
ax.set_ylabel('Sensitivity', fontsize='large', fontweight='bold')
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
plt.savefig(os.path.join(output_path, f'AUCs.svg'), format='svg')
plt.show()

# %%
#fig, ax = plt.subplots()
#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,  random_state = 73)
#classifier = LogisticRegression()
#classifier.fit(X_train, y_train)
#viz = plot_roc_curve(classifier, X_test, y_test, name='ROC', lw=1, ax=ax)
#ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)
#plt.title('Receiver Operating Characteristic')
#plt.legend(loc = 'lower right')#
#plt.xlim([0, 1])
#plt.ylim([0, 1])
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.show()
#disp = plot_confusion_matrix(classifier, X_test, y_test,display_labels = ['Healthy', 'Infected'],cmap = plt.cm.Blues,normalize = None)
