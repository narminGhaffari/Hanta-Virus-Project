
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
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib
from sklearn import metrics

#%%

data = pd.read_excel(open(r"###", 'rb'), sheet_name='Hanta vs. AKI_Thrombopenie') 

data_temp = data[['Fever', 'Visual disturbance/headache', 'LDH >300', 'Thrombo <150', 'AKI']] 
     
imp = SimpleImputer(strategy="most_frequent")
x = np.array(imp.fit_transform(data_temp))

y = data[['Hanta-pos']]

hantaVirusCount = np.sum(y)
y = imp.fit_transform(y)

#%%

classifier = LogisticRegression()
classifier.fit(x, y)
print(classifier.coef_)

points = np.abs(np.round(classifier.coef_))
points = points[0]
print(points)
#%%

scoreList = []

for i in range(len(data)):
    score = 0
    if data.iloc[i]['Fever'] == 1:
        score += points[0]
    if data.iloc[i]['Visual disturbance/headache'] == 1:
        score += points[1]
    if data.iloc[i]['LDH >300'] == 1:
        score += points[2]
    if data.iloc[i]['Thrombo <150'] == 1:
        score += points[3]
    if data.iloc[i]['AKI'] == 1:
        score += points[4]
    scoreList.append(score)

data['scores'] = scoreList

#%%
# Calculate prediction o the model based on the scores. 

y_pred_0 = [0] * len(y)
y_pred_1 = [0] * len(y)
y_pred_2 = [0] * len(y)
y_pred_3 = [0] * len(y)
y_pred_4 = [0] * len(y)
y_pred_5 = [0] * len(y)
y_pred_6 = [0] * len(y)
y_pred_7 = [0] * len(y)

for i in range(len(data)):
    
    if data.iloc[i]['scores'] >= 0 :
        y_pred_0[i] = 1
        
    if data.iloc[i]['scores'] >= 1 :
        y_pred_1[i] = 1
        
    if data.iloc[i]['scores'] >= 2 :
        y_pred_2[i] = 1
        
    if data.iloc[i]['scores'] >= 3:
        y_pred_3[i] = 1
        
    if data.iloc[i]['scores'] >= 4 :
        y_pred_4[i] = 1 
        
    if data.iloc[i]['scores'] >= 5 :
        y_pred_5[i] = 1
        
    if data.iloc[i]['scores'] >= 6 :
        y_pred_6[i] = 1   

    if data.iloc[i]['scores'] >= 7 :
        y_pred_7[i] = 1  
         
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
                if (data_temp.iloc[i]['scores'] == scoreList[0] or data_temp.iloc[i]['scores'] == scoreList[1]) and data_temp.iloc[i]['Hanta-pos'] == 1:
                    probability += 1
        else:
            score = scoreList_temp.count(scoreList[0])
            for i in range(len(data_temp)):
                if data_temp.iloc[i]['scores'] == scoreList[0] and data_temp.iloc[i]['Hanta-pos'] == 1:
                    probability += 1
        probabilityList.append(np.round((probability / score) * 100))
    probabilityList.sort()
    confidence_lower_mean = probabilityList[int(0.025 * len(probabilityList))]
    confidence_upper_mean = probabilityList[int(0.975 * len(probabilityList))]
    print('Lower CI: {}'.format(confidence_lower_mean))
    print('Higher CI: {}'.format(confidence_upper_mean))

#%%

class_names = ['Healthy', 'Infected']
def plot_confusion_matrix_m(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    matplotlib.rcParams.update({'font.size': 25})

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 25)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes, rotation=90)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label\n')
    plt.xlabel('\nPredicted label')

#%% 
def CalculateParameters(cm):
    total = sum(sum(cm))
    accuracy = (cm[0,0]+cm[1,1])/total
    sensitivity = cm[1,1]/(cm[1,1]+cm[1,0])
    specificity = cm[0,0]/(cm[0,0]+cm[0,1])
    print('accuracy: {}'.format(np.round(accuracy, 4)*100))
    print('Sensitivity: {}'.format(np.round(sensitivity, 4)*100))
    print('specificity: {}'.format(np.round(specificity, 4)*100))
    print('#######################')
    
#%% Generate the probabilities and the confidence intervals

probability_1 = 0
probability_2 = 0
probability_3 = 0
probability_4 = 0
probability_5 = 0
probability_6 = 0
probability_7 = 0

sore_0_1 = scoreList.count(1) + scoreList.count(0)
sore_2 = scoreList.count(2)
sore_3 = scoreList.count(3)
sore_4 = scoreList.count(4)
sore_5 = scoreList.count(5)
sore_6 = scoreList.count(6)
sore_7 = scoreList.count(7)

for i in range(len(data)):
    if (data.iloc[i]['scores'] == 1 or data.iloc[i]['scores'] == 0) and data.iloc[i]['Hanta-pos'] == 1:
       probability_1 += 1
    elif data.iloc[i]['scores'] == 2 and data.iloc[i]['Hanta-pos'] == 1:
       probability_2 += 1       
    elif data.iloc[i]['scores'] == 3 and data.iloc[i]['Hanta-pos'] == 1:
       probability_3 += 1
    elif data.iloc[i]['scores'] == 4 and data.iloc[i]['Hanta-pos'] == 1:
       probability_4 += 1
    elif data.iloc[i]['scores'] == 5 and data.iloc[i]['Hanta-pos'] == 1:
       probability_5 += 1
    elif data.iloc[i]['scores'] == 6 and data.iloc[i]['Hanta-pos'] == 1:
       probability_6 += 1
    elif data.iloc[i]['scores'] == 7 and data.iloc[i]['Hanta-pos'] == 1:
       probability_7 += 1

if not sore_0_1 == 0:
    probability_1 = np.round((probability_1 / sore_0_1) * 100)
    print('Probability of HantaVirus with 0 or 1 Risk score:  '  + str(probability_1))
    CalculateConfidenceInterval([0,1])
    
    cm = confusion_matrix(y, y_pred_0)
    plt.figure()
    plot_confusion_matrix_m(cm, classes=class_names, title = 'Score >= 0')
    plt.show()
    CalculateParameters(cm)

    cm = confusion_matrix(y, y_pred_1)
    plt.figure()
    plot_confusion_matrix_m(cm, classes=class_names, title = 'Score >= 1')
    plt.show()
    CalculateParameters(cm)    
if not sore_2 == 0:
    probability_2 = np.round((probability_2 / sore_2) * 100)
    print('Probability of HantaVirus with 2 Risk score:  '  + str(probability_2))
    CalculateConfidenceInterval([2])
    cm = confusion_matrix(y, y_pred_2)
    plt.figure()
    plot_confusion_matrix_m(cm, classes=class_names, title = 'Score >= 2')
    plt.show()
    CalculateParameters(cm)
if not sore_3 == 0:
    probability_3 = np.round((probability_3 / sore_3) * 100)
    print('Probability of HantaVirus with 3 Risk score:  '  + str(probability_3))  
    CalculateConfidenceInterval([3])
    cm = confusion_matrix(y, y_pred_3)
    plt.figure()
    plot_confusion_matrix_m(cm, classes=class_names, title = 'Score >= 3')
    plt.show()
    CalculateParameters(cm)
if not sore_4 == 0:
    probability_4 = np.round((probability_4 / sore_4) * 100)
    print('Probability of HantaVirus with 4 Risk score:  '  + str(probability_4))    
    CalculateConfidenceInterval([4])    
    cm = confusion_matrix(y, y_pred_4)
    plt.figure()
    plot_confusion_matrix_m(cm, classes=class_names, title = 'Score >= 4')
    plt.show()
    CalculateParameters(cm)
if not sore_5 == 0:
    probability_5 = np.round((probability_5 / sore_5) * 100)
    print('Probability of HantaVirus with 5 Risk score:  '  + str(probability_5))
    CalculateConfidenceInterval([5])
    cm = confusion_matrix(y, y_pred_5)
    plt.figure()
    plot_confusion_matrix_m(cm, classes=class_names, title = 'Score >= 5')
    plt.show()
    CalculateParameters(cm)
if not sore_6 == 0:
    probability_6 = np.round((probability_6 / sore_6) * 100)
    print('Probability of HantaVirus with 6 Risk score:  '  + str(probability_6))
    CalculateConfidenceInterval([6])
    cm = confusion_matrix(y, y_pred_6)
    plt.figure()
    plot_confusion_matrix_m(cm, classes=class_names, title = 'Score >= 6')
    plt.show()
    CalculateParameters(cm)
if not sore_7 == 0:
    probability_7 = np.round((probability_7 / sore_7) * 100)
    print('Probability of HantaVirus with 7 Risk score:  '  + str(probability_7))
    CalculateConfidenceInterval([7])
    cm = confusion_matrix(y, y_pred_7)
    plt.figure()
    plot_confusion_matrix_m(cm, classes=class_names, title = 'Score >= 7')
    plt.show()
    CalculateParameters(cm)

#%%

results = {}
results['tpr'] = {}
results['fpr'] = {}
results['auc'] = {}

randomState = 23
matplotlib.rcParams.update({'font.size': 20})

fpr_temp = np.linspace(0, 1, 100)

for i in range(5):
    classifier = LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,  random_state = randomState)
    
    classifier.fit(X_train, y_train)
    y_pred_proba = classifier.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    results['tpr'][i] = np.interp(fpr_temp, fpr, tpr)
    results['tpr'][i][0] = 0.0
    results['fpr'][i] = fpr_temp
    results['auc'][i] = auc
    randomState += 10

# Confidence Interval For For one fold
random_state = 40    
classifier = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,  random_state = randomState)
classifier.fit(X_train, y_train)
y_pred_proba = classifier.predict_proba(X_test)[::,1]
fpr_single, tpr_single, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc_single = metrics.roc_auc_score(y_test, y_pred_proba)

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

#viz = plot_roc_curve(classifier, X_test, y_test, name='ROC fold {}'.format(i),alpha=0.3, lw=1, ax=ax)

aucs_temp = np.array(list(results['auc'].values()))
aucs_temp.sort()
confidence_lower_mean = aucs_temp[int(0.025 * len(aucs_temp))]
confidence_upper_mean = aucs_temp[int(0.975 * len(aucs_temp))]

fig, ax = plt.subplots()   
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', alpha=.8)

mean_tpr = [(a+ b+ c+ d+ e) / 5 for a, b, c, d, e in zip(list(results['tpr'][0]), list(results['tpr'][1]), list(results['tpr'][2]), list(results['tpr'][3]),
                                                         list(results['tpr'][4]))]
mean_fpr = [(a+ b+ c+ d+ e) / 5 for a, b, c, d, e in zip(list(results['fpr'][0]), list(results['fpr'][1]), list(results['fpr'][2]), list(results['fpr'][3]), 
                                                         list(results['fpr'][4]))]

mean_auc = metrics.auc(mean_fpr, mean_tpr)
ax.plot(fpr_single, tpr_single, color = 'b', label = r'AUC = ' + str(round(auc_single,2)) + '(' + str(round(confidence_lower_single, 2)) + ' - ' + str(round(confidence_upper_single, 2)) + ')',
        lw = 2, alpha = .8)
ax.plot(mean_fpr, mean_tpr, color = 'r', label = 'Mean AUC = ' + str(round(mean_auc,2)) + '(' + str(round(confidence_lower_mean, 2)) + ' - ' + str(round(confidence_upper_mean, 2)) + ')',
        lw =4, alpha = .8)

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
ax.legend(loc="lower right", fontsize='large')
ax.set_xlabel('1 - Specificity', fontsize='large', fontweight='bold')
ax.set_ylabel('Sensitivity', fontsize='large', fontweight='bold')
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
plt.show()

# %%

fig, ax = plt.subplots()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,  random_state = 73)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

viz = plot_roc_curve(classifier, X_test, y_test, name='ROC', lw=1, ax=ax)
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)
plt.title('Receiver Operating Characteristic')
plt.legend(loc = 'lower right')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


disp = plot_confusion_matrix(classifier, X_test, y_test,
                             display_labels = ['Healthy', 'Infected'],
                             cmap = plt.cm.Blues,
                             normalize = None)

###########################################################################







    