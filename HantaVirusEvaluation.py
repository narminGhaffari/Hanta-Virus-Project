
# Written by Narmin Ghaffari Laleh - Nov 2020
# narminghaffari23@gmail.com

###############################################################################
import pandas as pd 
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from sklearn.metrics import plot_roc_curve
from sklearn.metrics import auc
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_auc_score

###############################################################################

# Load the Excel File and seprate the dependent and independent variables.
data = pd.read_excel(r'D:\Justus Project\ROC Hantavsnon.xlsx')
    
data_temp = data[['Fever', 'Headache/visual disturbance', 'Female Sex', 'Thrombo <150']]        
imp = SimpleImputer(strategy="most_frequent")
x = imp.fit_transform(data_temp)

y = data[['Hantavirus']]
y = imp.fit_transform(y)

###############################################################################

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,  random_state = 23)

logit_model = sm.Logit(y_train, X_train)
result = logit_model.fit()
print(result.summary2())

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
classifier.coef_

###############################################################################

points = np.abs(np.round(classifier.coef_))
points = points[0]
scoreList = []

for i in range(len(data)):
    score = 0
    if data.iloc[i]['Fever'] == 1:
        score += points[0]
    if data.iloc[i]['Headache/visual disturbance'] == 1:
        score += points[1]
    if data.iloc[i]['LDH >300'] == 1:
        score += points[2]
    if data.iloc[i]['Thrombo <150'] == 1:
        score += points[3]
    scoreList.append(score)

###############################################################################

data['scores'] = scoreList
probability_1 = 0
probability_2 = 0
probability_3 = 0
probability_4 = 0

sore_0_1 = scoreList.count(1) + scoreList.count(0)
sore_2 = scoreList.count(2)
sore_3 = scoreList.count(3)
sore_4 = scoreList.count(4)

for i in range(len(data)):
    if (data.iloc[i]['scores'] == 1 or data.iloc[i]['scores'] == 0) and data.iloc[i]['Hantavirus'] == 1:
       probability_1 += 1
    if data.iloc[i]['scores'] == 2 and data.iloc[i]['Hantavirus'] == 1:
       probability_2 += 1       
    if data.iloc[i]['scores'] == 3 and data.iloc[i]['Hantavirus'] == 1:
       probability_3 += 1
    if data.iloc[i]['scores'] == 4 and data.iloc[i]['Hantavirus'] == 1:
       probability_4 += 1


probability_1 = np.round((probability_1 / sore_0_1) * 100)
probability_2 = np.round((probability_2 / sore_2) * 100)
probability_3 = np.round((probability_3 / sore_3) * 100)
probability_4 = np.round((probability_4 / sore_4) * 100)


print('Probability of HantaVirus with 0 or 1 Risk score:  '  + str(probability_1))
print('Probability of HantaVirus with 2 Risk score:  '  + str(probability_2))
print('Probability of HantaVirus with 3 Risk score:  '  + str(probability_3))
print('Probability of HantaVirus with 4 or more Risk score:  '  + str(probability_4))


###############################################################################

data_temp = data[['Fever','Headache/visual disturbance', 'Female Sex', 'Thrombo <150']]  
imp = SimpleImputer(strategy="most_frequent")
x = imp.fit_transform(data_temp)


y = data[['Hantavirus']]
y = imp.fit_transform(y)

aucs = []
tprs = []

randomState = 23
fig, ax = plt.subplots()
mean_fpr = np.linspace(0, 1, 100)

#classifier = svm.SVC(kernel='linear', probability=True,random_state=randomState)
#classifier =DecisionTreeClassifier()
#classifier = KNeighborsClassifier()
classifier = LogisticRegression()

for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,  random_state = randomState)
    
    classifier.fit(X_train, y_train)
    viz = plot_roc_curve(classifier, X_test, y_test, name='ROC fold {}'.format(i),alpha=0.3, lw=1, ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    randomState += 10

# Confidence Interval For For one fold

randomState = 73
classifier = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,  random_state = randomState)

auc_values = []
nsamples = 1000
for b in range(nsamples):
    idx = np.random.randint(X_train.shape[0], size=X_train.shape[0])
    classifier.fit(X_train[idx], y_train[idx])
    
    pred = classifier.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test.ravel(), pred.ravel())
    auc_values.append(roc_auc)
    
auc_values = np.array(auc_values)
auc_values.sort()

confidence_lower = auc_values[int(0.05 * len(auc_values))]
confidence_upper = auc_values[int(0.95 * len(auc_values))]

viz = plot_roc_curve(classifier, X_test, y_test, name='ROC fold {}'.format(i),alpha=0.3, lw=1, ax=ax)

fig, ax = plt.subplots()   
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
 
ax.plot(viz.fpr, viz.tpr, color = 'b', label = r'AUC = ' + str(round(viz.roc_auc,2)) + '(' + str(round(confidence_lower, 2)) + ' - ' + str(round(confidence_upper, 2)) + ')', lw = 2, alpha = .8)

# Confidence Interval For Cross Validation
aucs = np.array(aucs)
aucs.sort()
confidence_lower_CrossVal = aucs[int(0.05 * len(aucs))]
confidence_upper_CrossVal = aucs[int(0.95 * len(aucs))]
ax.plot(mean_fpr, mean_tpr, color = 'r', label = 'Mean AUC = ' + str(round(mean_auc,2)) + '(' + str(round(confidence_lower_CrossVal, 2)) + ' - ' + str(round(confidence_upper_CrossVal, 2)) + ')', lw = 2, alpha = .8)


ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
ax.legend(loc="lower right", fontsize='large')
ax.set_xlabel('1 - Specificity', fontsize='large', fontweight='bold')
ax.set_ylabel('Sensitivity', fontsize='large', fontweight='bold')
plt.show()


###############################################################################

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







    