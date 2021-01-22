
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


from sklearn.metrics import confusion_matrix
import itertools

###############################################################################

# Load the Excel File and seprate the dependent and independent variables.
data = pd.read_excel(r'Path')
    
data_temp = data[['Fever', 'Headache', 'LDH >300', 'Thrombo <150', 'Hantavirus']] 
data_temp.dropna(inplace = True) 
 
x = data_temp[['Fever', 'Headache', 'LDH >300', 'Thrombo <150']]

y = data_temp[['Hantavirus']]

###############################################################################

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,  random_state = 23)

logit_model = sm.Logit(y_train, X_train)
result = logit_model.fit()
print(result.summary2())

classifier = LogisticRegression()
classifier.fit(x, y)
classifier.coef_

###############################################################################

points = np.abs(np.round(classifier.coef_))
points = points[0]
scoreList = []

for i in range(len(data)):
    score = 0
    if data.iloc[i]['Fever'] == 1:
        score += points[0]
    if data.iloc[i]['Headache'] == 1:
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
probability_5 = 0
probability_6 = 0

sore_0_1 = scoreList.count(1) + scoreList.count(0)
sore_2 = scoreList.count(2)
sore_3 = scoreList.count(3)
sore_4 = scoreList.count(4)
sore_5 = scoreList.count(5)
sore_6 = scoreList.count(6)

for i in range(len(data)):
    if (data.iloc[i]['scores'] == 1 or data.iloc[i]['scores'] == 0) and data.iloc[i]['Hantavirus'] == 1:
       probability_1 += 1
    if data.iloc[i]['scores'] == 2 and data.iloc[i]['Hantavirus'] == 1:
       probability_2 += 1       
    if data.iloc[i]['scores'] == 3 and data.iloc[i]['Hantavirus'] == 1:
       probability_3 += 1
    if data.iloc[i]['scores'] == 4 and data.iloc[i]['Hantavirus'] == 1:
       probability_4 += 1
    if data.iloc[i]['scores'] == 5 and data.iloc[i]['Hantavirus'] == 1:
       probability_5 += 1
    if data.iloc[i]['scores'] == 6 and data.iloc[i]['Hantavirus'] == 1:
       probability_6 += 1
       
       
probability_1 = np.round((probability_1 / sore_0_1) * 100)
probability_2 = np.round((probability_2 / sore_2) * 100)
probability_3 = np.round((probability_3 / sore_3) * 100)
probability_4 = np.round((probability_4 / sore_4) * 100)
probability_5 = np.round((probability_5 / sore_5) * 100)
probability_6 = np.round((probability_6 / sore_6) * 100)


print('Probability of HantaVirus with 0 or 1 Risk score:  '  + str(probability_1))
print('Probability of HantaVirus with 2 Risk score:  '  + str(probability_2))
print('Probability of HantaVirus with 3 Risk score:  '  + str(probability_3))
print('Probability of HantaVirus with 4 Risk score:  '  + str(probability_4))
print('Probability of HantaVirus with 5 Risk score:  '  + str(probability_5))
print('Probability of HantaVirus with 6 Risk score:  '  + str(probability_6))

###############################################################################


def plot_confusion_matrix_m(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

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
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


###############################################################################

# PLOT CONFUSION MATRIX PER PROBABILITY

y_pred_0 = [0] * len(y)
y_pred_1 = [0] * len(y)
y_pred_2 = [0] * len(y)
y_pred_3 = [0] * len(y)
y_pred_4 = [0] * len(y)
y_pred_5 = [0] * len(y)
y_pred_6 = [0] * len(y)

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
        
        
class_names = ['Healthy', 'Infected']

##########################################

cm = confusion_matrix(y, y_pred_0)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix_m(cm, classes=class_names,
                      title = 'Score >= 0')
plt.show()

total = sum(sum(cm))
accuracy = (cm[0,0]+cm[1,1])/total
sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
specificity = cm[1,1]/(cm[1,0]+cm[1,1])

##########################################

cm = confusion_matrix(y, y_pred_1)
plt.figure()
plot_confusion_matrix_m(cm, classes=class_names,
                      title = 'Score >= 1')
plt.show()
total = sum(sum(cm))
accuracy = (cm[0,0]+cm[1,1])/total
sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
specificity = cm[1,1]/(cm[1,0]+cm[1,1])

##########################################

cm = confusion_matrix(y, y_pred_2)
plt.figure()
plot_confusion_matrix_m(cm, classes=class_names,
                      title = 'Score >= 2')
plt.show()
total = sum(sum(cm))
accuracy = (cm[0,0]+cm[1,1])/total
sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
specificity = cm[1,1]/(cm[1,0]+cm[1,1])

##########################################

cm = confusion_matrix(y, y_pred_3)
plt.figure()
plot_confusion_matrix_m(cm, classes=class_names,
                      title = 'Score >= 3')
plt.show()
total = sum(sum(cm))
accuracy = (cm[0,0]+cm[1,1])/total
sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
specificity = cm[1,1]/(cm[1,0]+cm[1,1])

##########################################

cm = confusion_matrix(y, y_pred_4)
plt.figure()
plot_confusion_matrix_m(cm, classes=class_names,
                      title = 'Score >= 4')
plt.show()
total = sum(sum(cm))
accuracy = (cm[0,0]+cm[1,1])/total
sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
specificity = cm[1,1]/(cm[1,0]+cm[1,1])

##########################################

cm = confusion_matrix(y, y_pred_5)
plt.figure()
plot_confusion_matrix_m(cm, classes=class_names,
                      title = 'Score >= 5')
plt.show()
total = sum(sum(cm))
accuracy = (cm[0,0]+cm[1,1])/total
sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
specificity = cm[1,1]/(cm[1,0]+cm[1,1])
 
##########################################

cm = confusion_matrix(y, y_pred_6)
plt.figure()
plot_confusion_matrix_m(cm, classes=class_names,
                      title = 'Score >= 6')
plt.show()
total = sum(sum(cm))
accuracy = (cm[0,0]+cm[1,1])/total
sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
specificity = cm[1,1]/(cm[1,0]+cm[1,1])

###############################################################################
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

randomState = 23
classifier = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,  random_state = randomState)
y_test = y_test.to_numpy()

auc_values = []
nsamples = 1000
for b in range(nsamples):
    idx = np.random.randint(X_train.shape[0], size = X_train.shape[0])
    classifier.fit(X_train.iloc[idx], y_train.iloc[idx])
    
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







    