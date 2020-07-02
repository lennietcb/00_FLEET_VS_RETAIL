from sklearn.cross_validation import train_test_split 
from sklearn.preprocessing import StandardScaler 
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

#read prepared data
dataset_prep = pd.read_pickle('archive/prepared_data.pickle')
print(dataset_prep.shape)
dataset_prep.describe().transpose()

#remove some low ranking columns and define x and y
imp_cols=["class", "age_acpt",	"proceeds",	"age_sale", "cond_code", "days_on_lot", "Ask",	"Take", "Proceed_Over_Ask",	"Proceed_Over_Take",	"vehicle_grp_TRACTOR", "age_sale_grp_079 - 084", "age_sale_grp_085 - 090", "age_sale_grp_091 - 096", "age_sale_grp_103 - 108", "age_sale_grp_109 - 120", "age_sale_grp_115 - 120", "age_sale_grp_121 - 132", "age_sale_grp_127 - 132"] 
X = dataset_prep [imp_cols]
y = dataset_prep['y']

#Standard Scaler
xtrain, xtest, ytrain, ytest = train_test_split( x, y, test_size = 0.25, random_state = 0)

sc_x = StandardScaler() 
xtrain = sc_x.fit_transform(xtrain)  
xtest = sc_x.transform(xtest) 
  
print (xtrain[0:10, :]) 

#Implement Model I
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())

print(result.conf_int())

#Remove p value columns
imp_cols2 =["class", "age_acpt",	"proceeds",	"age_sale", "cond_code", "days_on_lot", "Ask",	"Take", "Proceed_Over_Ask",	"Proceed_Over_Take",	"vehicle_grp_TRACTOR", "age_sale_grp_079 - 084", "age_sale_grp_085 - 090", "age_sale_grp_091 - 096", "age_sale_grp_103 - 108", "age_sale_grp_115 - 120", "age_sale_grp_127 - 132"]
X = os_data_X [imp_cols2]

#Implement Model II
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())

#Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, y_train)

#Predictions and Accuracy
predictions = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

#Cross-validation
from sklearn.model_selection import cross_val_score

cross_val_score(logreg,X,y, cv=10, scoring='accuracy').mean()

#Adding Probabilities
predictions_prob = logreg.predict_proba(X_test)[:,1]

df_result = X_test
df_result["y_test"] = y_test
df_result["predictions"] = predictions
df_result["probability"] = predictions_prob
df_result["match"] = df_result.predictions == df_result.y_test

#Count Probability max0.5
df_result_1 = df_result[df_result["probability"] > 0.5]
frequency_df_result_1 = df_result_1['predictions'].value_counts()
frequency_df_result_1

#Count Match
frequency_pred = df_result['match'].value_counts()
frequency_pred

#Confusion Matrix
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

#Precision, Recall, F-measure and Support
# Precision ratio tp/(tp+fp) --- tp:true positives, fp:false positives ---
# Recall ratio tp/(tp+fn) --- tp:true positives, fp:false positives ---  
# F1: weighted mean of the precision, close to 1 is better
# Support: number of occurrences of each class in y_test
# Precision: ability of the classifier to not label a sample as positive if it is negative
# Recall: the ability of the classifier to find all the positive samples

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#ROC Curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()