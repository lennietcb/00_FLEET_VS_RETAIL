from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

#Feature Selection - Recursive Feature Elimination
logreg = LogisticRegression()
rfe = RFE(logreg, 20)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())

#print results
print(rfe.support_)
print(rfe.ranking_)

#Remove low ranking variables
imp_cols=["class", "age_acpt",	"proceeds",	"age_sale", "cond_code", "days_on_lot", "Ask",	"Take", "Proceed_Over_Ask",	"Proceed_Over_Take",	"vehicle_grp_TRACTOR", "age_sale_grp_079 - 084", "age_sale_grp_085 - 090", "age_sale_grp_091 - 096", "age_sale_grp_103 - 108", "age_sale_grp_109 - 120", "age_sale_grp_115 - 120", "age_sale_grp_121 - 132", "age_sale_grp_127 - 132"] 
X = os_data_X [imp_cols]
y = os_data_y['y']