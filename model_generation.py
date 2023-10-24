import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import label_binarize
from krippendorff import krippendorff

from AutoSpearmanPriority import AutoSpearmanReductive


def evaluate(y_test, to_evaluate):
    f1 = f1_score(y_test, to_evaluate, average='weighted', labels=np.unique(to_evaluate))
    mcc = matthews_corrcoef(list(y_test), to_evaluate)
    labels = [0,1,2]
    try:
        auc_roc = roc_auc_score(label_binarize(list(y_test),classes=labels), label_binarize(to_evaluate,classes=labels), average='weighted', multi_class='ovr', labels=labels)
    except:
        auc_roc = roc_auc_score(list(y_test), to_evaluate, average='weighted', multi_class='ovr',
                                labels=np.unique(to_evaluate))
    alpha_ordinal = krippendorff.alpha(reliability_data=[to_evaluate, list(y_test)],level_of_measurement='ordinal')
    try:
        to_evaluate = to_evaluate.tolist()
    except:
        pass
    return {
            'alpha': alpha_ordinal,
            'auc_roc': auc_roc,
            # 'y_test': list(y_test),
            # 'y_predict': to_evaluate,
            'f1': f1,
            'mcc': mcc,
        }

df = pd.read_csv('./dataset/mlcq_blob.csv')
features_col = 'cbo,cboModified,fanin,fanout,wmc,dit,noc,rfc,lcom,lcom*,tcc,lcc,totalMethodsQty,staticMethodsQty,publicMethodsQty,privateMethodsQty,protectedMethodsQty,defaultMethodsQty,visibleMethodsQty,abstractMethodsQty,finalMethodsQty,synchronizedMethodsQty,totalFieldsQty,staticFieldsQty,publicFieldsQty,privateFieldsQty,protectedFieldsQty,defaultFieldsQty,finalFieldsQty,synchronizedFieldsQty,nosi,loc,returnQty,loopQty,comparisonsQty,tryCatchQty,parenthesizedExpsQty,stringLiteralsQty,numbersQty,assignmentsQty,mathOperationsQty,variablesQty,maxNestedBlocksQty,anonymousClassesQty,innerClassesQty,lambdasQty,uniqueWordsQty,modifiers,logStatementsQty,churn,files,NR,NBF,nc,ncom,avg_cs,dsc,ce,exp,own,is_controller,is_procedural,is_test,is_util'.split(',')
severity_col = 'severity'

X = df[features_col].replace(np.nan,0).replace(True,1).replace(False,0)
X = AutoSpearmanReductive(X, 'blob')
y = df[severity_col].replace("none",0).replace("critical",2).replace("major",1).replace("minor",1)
LOOCV = LeaveOneOut()
y_val_arr = []
y_predict_arr = []
i = 0
lenY=len(y)
for train_index, val_index in LOOCV.split(X, y):
    i += 1
    print(f'{i}/{lenY}')

    x_train_, x_val = X.iloc[train_index], X.iloc[val_index]
    y_train_, y_val = y[train_index], y[val_index]

    y_val = int(y_val.values[0])
    clf = RandomForestClassifier(random_state=88)
    clf.fit(x_train_, y_train_)

    y_pred = clf.predict(x_val)[0]

    y_val_arr.append(y_val)
    y_predict_arr.append(int(y_pred))
    try:
        print(evaluate(y_val_arr, y_predict_arr))
    except:
        pass
performance = evaluate(y_val_arr, y_predict_arr)
print(performance)

model = RandomForestClassifier(random_state=88).fit(X,y)
with open('./model.pkl','wb') as f1:
    pickle.dump(model, f1)