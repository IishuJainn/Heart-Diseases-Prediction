# PROJECT FOR PREDICTING WHETHER A PERSON IS SUFFERING FROM DISEASES OR NOT

# Regular EDA(exploratory and data analysis) and plotting libraries
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
# Models from Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
# Models evaluation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve

# DATA EXPLORATION
# JUST BECOMING FAMILIAR WITH THE DATA
# 1.WHAT QUESTION WE NEED TO SOLVE
# 2.WHAT TYPE DATA WE HAVE AND HOW TO TREAT DIFFERNT TYPE OF DATA
# 3.WHAT DATA IS MISSING
# 4.WHERE ARE THE OUTLIERS AND WHY TO CARE ABOUT THEM
# 5.HOW CAN U ADD ,CHANGE OR REMOVE FEATURES TO GET MORE OUT OF OUR DATA
df=pd.read_csv("heart_disease.csv")
print(df["target"].value_counts())
df["target"].value_counts().plot(kind="bar",color=["salmon","lightblue"])
print(df.info())
print(df.isna().sum())
print(df.describe())

# NOW WE WILL JUST TRY TO FIND ANY PATTERN IN THE DATA OR WE CAN SAY WE WILL FIND THE RELATION OF DIFFERENT COLUMN WITH TARGET
print(df["sex"].value_counts()) # A LOT MORE MALE
print(pd.crosstab(df.target,df.sex)) # A LOT MORE WOMEN IS HAVING HEART DISEASES
pd.crosstab(df.target,df.sex).plot(kind="bar",figsize=(10,6),color=["salmon","lightblue"])
plt.title("Heart Diseases Frequency For Sex")
plt.xlabel("0 =  NO Diseases, 1 = Diseases")
plt.ylabel("Amount")
plt.legend(["Female","Male"])
plt.xticks(rotation=0) # FOR MAKING THE LABEL ON XAXIS AS VERTICAL
plt.show(block=True)

# NOW WE WILL COMPARE CERTAIN OTHER COLUMN
print(df["thalach"].value_counts()) # a large range of of value by the way thalach means max heart beat
# now we will compare age vs max heart beat(thalach) with the target
plt.figure(figsize=(10,6))
# Scatter with positive example
plt.scatter(df.age[df.target==1],df.thalach[df.target==1], c ="salmon")
# Scatter with negative example
plt.scatter(df.age[df.target==0],df.thalach[df.target==0], c ="lightblue")
plt.title("Heart Diseases in Function of Age and Heart Rate")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.legend(["Diseases",'No Diseases']);
# Checking the distribution of the age column with a histogram
df.age.plot.hist()

# Heart pain vs diseases
print(pd.crosstab(df.cp,df.target))

# Make the cross tab more visual
pd.crosstab(df.cp, df.target).plot(kind="bar",figsize=(10,6),color=["salmon","lightblue"])
plt.title("Heart Diseases Frequency per chest pain type")
plt.xlabel("Chest Pain Type")
plt.ylabel("Amount")
plt.legend(["NO Diseases ","Diseases"])
plt.xticks(rotation=0)
plt.show(block=True)
print(df.corr())
# Lets make it little prettier by using heat map
corr_matrix=df.corr()
fig, ax=plt.subplots(figsize=(15, 10))

# NOW WE WILL START THE MODEL TRAINING
X =df.drop("target",axis=1)
y=df.target
np.random.seed(42)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

# The data is splitted now we will try 3  differnt models
# 1.logistic regression
# 2.k- nearest neighbour classifier
# 3.random forest classifier
models= {"Logistic Regression": LogisticRegression(),
        "KNN": KNeighborsClassifier(),
        "Randon Forest": RandomForestClassifier()}
def fit_and_score( models, X_train,X_test,y_train,y_test):
    np.random.seed(42)
    model_scores={}
    for name, model in models.items():
        model.fit(X_train, y_train)
        model_scores[name]=model.score(X_test, y_test)
    return model_scores
model_scores=fit_and_score(models=models,X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)
print(model_scores)
model_compare=pd.DataFrame(model_scores, index=['accuracy'])

model_compare.T.plot.bar();
plt.show(block=True)

# lets look at the following
# 1.hyperparameter tuning
# 2.features importance
# 3.confusion_matrix
# 4.precision_score
# 5.recall_score
# 6.f1_score
# 7.classification_report
# 8.ROC plot_roc_curvearea under the curve

# HYPERPARAMETER TUNING (BY HAND)
# FOR KNN
train_scores =[]
test_scores =[]
neighbours = range(1, 21)
knn=KNeighborsClassifier()
for i in neighbours:
    knn.set_params(n_neighbors=i)
    knn.fit(X_train,y_train)
    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))

print(train_scores)
print(test_scores)
plt.plot(neighbours,train_scores, label="Train score")
plt.plot(neighbours,test_scores,label="Test score")
plt.xlabel("NO OF NEIGHBOURS")
plt.ylabel("Model score")
plt.legend()
plt.show(block=True)  # we can see the best score is at 11 neighbours which is about 75% which is very low so
# it gave an end for knn
# INSTEAD OF DOING HYPER PARAMETER TUNING WITH HAND WE CAN DIRECTLY USE A FUNCTION RANDOMIZED SEARCH CV THAT
# WILL GAVE US THE BEST VALUE FOR ANY MODEL
# WE WILL NOW TUNE LOGISTIC REGRESSION AND  RANDOM FOREST BY RANDOMIZED SEARCH CV

# CREATE A HYPER PARAMETER GRID FOR LOGISTIC REGRESSION
log_reg_grid={"C": np.logspace(-4,4,20),
              "solver": ["liblinear"]}

# WE JUST USED TWO FEATURES WE CAN EVEN USE MORE FOR BEST RESULT
# CREATE A HYPER PARAMETER GRID FOR RandomForestClassifier
rf_grid={"n_estimators": np.arange(10,1000,50),
         "max_depth": [None, 3 , 5 , 10],
         "min_samples_split": np.arange(2, 20, 2),
         "min_samples_leaf":  np.arange(1, 20, 2)}

# NOW WE GET THE GRID SET NOW LETS FIND THE BEST ONE BY RANDOMIZED SEARCH CV
np.random.seed(42)
rs_log_reg=RandomizedSearchCV(LogisticRegression(),
                              param_distributions=log_reg_grid,
                              cv=5,
                              n_iter=20,
                              verbose=True)
rs_log_reg.fit(X_train,y_train)
print(rs_log_reg.best_params_)  #{'solver': 'liblinear', 'C': 0.23357214690901212} BEST PARAMETERS
print(rs_log_reg.score(X_test,y_test))
# 0.8852459016393442 WE JUST MATCH OUR ORIGINAL SCORE WE CAN INCREASE IT BY USING MORE FEATURES IN RANDOMIZED SEARCH CV
# # NOW WE WILL CHK FOR RANDOM FOREST

np.random.seed(42)
rs_rf=RandomizedSearchCV(RandomForestClassifier(),
                        param_distributions=rf_grid,
                        cv=5,
                        n_iter=20,
                        verbose=True)
rs_rf.fit(X_train,y_train)
print(rs_rf.best_params_)  # {'n_estimators': 210, 'min_samples_split': 4, 'min_samples_leaf': 19, 'max_depth': 3} BEST PARAMETERS
print(rs_rf.score(X_test,y_test))  # .8688524590163934 WE JUST beated OUR ORIGINAL SCORE WE CAN INCREASE IT BY USING MORE FEATURES IN RANDAMIZED SEARCH CV

# NOW WE WILL USE THE GRID SEARCH CV THAT AUTOMATICALY TAKES THE IMPORTANT PARAMETERS FOR A MODEL
logs_reg_grid={"C": np.logspace(-4,4,30),
              "solver":["liblinear"]}
gs_log_reg=GridSearchCV(LogisticRegression(),
                              param_grid=logs_reg_grid,
                              cv=5,
                              verbose=True)
gs_log_reg.fit(X_train,y_train)
print(gs_log_reg.best_params_)  #{'solver': 'liblinear', 'C': 0.23357214690901212} BEST PARAMETERS
print(gs_log_reg.score(X_test,y_test))  #.868852 still the same

# Evaluating our tuned machine learning classifier, beyond accuracy
# roc curve
# confusion_matrix()
# classification_report()
# precision_score()
# recall
# f1_score()
# ALL OF THESE ARE ALREADY IMPORTED AT THE TOP

y_preds=gs_log_reg.predict(X_test)
plot_roc_curve(gs_log_reg,X_test, y_test)
print(confusion_matrix(y_preds,y_test))

# Lets make the confusion matrix look preetier
sns.set(font_scale=1.5)
def plot_confusion_mat(y_test,y_preds):
    fig, ax=plt.subplots(figsize=(3, 3))
    ax=sns.heatmap(confusion_matrix(y_test,y_preds),
                   annot=True,
                   cbar=False)
    plt.xlabel("TRUE LABEL")
    plt.ylabel("Predicted label")

plot_confusion_mat(y_test,y_preds)
plt.show(block=True)
print(classification_report(y_test,y_preds))

# THIS DATA IS ONLY FOR A SINGLE SPLIT NOW WE WILL USE CROSS VALIDATION
print(gs_log_reg.best_params_) # 'C': 0.20433597178569418, 'solver': 'liblinear'
clf=LogisticRegression(C= 0.20433597178569418, solver="liblinear")
cv_acc=cross_val_score(clf,
                       X,
                       y,
                       cv=5,
                       scoring="accuracy"
                       )
cv_acc=np.mean(cv_acc)
print(cv_acc)
# we can similarly find the precesion,recall and f1 score just changing the scoring parametet
cv_pre=cross_val_score(clf,
                       X,
                       y,
                       cv=5,
                       scoring="precision"
                       )
cv_pre=np.mean(cv_pre)
print(cv_pre)
cv_rec=cross_val_score(clf,
                       X,
                       y,
                       cv=5,
                       scoring="recall"
                       )
cv_rec=np.mean(cv_rec)
print(cv_rec)
cv_f1=cross_val_score(clf,
                       X,
                       y,
                       cv=5,
                       scoring="f1"
                       )
cv_f1=np.mean(cv_f1)
print(cv_f1)
cv_metrices=pd.DataFrame({"Accuracy": cv_acc,
                          "precision": cv_pre,
                          "Recall": cv_rec,
                          "f1 score": cv_f1},
                          index=[0])
cv_metrices.T.plot.bar(title="Cross-Validated classification metrices",
                       legend=False)
plt.show(block=True)

# NOW ARE MODEL IS READY THE LAST THING WE NEED TO DO IS TO FIND HOW EACH COLUMN CONTRIBUTED TO THE TARGET FOR OUR MODEL
clf = LogisticRegression(C= 0.20433597178569418, solver="liblinear")
clf.fit(X_train,y_train)
print(df.head())
print(clf.coef_)
feature_dict=dict(zip(df.columns, list(clf.coef_[0])))
print(feature_dict)
feature_df=pd.DataFrame(feature_dict,index=[0])
feature_df.T.plot.bar(title="Featue Importance", legend=False);
plt.show(block=True)

