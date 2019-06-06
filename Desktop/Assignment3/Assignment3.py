#!/usr/bin/env python
# coding: utf-8

# In[15]:


from sklearn import datasets		
from sklearn import svm    			# To fit the svm classifier
import numpy as np
import matplotlib.pyplot as plt            # To visuvalizing the data
import pandas as pd
from sklearn.model_selection import train_test_split


# In[43]:


data = pd.read_csv("sonar.all-data", header = None)
data_names = pd.read_csv("sonar.rocks")
dataset = pd.DataFrame(data)



# In[47]:


X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, 60].values


# In[53]:


train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[102]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


# In[103]:



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()  
train_x = sc.fit_transform(train_x)  
test_x = sc.transform(test_x)  


# In[104]:


from sklearn.ensemble import RandomForestClassifier

regressor = RandomForestClassifier(n_estimators=20, random_state=0)  
regressor.fit(train_x, train_y)  
y_pred = regressor.predict(test_x)  


# In[105]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(test_y,y_pred))  
print(classification_report(test_y,y_pred))  
print(accuracy_score(test_y, y_pred))  


# In[ ]:





# In[ ]:


#########################################   Grid Search for parameter tuning  #########################################


# In[106]:


param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}


# In[107]:


CV_rfc = GridSearchCV(estimator=regressor, param_grid=param_grid, cv= 5)
CV_rfc.fit(train_x, train_y)


# In[108]:


CV_rfc.best_params_


# In[109]:


rfc1=RandomForestClassifier(random_state=0, max_features='auto', n_estimators= 500, max_depth=7, criterion='entropy')


# In[110]:


rfc1.fit(train_x, train_y)


# In[111]:


pred_rf_tuned=rfc1.predict(test_x)


# In[112]:


print("Accuracy for Random Forest on CV data: ",accuracy_score(test_y,pred_rf_tuned))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


###################################################### Decision Tree ###################################################


# In[123]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


# In[124]:


clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(train_x,train_y)

#Predict the response for test dataset
y_pred_decisiontree = clf.predict(test_x)


# In[125]:


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(test_y, y_pred_decisiontree))


# In[126]:


# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(train_x,train_y)

#Predict the response for test dataset
y_pred_tuned = clf.predict(test_x)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(test_y, y_pred_tuned))


# In[ ]:





# In[127]:


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
#making the instance
model= DecisionTreeClassifier(random_state=1234)
#Hyper Parameters Set
params = {'max_features': ['auto', 'sqrt', 'log2'],
          'min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15], 
          'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11],
          'random_state':[123]}
#Making models with hyper parameters sets
model1 = GridSearchCV(model, param_grid=params, n_jobs=-1)
#Learning
model1.fit(train_x,train_y)
#The best hyper parameters set
print("Best Hyper Parameters:",model1.best_params_)
#Prediction
prediction=model1.predict(test_x)
#importing the metrics module
from sklearn import metrics
#evaluation(Accuracy)
print("Accuracy:",metrics.accuracy_score(prediction,test_y))
#evaluation(Confusion Metrix)
print("Confusion Metrix:\n",metrics.confusion_matrix(prediction,test_y))


# In[ ]:





# In[ ]:





# In[ ]:


###########################################################  SVC  ######################################################


# In[129]:


from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear')  
svclassifier.fit(train_x,train_y)


# In[130]:


y_pred_svc = svclassifier.predict(test_x) 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(test_y,y_pred_svc))  
print(classification_report(test_y,y_pred_svc))
print("Accuracy:",metrics.accuracy_score(test_y, y_pred_svc))


# In[ ]:


#############################################Grid Search tuning for SVC##################################################


# In[115]:


tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                    {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                    {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
                   ]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf1 = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf1.fit(train_x, train_y)

    print("Best parameters set found on development set:")
    print()
    print(clf1.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf1.cv_results_['mean_test_score']
    stds = clf1.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf1.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()


# In[116]:


clf1.best_params_


# In[119]:


my_svm = SVC(C=10, kernel="rbf", gamma = 0.01)
my_svm.fit(train_x, train_y)


# In[121]:


y_tuned_svc = my_svm.predict(test_x) 


# In[122]:


print(confusion_matrix(test_y,y_tuned_svc))  
print(classification_report(test_y,y_tuned_svc))
print("Accuracy:",metrics.accuracy_score(test_y, y_tuned_svc))



