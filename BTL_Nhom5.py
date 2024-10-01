from scipy.stats import randint as sp_randint
import streamlit as st 
# add new
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import_optional_dependency("openpyxl")
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import mean_squared_error

##### XU LY DU LIEU (RAISIN)

data = pd.read_excel("Raisin_Dataset.xlsx")
data.Class = [1 if i == "Kecimen" else 0 for i in data.Class]
print(data)
print(data.info())
print(data.isnull().sum())
X = data.drop(["Class"], axis = 1)
y = data["Class"]
print("X: ", X.shape)
print("y: ", y.shape)
print(data["Class"].value_counts())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 50)
MinMax = MinMaxScaler(feature_range= (0,1))
X_train = MinMax.fit_transform(X_train)
X_test = MinMax.transform(X_test)

##### CÀI ĐẶT THUẬT TOÁN DECISION TREE
class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        ''' constructor ''' 
        
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        # for leaf node
        self.value = value
class DecisionTreeClassifier_handmade():
    def get_params(self, deep = False):
        return {}
    def __init__(self, min_samples_split=2, max_depth=2):
        ''' constructor '''
        
        # initialize the root of the tree 
        self.root = None
        
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree ''' 
        
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        
        # split until stopping conditions are met
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split["info_gain"]>0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        
        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split '''
        
        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")
        
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    # update the best split if needed
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
                        
        # return best split
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''
        
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        ''' function to compute information gain '''
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode=="gini":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    
    def entropy(self, y):
        ''' function to compute entropy '''
        
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini_index(self, y):
        ''' function to compute gini index '''
        
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
        
    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    def fit(self, X, Y):
        ''' function to train the tree '''
        
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    
    def predict(self, X):
        ''' function to predict new dataset '''
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''
        
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
classifier = DecisionTreeClassifier_handmade(min_samples_split=2, max_depth=3)
y_train1 = y_train.to_numpy().reshape(-1,1)
classifier.fit(X_train,y_train1)
classifier.print_tree()
y_pred = classifier.predict(X_test)
y_pred_train = classifier.predict(X_train)
print("DecisionTree_handmade: \n")
print('Accuracy Score DecisionTree_handmade Train dataset:', metrics.accuracy_score(y_train, y_pred_train))
# print('Accuracy Score DecisionTree_handmade Test dataset:', accuracy_score(y_test, y_pred))
# # print('Mean Score Error DecisionTree Test dataset:', mean_squared_error(y_test, y_pred))
# print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
# print('Classification Report:\n', classification_report(y_test, y_pred))


##### K-Fold Cross-Validation
from sklearn.model_selection import cross_validate
def cross_validation(model, _X, _y, _cv=3):
      '''Function to perform 3 Folds Cross-Validation
       Parameters
       ----------
      model: Python Class, default=None
              This is the machine learning algorithm to be used for training.
      _X: array
           This is the matrix of features.
      _y: array
           This is the target variable.
      _cv: int, default=3
          Determines the number of folds for cross-validation.
       Returns
       -------
       The function returns a dictionary containing the metrics 'accuracy', 'precision',
       'recall', 'f1' for both training set and validation set.
      '''
      _scoring = ["accuracy"]
      results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True)
      
      return {"Training Error": 1 - results['train_accuracy'],
              # "Training Precision scores": results['train_precision'],
              # "Mean Training Precision": results['train_precision'].mean(),
              # "Training Recall scores": results['train_recall'],
              # "Mean Training Recall": results['train_recall'].mean(),
              # "Training F1 scores": results['train_f1'],
              # "Mean Training F1 Score": results['train_f1'].mean(),
              "Validation Error": 1 - results['test_accuracy'],
              "Mean Training Error": 100 - results['train_accuracy'].mean()*100,
              "Mean Validation Error": 100 - results['test_accuracy'].mean()*100,
              "Sum Error: ": (100 - results['train_accuracy'].mean()*100) + (100 - results['test_accuracy'].mean()*100)
              # "Validation Precision scores": results['test_precision'],
              # "Mean Validation Precision": results['test_precision'].mean(),
              # "Validation Recall scores": results['test_recall'],
              # "Mean Validation Recall": results['test_recall'].mean(),
              # "Validation F1 scores": results['test_f1'],
              # "Mean Validation F1 Score": results['test_f1'].mean()
              }
# Grouped Bar Chart for both training and validation data

def plot_result(x_label, y_label, plot_title, train_data, val_data):
        '''Function to plot a grouped bar chart showing the training and validation
          results of the ML model in each fold after applying K-fold cross-validation.
         Parameters
         ----------
         x_label: str, 
            Name of the algorithm used for training e.g 'Decision Tree'
          
         y_label: str, 
            Name of metric being visualized e.g 'Accuracy'
         plot_title: str, 
            This is the title of the plot e.g 'Accuracy Plot'
         
         train_result: list, array
            This is the list containing either training precision, accuracy, or f1 score.
        
         val_result: list, array
            This is the list containing either validation precision, accuracy, or f1 score.
         Returns
         -------
         The function returns a Grouped Barchart showing the training and validation result
         in each fold.
        '''
        
        # Set size of plot
        plt.figure(figsize=(12,6))
        labels = ["1st Fold", "2nd Fold", "3rd Fold"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.050000, 1)
        plt.bar(X_axis-0.2, train_data, 0.4, color='blue', label='Training')
        plt.bar(X_axis+0.2, val_data, 0.4, color='red', label='Validation')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()

##### K Fold Cross Validation tren Decision Tree tu viet
decision_tree_model = classifier
X1 = np.array(X)
y = data["Class"].values.reshape(-1, 1)
decision_tree_result = cross_validation(decision_tree_model, X1, y, 3)
print(decision_tree_result)

##
X1 = np.array(X)
y = data["Class"].values.reshape(-1, 1)
for i in range(2, 5):
  for j in range(2, 5):
    classifier = DecisionTreeClassifier_handmade(min_samples_split=i, max_depth=j)
    classifier.fit(X_train,y_train1)
    decision_tree_model = classifier
    decision_tree_result = cross_validation(decision_tree_model, X1, y, 3)
    print(i,j, decision_tree_result)

###
classifier = DecisionTreeClassifier_handmade(min_samples_split=2, max_depth=3)
y_train1 = y_train.to_numpy().reshape(-1,1)
classifier.fit(X_train,y_train1)
classifier.print_tree()
y_pred = classifier.predict(X_test)
y_pred_train = classifier.predict(X_train)
print("DecisionTree_handmade: \n")
print('Accuracy Score DecisionTree_handmade Train dataset:', metrics.accuracy_score(y_train, y_pred_train))
print('Accuracy Score DecisionTree_handmade Test dataset:', accuracy_score(y_test, y_pred))
# print('Mean Score Error DecisionTree Test dataset:', mean_squared_error(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))

##### K-Fold Cross-Validation Decision Tree use Sklearn

# Test thử 2 TH truoc
#Test lần lặp 1:
print("DecisionTree use Sklearn 1: ")
from sklearn.tree import DecisionTreeClassifier
decision_tree_model = DecisionTreeClassifier()
decision_tree_result = cross_validation(decision_tree_model, X, y, 3)
print(decision_tree_result)
model_name = "Decision Tree"
plot_result(model_name,
            "Error",
            "Error scores in 3 Folds",
            decision_tree_result["Training Error"],
            decision_tree_result["Validation Error"])

#Test lần lặp 2:
print("DecisionTree use Sklearn 2: ")
decision_tree_model = DecisionTreeClassifier(max_depth = 4, min_samples_split = 3)
decision_tree_result = cross_validation(decision_tree_model, X, y, 3)
print(decision_tree_result)
model_name = "Decision Tree with max_depth = 4, min_samples_split = 3"
plot_result(model_name,
            "Error",
            "Error scores in 3 Folds",
            decision_tree_result["Training Error"],
            decision_tree_result["Validation Error"])

# Test lần lặp voi max_depth = i, min_samples_split = j
for i in range(2, 5):
  for j in range(2, 5):
    decision_tree_model = DecisionTreeClassifier(max_depth = i, min_samples_split = j)
    decision_tree_result = cross_validation(decision_tree_model, X, y, 3)
    print(i, j, decision_tree_result)

## Chon mo hinh voi i = 4, j = 3 vi Train_Error + Validation_Error ra nho nhat

##### DECISION TREE USE SKLEARN
from sklearn import tree
print("DecisionTree")
clf = tree.DecisionTreeClassifier(max_depth = 4, min_samples_split = 3)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_train = clf.predict(X_train)
print('Accuracy Score DecisionTree Train dataset:', metrics.accuracy_score(y_train, y_pred_train))
print('Accuracy Score DecisionTree Test dataset:', accuracy_score(y_test, y_pred))
# print('Mean Score Error DecisionTree Test dataset:', mean_squared_error(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))

##### THU NGHIEM TREN 1 SO MO HINH KHAC

### SVM

## SVM with kernel = linear
model=SVC(kernel='linear') 
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)
print("SVC(kernel = 'linear')")
print('Accuracy Score SVM Train dataset:', accuracy_score(y_train, y_pred_train))
print('Accuracy Score SVM Test dataset:', accuracy_score(y_test, y_pred))
# print('Mean Score Error SVM Test dataset:', mean_squared_error(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))
print("dual_coef: ", model.dual_coef_)
print("coef_: ", model.coef_)
print("\n")
## SVM with kernel = rbf
model=SVC(C=1, kernel='rbf') 
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)
print("SVC(kernel = 'rbf')")
print('Accuracy Score SVM Train dataset:', accuracy_score(y_train, y_pred_train))
print('Accuracy Score SVM Test dataset:', accuracy_score(y_test, y_pred))
# print('Mean Score Error SVM Test dataset:', mean_squared_error(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))
print("\n")


### With Neural Network
from sklearn.neural_network import MLPClassifier
clf2 = MLPClassifier(solver='lbfgs', alpha=1e-5,
      hidden_layer_sizes=(7,), random_state=1)
clf2.fit(X_train, y_train)
MLPClassifier(alpha=1e-05, hidden_layer_sizes=(7,), random_state=1,
              solver='lbfgs')
y_pred = clf2.predict(X_test)
y_pred_train = clf.predict(X_train)
print("Neural Network")
print('Accuracy Score Neural Network Train dataset:', metrics.accuracy_score(y_train, y_pred_train))
print('Accuracy Score Neural Network Test dataset:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))

print("w = ", [coef.shape for coef in clf2.coefs_])