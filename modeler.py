from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

class Modeler():
    def __init__(self, dataframe, target, X_train, X_test, y_train, y_test):
        self.target = target
        self.df = dataframe
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.trials = []

    def random_forest(self, features, kwargs):
        '''bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
            oob_score=False, random_state=23, verbose=0, warm_start=False'''

        trial = {'type': 'Random Forest',
                'params':kwargs,
                'features':features}

        try:
            X = self.df[features]
            y = self.df[self.target]
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23)
            X_train = self.X_train
            X_test = self.X_test
            y_train = self.y_train
            y_test = self.y_test


            rfc = RandomForestClassifier(**kwargs)

            rfc.fit(X_train, y_train)
            rfc_pred = rfc.predict(X_test)
            trial['Test Accuracy Score'] = accuracy_score(y_test, rfc_pred)
            trial['Test F1 Score'] = f1_score(y_test, rfc_pred)

            # checking accuracy on the test data
            print('Test Accuracy score: ', str(accuracy_score(y_test, rfc_pred)))
            # checking accuracy on the test data
            print('Test F1 score: ', str(f1_score(y_test, rfc_pred)))

            self.trials.append(trial)

        except Exception as e:
            print(e)

    def grid_search(self,features, kwargs):
        '''GridSearchCV(cv=5, error_score='raise-deprecating',
        estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
            oob_score=False, random_state=23, verbose=0, warm_start=False),
        fit_params=None, iid='warn', n_jobs=-1,
        param_grid={'n_estimators': [100, 200, 300, 400]},
        pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
        scoring=None, verbose=0)'''

        trial = {'type': 'grid_search',
                'params':kwargs,
                'features':features}

        try:
            X = self.df[features]
            y = self.df[self.target]
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23)
            X_train = self.X_train
            X_test = self.X_test
            y_train = self.y_train
            y_test = self.y_test

            CV_rfc = GridSearchCV(**kwargs)
            CV_rfc.fit(X_train, y_train)

            print(CV_rfc.best_params_)


            #Identify the best score during fitting with cross-validation
            print(CV_rfc.cv_results_)

            CV_rfc_pred = CV_rfc.best_estimator_.predict(X_test)

            trial['Test Accuracy Score'] = accuracy_score(y_test, CV_rfc_pred)
            trial['Test F1 Score'] = f1_score(y_test, CV_rfc_pred)

            self.trials.append(trial)

            # checking accuracy
            print('Test Accuracy score: ', str(accuracy_score(y_test, CV_rfc_pred)))
            # checking accuracy
            print('Test F1 score: ', str(f1_score(y_test, CV_rfc_pred)))

        except Exception as e:
            print(e)

    def decision_tree(self, features, kwargs):
        '''DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')'''

        trial = {'type': 'Decision Tree',
                'params':kwargs,
                'features':features}

        try:
            X = self.df[features]
            y = self.df[self.target]
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23)
            X_train = self.X_train
            X_test = self.X_test
            y_train = self.y_train
            y_test = self.y_test

            clf = DecisionTreeClassifier(**kwargs)
            clf.fit(X_train, y_train)
            clf_y_pred = clf.predict(X_test)

            trial['Test Accuracy Score'] = accuracy_score(y_test, clf_y_pred)
            trial['Test F1 Score'] = f1_score(y_test, clf_y_pred)

            print('Accuracy:' + str(accuracy_score(y_test, clf_y_pred)))
            print('F1: ' + str(f1_score(y_test, clf_y_pred)))

            self.trials.append(trial)

        except Exception as e:
            print(e)

    def log_reg(self, features, kwargs):

        '''LogisticRegression(C=1000000000.0, class_weight=None, dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='ovr', penalty='l2', random_state=None,
          solver='liblinear', tol=0.0001, verbose=0)'''

        trial = {'type': 'Logistic Regrerssion',
                'params':kwargs,
                'features':features}

        try:
            X = self.df[features]
            y = self.df[self.target]
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23)
            X_train = self.X_train
            X_test = self.X_test
            y_train = self.y_train
            y_test = self.y_test

            logreg = LogisticRegression()
            logreg.fit(X_train, y_train)
            y_pred_class = logreg.predict(X_test)

            trial['Test Accuracy Score'] = accuracy_score(y_test, y_pred_class)
            trial['Test F1 Score'] = f1_score(y_test, y_pred_class)

            print('Accuracy:' + str(accuracy_score(y_test, y_pred_class)))
            print('F1: ' + str(f1_score(y_test, y_pred_class)))

            self.trials.append(trial)

        except Exception as e:
            print(e)




    def knn(self, features, kwargs):
        '''KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=None, n_neighbors=1, p=2,
           weights='uniform')'''

        trial = {'type': 'KNN',
                'params':kwargs,
                'features':features}

        try:
            X = self.df[features]
            y = self.df[self.target]
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23)
            X_train = self.X_train
            X_test = self.X_test
            y_train = self.y_train
            y_test = self.y_test

            knn = KNeighborsClassifier(**kwargs)
            knn.fit(X_train, y_train)
            knn_y_pred = knn.predict(X_test)

            trial['Test Accuracy Score'] = accuracy_score(y_test, knn_y_pred)
            trial['Test F1 Score'] = f1_score(y_test, knn_y_pred)

            print('Accuracy:' + str(accuracy_score(y_test, knn_y_pred)))
            print('F1: ' + str(f1_score(y_test, knn_y_pred)))

            self.trials.append(trial)

        except Exception as e:
            print(e)
