from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# --------------------------------------------------#
#                Train-test Split Data              #
# --------------------------------------------------#
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.33, random_state=0)

# --------------------------------------------------#
#        Grid Search Cross-Validation               #
# --------------------------------------------------#
#  Example with DecissionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

param_grid = {'max_depth': list(range(2,16,2)),
              'min_samples_split': list(range(2,16,2))}
# np.random.seed(0)
clf = GridSearchCV(DecisionTreeRegressor(), 
                   param_grid,
                   scoring='neg_mean_squared_error',
                   cv=3, 
                   n_jobs=1, verbose=1)

clf.fit(X=X_train, y=y_train)

print(clf.best_params_, -clf.best_score_)

# We could also input in the cv field. If we want to manage better params:
from sklearn.model_selection import KFold
cv_grid = KFold(n_splits=3, shuffle=True, random_state=0)
# and GridSearchCV(..., cv=cv_grid, ...)

# --------------------------------------------------#
#        Randomized Search Cross-Validation         #
# --------------------------------------------------#
from sklearn.model_selection import RandomizedSearchCV
budget = 10
clf = RandomizedSearchCV(tree.DecisionTreeRegressor(), 
                         param_grid,
                         scoring='neg_mean_squared_error',
                         cv=cv_grid, 
                         n_jobs=1, verbose=1,
                         n_iter=budget
                        )
# For Randomized Search, we can define the search space with statistical distributions, rather than using particular values as we did before.
# For continuous hyper-parameters we could use continuous distributions such as uniform or expon (exponential).

from scipy.stats import uniform, expon
from scipy.stats import randint as sp_randint

param_grid = {'max_depth': sp_randint(2,16),
              'min_samples_split': sp_randint(2,16)}
clf = RandomizedSearchCV( ..., param_grid, ....)


# --------------------------------------------------#
#               Nested CrossValidation              #
# --------------------------------------------------#
# Example: Model evaluation with 5-fold crossvalidation and hyper-parameter tuning with 3-fold crossvalidation
# (https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
# There is an external loop (for evaluating models) and an 

from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from scipy.stats import uniform, expon
from sklearn import metrics

# Evaluation of model (outer loop)
cv_evaluation = KFold(n_splits=5, shuffle=True, random_state=0)

# Internal loop (for hyper-parameter tuning).
clf = RandomizedSearchCV(tree.DecisionTreeRegressor(), 
                         param_grid,
                         scoring='neg_mean_squared_error',
                         # 3-fold for hyper-parameter tuning
                         cv=3, 
                         n_jobs=1, verbose=1,
                         n_iter=budget
                        )

scores = -cross_val_score(clf, 
                          X, y, 
                          scoring='neg_mean_squared_error', 
                          cv = cv_evaluation)

print(scores)
# The mean of the 5-fold crossvalidation is the final score of the model
print(scores.mean(), "+-", scores.std())

# --------------------------------------------------#
#              Obtaining final model                #
# --------------------------------------------------#

# Recall clf is one of the CV methods
clfFinal = clf.fit(X,y) 
print(clf.best_params_, -clf.best_score_)

# --------------------------------------------------#
#          Model-based Opt. (Bayesian Opt.)         #
# --------------------------------------------------#

from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
from sklearn import metrics

from scipy.stats import uniform, expon
from scipy.stats import randint as sp_randint

# Search space with integer uniform distributions
param_grid = {'max_depth': Integer(2,16),
              'min_samples_split': Integer(2,16)}

budget = 20

clf = BayesSearchCV(tree.DecisionTreeRegressor(), 
                    param_grid,
                    scoring='neg_mean_squared_error',
                    cv=3,    
                    n_jobs=1, verbose=1,
                    n_iter=budget
                    )
clf.fit(X=X_train, y=y_train)

y_test_pred = clf.predict(X_test)
print(metrics.mean_squared_error(y_test, y_test_pred))

print(clf.best_params_, -clf.best_score_)

# --------------------------------------------------#
#     Model-based Opt. (Bayesian Opt.) with Optuna  #
# --------------------------------------------------#
import optuna

# trial_suggest_categorical(‘criterion‘, [‘gini', ‘entropy']) 
# trial.suggest_int(‘max_depth‘,1, 6) 
# trial.suggest_uniform(‘min_samples_split‘, 0, 1) 
# trial.suggest_loguniform(‘C‘, 10-6, 10+6) 
# trial.suggest_discrete_uniform(‘min_samples_split’, 0.0, 1.0, 0.1)

def objective(trial):
    
    max_depth = trial.suggest_int("max_depth", 2, 16)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 16)
    
    clf = tree.DecisionTreeRegressor(max_depth=max_depth, 
                                        min_samples_split=min_samples_split)
    
    scores = -cross_val_score(clf, X_train, y_train, 
                              scoring='neg_mean_squared_error', 
                              n_jobs=1,
                              # Using again the same inner 3-folds for hyper-parameter tuning
                              cv=cv_grid)
    inner_mse = scores.mean()
    return inner_mse

budget = 20
np.random.seed(0)
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=budget)
# Output: [I 2020-11-29 20:46:06,930] Trial 0 finished with value: 30.763603246031238 and parameters:
#                        {'max_depth': 2, 'min_samples_split': 16}. Best is trial 0 with value: 30.763603246031238.
# ...

print(study.best_params, study.best_value)

# Now, we have to train a model with those hyper-parameters on the complete train partition.
#  Then, the model will be evaluated on the test partition. We have to do it by hand:

clf = tree.DecisionTreeRegressor(**study.best_params) # Train the model with the best hyper-parameters
clf.fit(X_train, y_train) # on the complete train partition
y_test_pred = clf.predict(X_test) # Get the predictions on the test partition
print('outer MSE with Optuna:')
print(metrics.mean_squared_error(y_test, y_test_pred))

# Final:
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=budget,show_progress_bar=False)
clf = tree.DecisionTreeRegressor(**study.best_params) # Train the model with the best hyper-parameters
clfFinal = clf.fit(X,y) #Full data


# --------------------------------------------------------------------------------#
#     Combined Algorithm Selection and Hyper-parameter (CASH) Tuning with Optuna  #
# --------------------------------------------------------------------------------#
from sklearn import neighbors, tree
from sklearn.model_selection import cross_val_score
from sklearn import metrics

def objective(trial):
    
    method = trial.suggest_categorical('method', ['tree', 'knn'])
    if(method == 'tree'):
        max_depth = trial.suggest_int("max_depth", 2, 16)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 16)
        clf = tree.DecisionTreeRegressor(max_depth=max_depth, 
                                         min_samples_split=min_samples_split)
    else:
        n_neighbors = trial.suggest_int('n_neighbors', 1, 6)
        clf = neighbors.KNeighborsRegressor(n_neighbors = n_neighbors)
        
    scores = -cross_val_score(clf, X_train, y_train, 
                              scoring='neg_mean_squared_error', 
                              n_jobs=1, cv=cv_grid)
    inner_mse = scores.mean()
    return inner_mse

budget = 20
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=budget)

print(study.best_params, study.best_value)
#... train now on full data, etc.


# -------------------------------------------------#
#     KMeans Elbow Method for finding optimal K    #
# -------------------------------------------------#

def kmeans_elbow(x_Data, k_max = 10):
    '''
	Computes several KMEANS clustering algorithm over data and plots distortion
	graph to evaluate the optimum K value using the Elbow method.
    '''

    distortions = []
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, n_jobs=-1).fit(df_X_std)
        kmeanModel.fit(df_X_std)
        distortions.append(sum(np.min(cdist(df_X_std, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / df_X_std.shape[0])

    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('K vs Distortion value (Elbow method)')
    plt.show()