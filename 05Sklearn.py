# =============================================
#    Scikit-learn
# =============================================
# =============================================
#    Getting Started
# =============================================
# ==== Fitting and predicting: estimator basics
# Here is a simple example where we fit a RandomForestClassifier to some very basic data:
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
X = [[1, 2, 3], [11, 12, 13]]
y = [0, 1]
clf.fit(X, y)

# Once the estimator is fitted, it can be used for predicting target values of new data.
clf.predict(X)  # predict classes of the training data
clf.predict([[4, 5, 6], [14, 15, 16]])  # predict classes of new data

# ==== Transformers and pre-processors
from sklearn.preprocessing import StandardScaler
X = [[0, 15], [1, -10]]
# scale data according to computed scaling values
StandardScaler().fit(X).transform(X)
