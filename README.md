# ml-library
Pure Python and NumPy implementations of machine learning algorithms.  
The aim is to deepen the author's understanding of these algorithms.  
Not intended to achieve outstanding performance/efficiency.  
Generally inspired by [scikit-learn](https://github.com/scikit-learn/scikit-learn).

![](https://github.com/lkk7/ml-library/blob/master/examples/lin_regr.gif)
![](https://github.com/lkk7/ml-library/blob/master/examples/log_regr.gif)

## Included
"..." means "to do more", "(...)" means "to update"
- Regression
  * Linear regression:
    * Solving methods: Gradient descent, SGD, normal equation
    * Regularization methods: L1 (lasso), L2 (ridge), elastic net
  * ...
- Classification
  * Logistic Regression
    * Solving methods: Gradient descent, SGD
    * Regularization methods: L1, L2, elastic net
  * Perceptron (...)
  * K-nearest neighbors
  * Naive Bayes classifier (...)
  * ...
- Unsupervised methods
  * K-means clustering
  * ...
- Other
  * Metrics
    * Euclidean, Manhattan distance
    * ...
  * Model selection
    * K-fold cross-validation

## Notes
As the library is inspired by [scikit-learn](https://github.com/scikit-learn/scikit-learn), it could be expected to be designed in a tiny bit more object-oriented way.  
However, the author's main focus is to develop readable and usable algorithms one by one in order to learn their foundations. Besides, every model has its `fit()` and `predict()` (if possible) as expected from a [scikit-learn](https://github.com/scikit-learn/scikit-learn) clone.
