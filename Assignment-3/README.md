# Assignment-3
Implemented the following from scratch in this assignment:
1. Decision Trees for credit card defaults
2. Neural Network for poker hand dataset

## Decision Trees

### Command
`./dtree.sh part_number(1/2/3/4/5/6) train_datapath test_datapath validation_datapath`

### Helpers
1. [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
2. [Decision Tree](https://scikit-learn.org/stable/modules/tree.html)
3. [Decision Tree](https://github.com/scikit-learn/scikit-learn/issues/5442)
4. [Random Forests](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
5. [CART Algorithm](https://en.wikipedia.org/wiki/Predictive_analytics#Classification_and_regression_trees_.28CART.29)

### Notes
1. [Dataset](https://drive.google.com/drive/folders/13thcAc_eEa_NPmUt8tqg2_v3wmel9uxx)

## Neural Network

### Workflow
1. `./preprocess.sh raw_train_datapath raw_test_datapath ohe_train_datapath ohe_test_datapath`
	* Above command generates one hot encoded file for the raw data and saves it into the ohe_ path provided
	* Number of columns in ohe file have been hard-coded. Can be changed depending on raw data
2. `./nn.sh config_datapath train_datapath test_datapath`
	* Runs neural network using one hot encoded data obtained from datapath provided

### Helpers
1. [One Hot encoding](https://www.kaggle.com/dansbecker/using-categorical-data-with-one-hot-encoding)
2. [Sub-gradient](https://en.wikipedia.org/wiki/Subderivative)

### Notes
1. [Dataset](https://archive.ics.uci.edu/ml/datasets/Poker+Hand)
2. Config files are stored in `./config/` each of which includes the following
	* Number of inputs to first layer
	* Number of outputs from neural network
	* Batch size for SGD
	* Number of layers
	* Netowrk architecture (Eg: 2 layered network with 5 neurons each will be 5 5)
	* Activation function
	* Learning rate (adaptive/fixed)

## Report
Click [here](https://docs.google.com/document/d/1N5b3_DP649q8z6UsubnyO47vMeagCoUeG-sY4Oe3i0U) for report.