# Assignment-2
Implemented the following in this assignment:
1. Naive Bayes classifier for rating prediction based on review
2. MNIST handwritten digit classifier using CVXOPT and libsvm packages

## Naive Bayes

### Command
`./run.sh 1 train_data_path test_data_path part_number_character`

### Notes
1. [Dataset](https://owncloud.iitd.ac.in/nextcloud/index.php/s/8A6HrkHcB3E7iKk)
2. `utils.py` contains important functions which are relevant to the assignment
3. part_num_character = a or b or c ...

## SVM
### Command
`./run.sh 1 train_data_path test_data_path type_of_classification part_num_character`

### Helpers
1. [CVXOPT](https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf)
2. Refer to Bishop's book (Pg 330-340) for getting reference about theory of various formulae used in SVMs
3. [Instructions for installing libsvm](https://superuser.com/questions/159966/how-to-install-libsvm-for-python-2-65-on-mac-os-x-10-6-4-snow-leopard)
	* First copy svm.py to `libsvm-3.23/python/` 
	* Run the file from there
	* Make sure to provide correct path while using `run.sh`

### Notes
1. [Dataset](https://drive.google.com/file/d/1OgQOTgODBKCuYX1B3E1gDmhjbOOcq4Wq/view)
2. `type_of_classification` = '0' for binary and '1' for multiclass classification
3. part_num_character = a or b or c ...

## Report
Click [here](https://www.overleaf.com/read/ywvgqcrzhqpj) to go to the report