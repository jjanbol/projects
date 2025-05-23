![estimator](https://raw.githubusercontent.com/jjanbol/projects/refs/heads/main/wrist/Unknown.png)


The parameters were found by parameter grid search in which the values of segment length varied from 8 to 1024 and k number of clusters varied from 15 to 300. These numbers were chosen to cover a wide range as it is standard approach for grid parameter search. As a result k = 25 and segment length = 128 were found to be the best in this case.

The segmentation length decides how much data is extracted from each activity. With smaller numbers of segment length, it will be able to capture more detailed patterns and if the segment length is too large it may become too generalized. Also the number of rows that are being abandoned due to the segment length also determines how much information is lost which will affect the training. In this case 128 may be the sweet spot where it doesn't lose or abandon too much data while capturing enough of the details from the data.

The k value decides also the level of detail that is being captured. if the k value is too high, it might lead to overfitting as it would create too many clusters and sensitive to noise. If it is too low, it might generalize so much that accuracy is low. k=25 might be the parameter that balances the amount of details with generalization well, leading to better test accuracy (87%) overall.
