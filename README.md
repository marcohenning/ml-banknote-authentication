# Machine Learning Banknote Authentication

A machine learning model that authenticates banknotes based on the banknote authentication dataset.

## About

This simple machine learning model is trained on the [banknote authentication dataset](https://github.com/Kuntal-G/Machine-Learning/blob/master/R-machine-learning/data/banknote-authentication.csv) and classifies banknotes into the categories "authentic" and "forged" based on four parameters obtained through image analysis of the banknote. It's built with PyTorch and uses other libraries like pandas for handling the dataset, Matplotlib for plotting the training results, scikit-learn to split the data into train and test sets and PrettyTable for printing the results to the console in table format.

## Dataset

The dataset is made up of four columns containing attributes and one column containing the final classification. The data was extracted from images that were taken from genuine and forged banknotes using an industrial camera used for print inspection. Four specific features were then extracted from the images. They describe the variance of the image, the skew of the image, the curtosis of the image and the entropy of the image. Each banknote can either be classified as authentic or forged. The entire dataset consists of 1372 records with an even distribution between the two classes.

| variance | skew | curtosis | entropy | class |
|:--------:|:----:|:--------:|:-------:|:-----:|
| 3.6216 | 8.6661 | -2.8073 | -0.44699	| 0 |
| -1.3971 | 3.3191 | -1.3927 | -1.9948 | 1 |

## Results

The neural network was trained for 200 epochs with a learning rate of 0.01 using Adam as the optimizer and CrossEntropyLoss as the criterion. Training was done on 80% of the dataset while the remaining 20% were used to validate the results.

```
+-------+---------------+
| Epoch |      Loss     |
+-------+---------------+
|   1   |   0.6560994   |
|   10  |   0.4711799   |
|   20  |   0.24263428  |
|   30  |   0.07837704  |
|   40  |  0.022396613  |
|   50  |  0.007130639  |
|   60  |  0.0033932838 |
|   70  |  0.0021025292 |
|   80  |  0.0015182103 |
|   90  |  0.001204703  |
|  100  |  0.0010083504 |
|  110  | 0.00086529134 |
|  120  |  0.0007534996 |
|  130  | 0.00066360075 |
|  140  |  0.0005885827 |
|  150  |  0.0005209655 |
|  160  | 0.00046208754 |
|  170  |  0.0004130033 |
|  180  |  0.0003714374 |
|  190  |  0.0003360066 |
|  200  | 0.00030568257 |
+-------+---------------+
```

![Training](https://github.com/user-attachments/assets/6f6ef2f8-d22f-4ddd-be02-33229751d719)

The model achieved an accuracy of 100% on both the training and the testing set.

```
Test Accuracy: 100.0% (275/275)
```

## License

This software is licensed under the [MIT license](LICENSE).
