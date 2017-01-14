![ucl-logo](http://static.ucl.ac.uk/img/ucl-logo.svg)
## What is this project about?
This is a third-year project at the Department of Electronic & Electrical Engineering at University College London (UCL), supervised by Dr Miguel Rodrigues, named "Stock Market Prediction Using Deep Learning Techniques".

* [My website about the correlation between the stock market and Twitter] (https://zhedongzheng.github.io)

* Predictors
![predictors](https://github.com/zhedongzheng/stock-market-prediction-using-deep-learning/blob/master/intro/features.png)

* Collecting data
![data](https://github.com/zhedongzheng/stock-market-prediction-using-deep-learning/blob/master/intro/data.png)

## How to run the code?
* To run all the code, you need to pre-install the following libraries in Python 3:
  * numpy, pandas, matplotlib, scikit-learn (basic suite, most people already have)
  * [pandas-datareader] (https://github.com/pydata/pandas-datareader)
  * [opencv] (http://opencv.org/)
  * [python-twitter] (https://github.com/bear/python-twitter)
  * [tensorflow] (https://www.tensorflow.org/)
  * these libraries can all be easily installed through either `pip` or `conda`
* To run any code, you just need to download the folder "source-code", and cd into the folder, then run the code. For example, if you want to run `twi_sen_vs_price.py` in the folder `../source-code/plot`, this is an example of command line code (windows):
	
	```
	d:
	cd D:\..\source-code\plot 
	python twi_sen_vs_price.py
	```
	Then you will see the result
