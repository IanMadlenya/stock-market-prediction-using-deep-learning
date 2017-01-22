![ucl-logo](http://static.ucl.ac.uk/img/ucl-logo.svg)
## What is this project about?
This is a research project at the Department of Electronic & Electrical Engineering at University College London (UCL), supervised by Dr Miguel Rodrigues, named "Stock Market Prediction Using Deep Learning Techniques".

* Output data: next-day stock price trend (1 for rise, 0 for not rise)
* Input data
	* past price series (this generates about 50% predicting accuracy, which is not desirable)
		* Expressed in arrays (traditional)
		* Expressed in images (innovative)
	* Twitter data (this can boost predicting accuracy because it is more informative)
		* Volume
		* Sentiment
		* Volume & Sentiment
* [My website about the correlation between stock market and Twitter] (https://zhedongzheng.github.io)

## How to run the code?
* To run all the code, you need to pre-install the following libraries in Python 3:
  * numpy, pandas, matplotlib, scikit-learn (basic suite, most people already have)
  * [pandas-datareader] (https://github.com/pydata/pandas-datareader) (for obtaining stock prices)
  * [opencv] (http://opencv.org/) (for images)
  * [python-twitter] (https://github.com/bear/python-twitter) (for visiting Twitter api)
  * [tensorflow r0.12] (https://www.tensorflow.org/) (for advanced neural networks)
  * these libraries can all be easily installed through either `pip` or `conda`
* To run any code, you just need to download the folder "source-code", and cd into the folder, then run the code. For example, if you want to run `twi_sen_vs_price.py` in the folder `../source-code/plot`, this is an example of command line code (windows):
	
	```
	d:
	cd D:\..\source-code\plot 
	python twi_sen_vs_price.py
	```
	Then you will see the result
