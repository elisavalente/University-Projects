# LSTM for Sales Forecast

The proposed project aims to forecast sales of a weight control product, using deep learning.

- Data set analysis
	- The data set provided contains the record of sales and advertising costs for a weight control product for 36 consecutive months. 
	- It consists of 3 columns and 36 entries, with the columns corresponding, in order, to the month, the advertising cost and the number of sales. 
	- The cost varies between 12 and 36.5 and the value of sales between 1 and 65. 

	Based on this data, the intention is to create an LSTM model that is able to forecast sales according to the investment in advertising and the month in question. For this, will be used 24 months for training and 12 for testing.