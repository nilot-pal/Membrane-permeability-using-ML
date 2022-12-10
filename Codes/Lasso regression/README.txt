This folder contains the code for the LASSO model training and plotting,
developed by Nilotpal Chakraborty and Anri Karanovich 
(see the comments in the python files for a detailed breakdown of contributions)

The main program is main.py

To run the program:

	1.   Open the main.py file, go to lines 30-36, and uncomment the correct line
     	which correspoinds to the set of input data features that you are planning to use 
	(see the final report, Table 1, for reference)

	For example, the following selection uses Set_1 (key features only, denoted here as Inp)
		ModelType, include_Inp, include_FP, include_MD = ("Set_1", True, False,  False)
		#ModelType, include_Inp, include_FP, include_MD = ("Set_2", True, True, False)
		...
		#ModelType, include_Inp, include_FP, include_MD = ("Set_7", False, True, True)
	
	2.   Choose the values of other running parameters (lines 38-61; see comments for explanations)

	3.   Run the main.py program.

	     Note that this file MUST be in the same subfolder as module_1.py, module_2.py, 
	     AND the Datasets subfolder, which contains the raw dataset file 'oc8b00718_si_002.txt',
	     as well as some pre-processed CSV data files (created by Seokgyun Ham), 
	     which are used by default to ensure consistency wiht the MLP model. 
	     (To read the raw dataset directly, set source='txt' in line 41 of lasso_main.py) 

	4. The output data will be in the newly-created subfolder, whose name starts as Set_#_...
	   Of special interest are:
		- Set_#_performance.txt:  performance measures of the model, largest weights, etc.
		- MSE_vs_alpha_cv_5: 	  plot of the average-fold MSE vs the alpha parameter in LassoCV
					  alpha search (Fig. 6 of the report)
		- learning_curve:  	  learning curve (scores vs number of input samples) (Fig. 7 in report)
		- Weight_Evolution:       weight evolution curve (Fig.9)
		

	   other files in this subfolder are just supplementary files, not containing output data 


	     
 

