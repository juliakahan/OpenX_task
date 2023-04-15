# OpenX_task

File *openx.py* contains methods defining heuristic model, random forest and linear regression model, as well as the neural network model and their evalution methods. 
File *app.py* provides the REST API serving chosen model, which may be obtained by running the openx.py file - *due to the long training time and stated deadline the current version of the tool does not contain saved .h5 models* and Docker Image, but the work is in progress and results can change in 4h. In *features.json* you can find the input feature values and model selection. The Covertype dataset may be found in */DATA/covertype.info*
