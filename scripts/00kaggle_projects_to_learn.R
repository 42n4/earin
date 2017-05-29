The data mining project:
1) Your third project task is to choose data from my kaggle.com project list:
https://github.com/pwasiewi/dokerz/blob/master/rstudio/00kaggle_projects_to_learn.R
(I have selected just smaller sets – maybe not all links work or not all sets are suitable for case studies) or from http://archive.ics.uci.edu/ml/datasets.html and make a case study.
a) Do not choose datasets used during my lectures or too simple to make your case study (I mean datasets with a small number of attributes taken from games or simulations).
b) The used datasets should be obtained from the real world research. I think they are more interesting and appriopriate. 
c) I proposed some datasets, but you should choose the appriopriate one for your case study (fitting your classifiers e.g. trees, svm, regression and so on).
2) First you may read and run (maybe using my docker 42n4/rstudio?) some exemplary case studies from the Cichosz book commented during my lectures: 
https://github.com/pwasiewi/dokerz/blob/master/rstudio/dm_casestudy01.R
https://github.com/pwasiewi/dokerz/blob/master/rstudio/dm_casestudy02.R
https://github.com/pwasiewi/dokerz/blob/master/rstudio/dm_casestudy03.R
3) You should provide a (short - max 2-3 pages) documentation file (tex, word, txt) to my email containing: 
a) the description of attributes (columns) e.g. the number of discrete and continuous attributes (make factors from discrete ones, get rid of useless ones or remove one attribute from each pair of strongly correlated attributes). Find or make your target class attribute.
b) the description of used classification methods and their validation process (changing their parameters e.g. minsplit, cp, k-fold crossvalidation, boosting, bagging, adding cost matrices to them,  modifing input data e.g. standardization i.e. standard score, normalization, removing NA and so on).
c) the validation process summary including ROC plots and their comparison.
d) You may use some clustering methods for not labeled datasets (without an obvious target attribute). After this operation you can learn classifiers utilising input data and obtained cluster labels.
e) Enclose please your R code with comments.
4) All projects should be done individually and may be done in pairs on that condition that you divide your work into two parts belonging to individual persons, but each person should make some classifier validations.
5) Deadline: the 14th of June
6) The suggested language is R (ask for permission to use another programming language such as python). The proposed classifier library wrapper: caret
7) My e-learning materials are provided at these sites:
https://github.com/pwasiewi/earin
https://github.com/pwasiewi/dokerz/tree/master/rstudio
The latter site has instructions to run a docker with all needed libraries for three lecture case studies.  
8) My office hours on Fridays at 2 o’clock pm in room 22. Just email me and I will help you to choose your dataset, if you have some doubts or just to make further research.
My email: pwasiewi@elka.pw.edu.pl

#Datasets for the 3rd project (send me your choice - several alternatives - by email) 
https://www.kaggle.com/c/shelter-animal-outcomes/data
https://www.kaggle.com/annavictoria/speed-dating-experiment
https://www.kaggle.com/mylesoneill/game-of-thrones
https://www.kaggle.com/c/titanic
https://www.kaggle.com/c/detecting-insults-in-social-commentary/data
https://www.kaggle.com/c/random-number-grand-challenge/data
https://www.kaggle.com/c/rossmann-store-sales/data
https://www.kaggle.com/c/just-the-basics-strata-2013/data
https://www.kaggle.com/c/just-the-basics-the-after-party/data
https://www.kaggle.com/c/santander-customer-satisfaction/data
https://www.kaggle.com/c/Eurovision2010/data
https://www.kaggle.com/c/worldcup2010/data
https://www.kaggle.com/c/hivprogression/data
https://www.kaggle.com/c/tourism1/data
https://www.kaggle.com/c/march-machine-learning-mania-2016/data
https://www.kaggle.com/c/march-machine-learning-mania-2015/data
https://www.kaggle.com/c/march-machine-learning-mania/data
https://www.kaggle.com/c/twitter-personality-prediction/data
https://www.kaggle.com/c/informs2010/data
https://www.kaggle.com/c/flavours-of-physics/data
https://www.kaggle.com/c/deloitte-western-australia-rental-prices/data
https://www.kaggle.com/c/whats-cooking/data
https://www.kaggle.com/c/walmart-recruiting-trip-type-classification/data
https://www.kaggle.com/c/predict-west-nile-virus/data
https://www.kaggle.com/c/crowdflower-search-relevance/data
https://www.kaggle.com/c/liberty-mutual-group-property-inspection-prediction/data
https://www.kaggle.com/c/caterpillar-tube-pricing/data
https://www.kaggle.com/c/finding-elo/data
https://www.kaggle.com/c/15-071x-the-analytics-edge-spring-20152/data
https://www.kaggle.com/c/malware-classification/data
https://www.kaggle.com/c/15-071x-the-analytics-edge-competition-spring-2015/data
https://www.kaggle.com/c/restaurant-revenue-prediction/data
https://www.kaggle.com/c/forest-cover-type-prediction/data
https://www.kaggle.com/c/otto-group-product-classification-challenge/data
https://www.kaggle.com/c/walmart-recruiting-sales-in-stormy-weather/data
https://www.kaggle.com/c/bike-sharing-demand/data
https://www.kaggle.com/c/random-acts-of-pizza/data
https://www.kaggle.com/c/poker-rule-induction/data
https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data
https://www.kaggle.com/c/the-analytics-edge-mit-15-071x/data
https://www.kaggle.com/c/allstate-purchase-prediction-challenge/data
https://www.kaggle.com/c/acquire-valued-shoppers-challenge/data
https://www.kaggle.com/c/criteo-display-ad-challenge/data
https://www.kaggle.com/c/afsis-soil-properties/data
https://www.kaggle.com/c/learning-social-circles/data
https://www.kaggle.com/c/data-science-london-scikit-learn/data
https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data
https://www.kaggle.com/c/amazon-employee-access-challenge/data
https://www.kaggle.com/c/the-seeclickfix-311-challenge/data
https://www.kaggle.com/c/battlefin-s-big-data-combine-forecasting-challenge/data
https://www.kaggle.com/c/belkin-energy-disaggregation-competition/data
https://www.kaggle.com/c/see-click-predict-fix/data
https://www.kaggle.com/c/crowdflower-weather-twitter/data
https://www.kaggle.com/c/conway-s-reverse-game-of-life/data
https://www.kaggle.com/c/genentech-flu-forecasting/data
https://www.kaggle.com/c/pakdd-cup-2014/data
https://www.kaggle.com/c/GEF2012-wind-forecasting/data
https://www.kaggle.com/c/global-energy-forecasting-competition-2012-load-forecasting/data
https://www.kaggle.com/c/DarkWorlds/data
https://www.kaggle.com/c/traveling-santa-problem/data
https://www.kaggle.com/c/visualize-the-state-of-education-in-colorado/data
https://www.kaggle.com/c/predict-who-is-more-influential-in-a-social-network/data
https://www.kaggle.com/c/bluebook-for-bulldozers/data
https://www.kaggle.com/c/chess/data
https://www.kaggle.com/c/tourism2/data
https://www.kaggle.com/c/socialNetwork/data
https://www.kaggle.com/c/R/data
https://www.kaggle.com/c/unimelb/data
https://www.kaggle.com/c/ChessRatings2/data
https://www.kaggle.com/c/overfitting/data
https://www.kaggle.com/c/ClaimPredictionChallenge/data
https://www.kaggle.com/c/GiveMeSomeCredit/data
https://www.kaggle.com/c/DontGetKicked/data
https://www.kaggle.com/c/dsg-hackathon/data
https://www.kaggle.com/c/online-sales/data
https://www.kaggle.com/c/MusicHackathon/data
https://www.kaggle.com/c/msdchallenge/data
https://www.kaggle.com/c/harvard-business-review-vision-statement-prospect/data
https://www.kaggle.com/c/acm-sf-chapter-hackathon-small/data
#to learn sth about
https://www.kaggle.com/c/introducing-kaggle-scripts
