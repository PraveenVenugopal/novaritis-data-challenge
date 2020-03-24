# Data Challenge
A temporary store for code, models and visuals. An incremental and continuous approach was taken to handle the challenge. Data processing and visual (univariate) analysis was completed in 4 hours. A two-pronged approach using Supervised learning (DL/ML) methods and dimensionality reduction/ clustering methods was employed. 

The former, though straight forward had significant challenges due to severe class imbalance and high dimensionality. The latter was more informative and lead to some interesting learning. Inferences from failures of Supervised learning was used to construct a new regression based Forecast model (as opposed to the classification models earlier). Visuals are employed to show the effectivity of the forecasts on test set.Please refer the slides for more technical details.

# Pre-requisites
1. Python 3.6.1 or higher
2. Keras 2.2.4
3. Matplotlib 3.0.3
4. Numpy 1.15.4
5. Pandas 0.24.1
6. Scikit-Learn 0.22.1
7. Tensorflow 1.7.1



# Instructions
1. Clone the repository to your local system
2. Place the excel data sheet "IE01.552.051-data_pack.xlsx" in to the repo ( inside the folder "
novartis-data-challege " )
3. Run data_processor.py
4. Run construct_deep_forecast_models.py
5. The forecast plots and trained models will be stored in "img" and "models" folders respectively.
