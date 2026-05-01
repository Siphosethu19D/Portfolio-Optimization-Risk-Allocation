# Portfolio-Optimization-Risk-Allocation

This project applies portfolio optimisation methods from the paper using real historical market data. The study uses 10 Dow Jones assets over 8 quarterly periods and compares different investment strategies. The main methodology includes a two-state Markov regime-switching model (bull and bear markets), Cardinality-Constrained Quadratic Optimisation (CCQO) solved using Gurobi, and performance evaluation using Sharpe ratios after including transaction costs.

The repository contains the main Python code file, the README.md file, the requirements.txt file, and a data folder containing the CSV files used in the project. The CSV files are stored in the data folder on GitHub and should remain there so the code can load them correctly.

To run the project, first install all required Python libraries using the requirements.txt file. The main libraries used are numpy, pandas, gurobipy, and matplotlib. After installing the libraries, open the Python notebook or Python script and run one chunk at a time in the correct order.

The first section of the code imports all required libraries and sets the model configuration. This includes the paper parameters, the list of the 10 Dow Jones assets, and the ticker names used in the dataset.

The second section loads and prepares the data. Stock data is downloaded from Yahoo Finance, while the T-bill data used as the risk-free asset comes from FRED. Since the data was already downloaded previously, the project loads the CSV files directly from the data folder.

The third section computes excess returns and the second moment matrix. This includes calculating the mean excess returns and the matrix D. In this project, D represents the second moment of excess returns and not only the covariance matrix.

The fourth section defines the CCQO solver using Gurobi. This function is the main optimisation tool used in all three portfolio strategies.

The fifth section implements the CMV-Static portfolio strategy. This is the buy-and-hold benchmark strategy. The first two quarters are used as training data, the optimisation problem is solved once, and the portfolio weights are then held fixed from Quarter 3 to Quarter 8.

The sixth section implements the CMMV-Independent strategy. This strategy assumes returns are independent over time. The model is re-estimated at each rebalancing period using updated data, but no regime-switching structure is used.

The seventh section implements the CMMV portfolio strategy. This is the multi-period model with regime switching. Market periods are classified into bull and bear states using market return thresholds, regime-specific parameters are estimated, and the dynamic policy is applied.

After the three strategies are completed, transaction costs are applied to all strategies. The final sections evaluate performance using Sharpe ratios and other summary measures. The project also reproduces selected important results from the paper, such as the Sharpe Ratio comparison table.

To run the project successfully, make sure the CSV files remain inside the data folder, install all required packages, and run the code one chunk at a time in order. Gurobi must also be installed and properly licensed before running the optimisation sections.
