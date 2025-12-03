Selecting indicators for inputs fed to a neural network is a critical preparatory step that significantly influences the training process, model efficiency, and final performance.

### Usefulness of Selecting Indicators for Neural Network Inputs

The primary objective of selecting indicators is to determine an **optimal set of indicators and historical depth** to analyze data and forecast target values effectively.

1.  **Managing Information Volume and Complexity:** If you load all available information into the neural network, the learning process may prolong indefinitely without guaranteeing the desired outcome. A large volume of information necessitates a considerably large input layer of neurons and substantial connections, requiring more training time.
2.  **Addressing Data Diversity:** When diverse input data is fed to the neural network, some inputs might have values that significantly exceed the magnitudes of others. Such large values will have a greater impact on the final result, even if their actual influence might be low, which complicates the training process as it becomes challenging to discern the impact of smaller values.
3.  **Efficiency and Resource Reduction:** Conducting preparatory work, such as correlation analysis, allows for the optimal selection of features, enabling the reduction of resource expenses during the neural network creation and training phases. The number of neurons in the input layer and their type entirely depend on the finalized dataset being used.

***

### Correlation and Indicator Selection Based on Correlation

The concept of correlation is fundamental in selecting which features to include in the input layer, particularly when dealing with the problem of using highly correlated features.

#### The Concept of Correlation

In mathematical statistics, the **correlation coefficient** is a metric used to quantify the linear relationship or dependence between random variables.

*   The coefficient takes values in the range from **-1 to 1**.
*   A value of **1 indicates a direct relationship** (as one variable increases, the other increases).
*   A value of **-1 indicates an inverse relationship** (as one variable increases, the other decreases).
*   A value of **0 indicates a complete absence of dependence** between the random variables.

The presence of a correlation between features can indicate either a **cause-and-effect relationship** between them or that both variables are **dependent on a common underlying factor**.

#### Selecting Indicators Based on Correlation

Correlation analysis is used primarily to manage redundancy and exclude irrelevant features:

1.  **Excluding Irrelevant Features (Low Correlation):** Indicators showing a **low correlation** (values close to 0) with the expected price movements (target data) can be safely **excluded** from further analysis. For example, data showing a correlation close to 0 with the target data, such as the ATR indicator values, suggest they should be removed.
2.  **Excluding Redundant Features (High Correlation):** The use of **highly correlated features** (values near 1 or -1) poses a significant challenge.
    *   Indicators showing a strong correlation (e.g., coefficient greater than 0.70, or a strong inverse correlation like -0.76) will **only duplicate signals**.
    *   Including strongly correlated indicators complicates the neural network's architecture and maintenance, and their expected impact will be minimal.
    *   Using multiple variables dependent on one factor **exaggerates its impact** on the overall outcome.
    *   This leads to unnecessary neural connections that complicate the model and delay learning.

The process often involves:

*   **Initial screening:** Calculating the correlation of various input data (like indicator values) against the expected target values (e.g., upcoming price movement magnitude or direction).
*   **Redundancy check:** Analyzing the correlation between different indicators. By finding which indicators are highly correlated with others (like correlating Stochastic, CCI, or MFI with RSI), redundant indicators can be excluded from the indicator basket.

#### Expanding Correlation Analysis for Non-Linear Dependencies

Traditional correlation analysis focuses on linear relationships. However, the relationship between input features and target values might be non-linear (e.g., power law or logarithmic).

*   If a relationship is non-linear, the data needs to be **prepared beforehand** (e.g., by exponentiation).
*   Correlation testing is then used to check the usefulness of such transformations.
*   For instance, analyzing the change in correlation when indicator values are set to different degrees can reveal potential opportunities to expand the indicator basket if the correlation with the expected price movement decreases much slower than with the original values.

#### Correlation in Time Series Data

When dealing with time series data, correlation analysis can also be used to determine the necessary **historical depth** to feed into the neural network.

*   Time series exhibit "historical memory," meaning each subsequent value depends on a certain depth of historical values.
*   Testing the correlation of data with a historical shift (time lag) can show how the indicator's influence on the target result decreases as the time lag increases.
*   A rapid decline in correlation up to a certain bar suggests that limiting the analysis depth to that range would be most effective, optimizing the utilization of computational resources.

Selecting features based on correlation acts like a quality control filter for the neural network's diet. It ensures the network receives distinct, non-redundant, and relevant data points, avoiding the inefficiencies caused by feeding it either useless noise (low correlation) or repetitive echoes (high correlation)..