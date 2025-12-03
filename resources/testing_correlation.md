Identifying strongly correlated indicators with price movements is a crucial preparatory step in building an efficient neural network model, especially for complex architectures like the Long Short-Term Memory (LSTM) network designed for time series data. This process, carried out practically within the MQL5 environment, involves statistical analysis to ensure the inputs are relevant and non-redundant.

The selection of indicators and the required historical data depth are finalized during this preparatory phase, which significantly reduces expenses during the neural network creation and training stages.

### Practical Steps to Identify Strong Correlations in MQL5

The practical analysis involves creating an MQL5 script to perform Pearson correlation calculations, comparing potential input data (indicators) against expected outcomes (target values, often based on future price movements).

#### 1. Setup and Data Acquisition

1.  **Include the Statistical Library:** The process requires utilizing the Mathematical Statistics Library from the MetaTrader 5 platform, which is included using: `#include <Math\Stat\Math.mqh>`.
2.  **Obtain Indicator Handles:** Handles must be acquired for all technical indicators intended for testing (e.g., RSI, MACD, CCI, ZigZag). This is done using functions like `iRSI`, `iMACD`, or `iCustom` (for ZigZag).
3.  **Load Historical Data and Indicators:** Historical price data (e.g., `CopyClose`) and indicator values (e.g., `CopyBuffer` using the acquired handles) are loaded into dynamic arrays (`double[]`) over a specified period.
4.  **Data Preprocessing:** It is necessary to move away from absolute price values and toward a relative range. For instance, instead of using the candlestick closing price, you might use the difference between the opening and closing prices (candlestick body size) as an indicator of price movement intensity.

#### 2. Defining Target Values

To calculate the correlation, the inputs (indicators) must be compared against specific, measurable target outcomes. A common approach for forecasting market movement uses the ZigZag indicator to identify historical extremes:

1.  **Direction of Movement (Classification Target):** The direction to the nearest future extreme (e.g., "Buy" or "Sell"). This is typically assigned binary values, such as 1 for bullish patterns and -1 for bearish patterns.
2.  **Magnitude of Movement (Regression Target):** The expected strength or distance to that nearest future extreme point.

The target values are calculated by iterating through the historical data (often in reverse order, from newest to oldest) and measuring the deviation of the last recorded extremum (from ZigZag) from the bar's closing price.

#### 3. Calculating Correlation Coefficients

The **Pearson correlation coefficient** is calculated using the `MathCorrelationPearson` function from the MQL5 standard statistical analysis library. This function returns a value between -1 and 1.

$$ \text{Correlation(Indicator, Target)} = \text{MathCorrelationPearson(indicator array, target array, result)} $$

*   **Excluding Irrelevant Features:** Indicators showing a **low correlation** (values close to 0) with the target price movements should be safely **excluded** from further analysis.
*   **Excluding Redundant Features:** Indicators showing a **strong correlation** (e.g., coefficient greater than 0.70, or a strong inverse correlation like -0.76) with *other inputs* (indicator-to-indicator correlation) should be excluded. Using highly correlated features only duplicates signals, complicates the neural network's architecture, and delays learning. For example, Stochastic, CCI, and MFI showed a strong correlation (above 0.70) with RSI and were thus excluded in one analysis.

***

### Selecting Inputs for an LSTM Neural Network

The correlation analysis must be extended to account for the unique characteristics of time series processing by Recurrent Neural Networks (RNNs), such as the LSTM block, which relies on sequenced data and "historical memory".

#### Determining Optimal Historical Depth (Time Shift)

Traditional correlation analysis helps select relevant features, but for time series inputs, determining the optimal number of past bars to feed the network is critical. This is achieved through **correlation testing with a historical shift** (time lag):

1.  A modified correlation function (`ShiftCorrelation`) is used to test how the indicator's correlation with the target result changes as the time lag increases.
2.  A rapid decline in correlation as the time shift increases indicates that older data points are losing their predictive power.
3.  The analysis might show, for example, that the correlation of RSI with the target decreases rapidly up to the **30th bar**. This suggests that limiting the analysis depth to 30 bars would be most effective, thereby optimizing resource utilization.

#### Preparing Inputs for the LSTM Block in MQL5

After selecting the final indicator set and the historical depth, the data must be formatted correctly for the LSTM neural layer (`CNeuronLSTM`):

1.  **Tensor Dimensions:** The input data to the recurrent layer must be structured as a sequence. The total size of the input layer in the network architecture is determined by the product of the number of historical bars to analyze (`BarsToPattern` or optimal depth) multiplied by the number of selected indicators per bar (`NeuronsToBar`).
2.  **Chronological Ordering:** Unlike other neural network types where data patterns can often be shuffled randomly during training (to generalize features), recurrent models are highly sensitive to the temporal order. Therefore, the **chronological sequence** of the inputted raw data must be strictly maintained for accurate training and operation.
3.  **Data Flow:** In an MQL5 Expert Advisor template designed for the LSTM, the indicator data for the chosen historical depth is loaded into a dedicated data buffer (`CBufferType *input_data`) following the sequence established during the model training. This buffer is then passed to the network's `FeedForward` method.

Using correlation analysis in this meticulous way ensures that the LSTM network, when constructed using MQL5 (which supports building an LSTM block, see Chapter 4.2.2), receives a compact, relevant, and non-redundant signal sequence, maximizing its ability to learn time-dependent patterns.