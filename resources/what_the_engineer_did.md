The author explicitly defined the neural network's target variables using the ZigZag indicator to convert the ambiguous goal of "forecasting future movement" into specific, measurable outcomes that fit a **regression task**,.

Here is a detailed breakdown of the target definition, ZigZag parameters, and prediction horizon based on the sources:

### Q1.1: How exactly did he define the target using ZigZag?

The target was defined as forecasting **two specific values** for the upcoming price movement:

1.  **Direction of Movement (Target 1):** This determines the probable direction of the upcoming price movement (Buy or Sell),.
    *   This aspect of the task is often regarded as a binary classification problem (upward movement/Buy or downward movement/Sell).
    *   The direction was derived from the sign of the magnitude calculation: a positive sign (or value $\ge 0$) indicated a buy direction, while a negative sign indicated a sell direction,.
2.  **Expected Strength/Magnitude of Movement (Target 2):** This provides a **quantitative assessment** of the expected price movement to the nearest extreme.
    *   The magnitude was calculated by taking the difference between the price value of the last identified extremum (peak/trough) and the closing price of the analyzed bar,.
    *   This part of the task is specifically handled as a **regression task**.

The neural network model's overall output layer contained **two neurons** to forecast these two movement target values (direction and expected strength).

### Q1.2: What were the ZigZag parameters?

The author specified the following fixed parameters for the ZigZag indicator when generating the training targets,:

*   **Timeframe:** M5 (5-minute timeframe).
*   **Depth (Candlesticks to search for extrema):** $\mathbf{48}$.
    *   This was selected to reflect **four-hour extrema**, given that 48 M5 bars equal 4 hours.
*   **Deviation (Minimum distance between neighboring extrema):** **1 point**.
*   **Backstep (Minimum distance between neighboring extrema in candlesticks):** $\mathbf{47}$.

The target values were calculated by comparing the closing price of the current bar against the most recently recorded ZigZag extremum,.

### Q1.3: How far ahead was the prediction horizon?

The prediction horizon focused on the **next ZigZag reversal point**, specifically predicting the **direction and distance to the nearest future extremum**,.

*   **Mechanism:** The author iterated backward through the historical data, saving the price value of the most recently found extremum (`extremum` variable),.
*   **Target Creation:** For each bar, the model was taught to determine the potential direction and strength of the price movement required to reach that recorded **nearest future extremum**,.
*   **Data Integrity:** The calculation required checking that the extremum always occurred *after* the analyzed closing price, confirming the focus on the **upcoming** movement relative to the bar being analyzed,.

This means the prediction horizon was dynamic, spanning the duration of the current swing until the next projected ZigZag reversal (trough or peak).

***Question 2
The author conducted extensive feature engineering and correlation analysis to select a refined set of indicators for the perceptron test, focusing on using non-redundant and relevant data.

Here is a detailed breakdown of how the indicators were calculated and used:

### Q2.1: How exactly did he calculate RSI to get 40% correlation?

The sources indicate that the **RSI (Relative Strength Index)** was selected because it showed the **highest overall correlation** with the expected price movement, achieving **0.40 for direction** and **0.22 for magnitude**.

*   **RSI Parameters:** The specific parameters used during the initial screening of the RSI were **$\mathbf{12}$** (period) and **$\mathbf{PRICE\_TYPICAL}$** (price type), as defined in the MQL5 function call: `iRSI(_Symbol,PERIOD_M5,12,PRICE_TYPICAL)`.
*   **Normalization Used for Training:** The RSI indicator inherently normalizes its values within a range from 0 to 100. For training the neural network, the author further processed these values to fit the $\mathbf{[-1, 1]}$ range and center them around zero. This was done using the formula: $\mathbf{(RSI_{i} - 50.0) / 50.0}$.
*   **Divergence:** The sources do not mention using RSI divergence in the analysis; the focus was on the indicator's raw correlation with target movement.

### Q2.2: For MACD, what exactly did he use?

The author utilized a **derived metric** from the MACD indicator rather than the main line, signal line, or histogram individually.

*   **Specific Metric Chosen:** The chosen metric was the **absolute difference between the MACD Main and Signal lines** (MACD Main-Signal).
*   **Rationale:** This difference was found to be **highly useful** and demonstrated a strong correlation with the target price data, confirming the effectiveness of incorporating classical indicator signals. It was deemed **more relevant** than the individual MACD line values.
*   **Parameters:** The parameters used for the MACD during indicator data acquisition were: `iMACD(_Symbol,PERIOD_M5,12,48,12,PRICE_TYPICAL)`. This corresponds to:
    *   Fast EMA Period: **$\mathbf{12}$**
    *   Slow EMA Period: **$\mathbf{48}$**
    *   Signal Period: **$\mathbf{12}$**

### Q2.3: Did he use RAW indicator values or DERIVED features?

The approach involved converting raw indicator data into **derived features** and carefully selecting them to maximize relevance and minimize redundancy:

*   **Derived Features (Chosen):**
    *   **RSI Value:** The raw RSI value was normalized (mean centered, scaled to $[-1, 1]$).
    *   **MACD Difference:** The absolute difference between the MACD Main and Signal lines was used, and this derived value was then normalized.
*   **Raw and Derived Features (Excluded due to Low Correlation):** Indicators showing correlation values close to zero were excluded as irrelevant. These included **ATR indicator values** and **Candlestick deviations** (High – Close and Close – Low).
*   **Raw Features (Excluded due to Redundancy):** Indicators that were highly correlated with the chosen RSI (coefficient greater than 0.70) were excluded, as they would **only duplicate signals**. These included Stochastic, CCI, MFI, and the three Bollinger Bands lines.
*   **Non-Linear Transformations:** The author tested non-linear transformations by **raising indicator values to different exponents** to check for potential improvements in correlation, suggesting an awareness of non-linear dependencies. This process confirmed that while correlation with original values decreased, the correlation with expected price movement decreased slower, suggesting an opportunity to expand the input feature set.

In summary, the author primarily used **normalized raw indicator values** (RSI) and a **normalized derived metric** (MACD difference) as input features, having rigorously filtered out less correlated and redundant signals.

*** Question 3-5
The author approached the development and testing of the LSTM model by focusing on rigorous data preparation, specific architectural choices, and iterative training strategies.

### 3. Data Preprocessing

**Q3.1: What timeframe did he use?**

The author primarily used the **M5 (5-minute timeframe)** for loading and analyzing historical data for training the neural network models.

**Q3.2: How did he handle the data?**

Data handling was comprehensive, prioritizing relevance and consistency:

*   **Normalization/Standardization:** Features were **normalized/standardized** to ensure comparable statistical characteristics. The resulting output range was typically aimed at **[-1, 1] centered around zero**. For the RSI indicator, which ranges from 0 to 100, the normalization formula used was **(RSI - 50.0) / 50.0**.
*   **Absolute vs. Relative Values:** The approach intentionally moved **away from absolute price values** to a relative range. Key input features were derived values, such as the **difference between the MACD Main and Signal lines** and the **difference between the candlestick's Close and Open prices** (candlestick body size), which indicates intensity of movement.
*   **Feature Filtering:** The dataset was refined by excluding irrelevant and redundant features based on correlation analysis.
    *   Indicators showing near-zero correlation (e.g., **ATR values** and **Candlestick deviations**) were deemed irrelevant and excluded.
    *   Indicators showing **high correlation (greater than 0.70)** with selected features like RSI (e.g., Stochastic, CCI, MFI, Bollinger Bands) were excluded to manage redundancy and simplify the architecture.

### 4. LSTM Architecture

**Q4.1: What was his sequence length (How many bars did LSTM look back)?**

The typical sequence length analyzed was **40 historical bars**. This resulted in an input layer size of **160 neurons** (40 historical bars $\times$ 4 indicators per bar). Additionally, the internal historical depth ($m\_iDepth$) for unrolling the recurrent network during backpropagation was specifically set to **5 iterations**.

**Q4.2: What was his LSTM structure?**

The architecture centered around a core recurrent unit size and layered internal structure:

*   **Units/Hidden Units:** The recurrent LSTM layer utilized **40 units (neurons)**. The four internal neural layers of the LSTM block (Forget Gate, Input Gate, New Content, Output Gate) all contained the **same number of neurons** equal to the output size, which was **40 neurons**.
*   **Activation Functions:** The **Sigmoid function** was used for the internal gates, while the **Hyperbolic Tangent (Tanh)** function was used for the New Content layer and the overall output layer.
*   **Dropout Rate:** When Dropout was tested as a regularization method (often replacing Batch Normalization), a common rate used was **0.3**.

### 5. Training Strategy

**Q5.1: How did he handle class imbalance?**

The sources do not explicitly detail the use of methods like class weighting, oversampling, or undersampling to handle potential class imbalance between bullish and bearish ZigZag swings (Q5.1).

**Q5.2: Did he filter training samples?**

Yes, the training sample was implicitly filtered by the process of defining the target variables:

*   **Noise Reduction:** The author relied on the **ZigZag indicator** during target creation specifically to filter out **small price fluctuations considered noise** and focus only on periods defined by the most **significant extremes**.
*   **Focus on Swings:** By calculating the target based on the difference between the closing price and the **nearest future ZigZag extremum**, the training process inherently focused the model on learning patterns that precede or occur within these significant swings.
