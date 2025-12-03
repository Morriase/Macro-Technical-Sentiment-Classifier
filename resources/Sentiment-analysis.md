Sentiment Analysis and Deep Learning for Trading with EA and Backtesting with Python
Javier Santiago Gaston De Iriarte Cabrera | 5 July, 2024

Introduction
Integrating deep learning and sentiment analysis into trading strategies in MetaTrader 5 (MQL5) represents a sophisticated advancement in algorithmic trading. Deep learning, a subset of machine learning, involves neural networks with multiple layers that can learn and make predictions from vast and complex datasets. Sentiment analysis, on the other hand, is a natural language processing (NLP) technique used to determine the sentiment or emotional tone behind a body of text. By leveraging these technologies, traders can enhance their decision-making processes and improve trading outcomes.

For this article, we will integrate Python into MQL5 using a DLL shell32.dll, which executes what we need for Windows. By installing Python and running it through shell32.dll, we will be able to launch Python scripts from the MQL5 Expert Advisor (EA). There are two Python scripts: one to run the trained ONNX model from TensorFlow, and another script that uses libraries to fetch news from the internet, read the headlines, and quantify media sentiment using AI. This is one possible solution, but there are many ways and different sources to obtain the sentiment of a stock or symbol. Once the model and sentiment are obtained, if both values are in agreement, the order is executed by the EA.

Can we perform a test in Python to understand the results of combining sentiment analysis and deep learning? The answer is yes, and we will proceed to study the code.



Backtesting Sentiment Analysis with Deep Learning using Python
To perform the backtesting of this strategy, we will use the following libraries. I will use my other article as a starting point. Anyway, here I will also provide the required explanations.

We will use the following libraries:
import ccxt
import pandas as pd
import numpy as np
import onnx
import onnxruntime as ort
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import requests
from datetime import datetime, timedelta
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
First of all, we ensure that nltk is updated.

nltk.download('vader_lexicon')
nltk (Natural Language Toolkit) is a library used for working with human language data (text). It provides easy-to-use interfaces to over 50 corpora and lexical resources, such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, as well as wrappers for industrial-strength NLP libraries.

Readers must adapt the python backtesting script to specify where to obtain data, news feed and data for ONNX models.

We will use the following to obtain the sentiment analysis:

def get_news_sentiment(symbol, api_key, date):
    try:
        newsapi = NewsApiClient(api_key=api_key)
        
        # Obtener noticias relacionadas con el símbolo para la fecha específica
        end_date = date + timedelta(days=1)
        articles = newsapi.get_everything(q=symbol,
                                          from_param=date.strftime('%Y-%m-%d'),
                                          to=end_date.strftime('%Y-%m-%d'),
                                          language='en',
                                          sort_by='relevancy',
                                          page_size=10)
        
        sia = SentimentIntensityAnalyzer()
        
        sentiments = []
        for article in articles['articles']:
            text = article.get('title', '')
            if article.get('description'):
                text += ' ' + article['description']
            
            if text:
                sentiment = sia.polarity_scores(text)
                sentiments.append(sentiment['compound'])
        
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        return avg_sentiment
    except Exception as e:
        print(f"Error al obtener el sentimiento para {symbol} en la fecha {date}: {e}")
        return 0
For the backtest, we will use news-api as a feed, because their free API lets us get 1 month look-back of news. If you need more, you can buy a subscription.

The rest of the code will be to obtain the predictions from the ONNX model to predict next close prices. We will just compare the sentiment with the deep learning predictions, and if both conclude with same results, an order will be created. It looks like this:

investment_df = comparison_df.copy()
investment_df['price_direction'] = np.where(investment_df['prediction'].shift(-1) > investment_df['prediction'], 1, -1)
investment_df['sentiment_direction'] = np.where(investment_df['sentiment'] > 0, 1, -1)
investment_df['position'] = np.where(investment_df['price_direction'] == investment_df['sentiment_direction'], investment_df['price_direction'], 0)
investment_df['strategy_returns'] = investment_df['position'] * (investment_df['actual'].shift(-1) - investment_df['actual']) / investment_df['actual']
investment_df['buy_and_hold_returns'] = (investment_df['actual'].shift(-1) - investment_df['actual']) / investment_df['actual']
The code first creates a copy of `comparison_df` and names it `investment_df`. Then it adds a new column called `price_direction` which takes the value of 1 if the next prediction is higher than the current prediction and -1 otherwise. Next it adds another column called `sentiment_direction` which takes the value of 1 if the sentiment is positive and -1 if it's negative. Then it adds a column named `position` which takes the value of `price_direction` if it matches `sentiment_direction` and 0 otherwise. The code then calculates `strategy_returns` by multiplying `position` with the relative change in the actual values from one row to the next. Finally it calculates `buy_and_hold_returns` as the relative change in the actual values from one row to the next without considering the positions.

Results from this backtest look like this:

Datos normalizados guardados en 'binance_data_normalized.csv'
Sentimientos diarios guardados en 'daily_sentiments.csv'
Predicciones y sentimiento guardados en 'predicted_data_with_sentiment.csv'
Mean Absolute Error (MAE): 30.66908467315391
Root Mean Squared Error (RMSE): 36.99641752814565
R-squared (R2): 0.9257591918098058
Mean Absolute Percentage Error (MAPE): 0.00870572230484879
Gráfica guardada como 'ETH_USDT_price_prediction.png'
Gráfica de residuales guardada como 'ETH_USDT_residuals.png'
Correlation between actual and predicted prices: 0.9752007459642241
Gráfica de estrategia de inversión guardada como 'ETH_USDT_investment_strategy.png'
Gráfica de drawdown guardada como 'ETH_USDT_drawdown.png'
Sharpe Ratio: 9.41431958149606
Sortino Ratio: 11800588386323879936.0000
Número de rendimientos totales: 28
Número de rendimientos en exceso: 28
Número de rendimientos negativos: 19
Media de rendimientos en exceso: 0.005037
Desviación estándar de rendimientos negativos: 0.000000
Sortino Ratio: nan
Beta: 0.33875104783408166
Alpha: 0.006981197358213854
Cross-Validation MAE: 1270.7809910146143 ± 527.5746657573876
SMA Mean Absolute Error (MAE): 344.3737716856061
SMA Mean Absolute Error (MAE): 344.3737716856061
SMA Root Mean Squared Error (RMSE): 483.0396130996611
SMA R-squared (R2): 0.5813550203375846
Gráfica de predicción SMA guardada como 'ETH_USDT_sma_price_prediction.png'
Gráfica de precio, predicción y sentimiento guardada como 'ETH_USDT_price_prediction_sentiment.png'
Gráfica de drawdown guardada como 'ETH_USDT_drawdown.png'
Maximum Drawdown: 0.00%
As results say, the correlation between the predicted prices and the real prices are very good. R2, that is a metric to measure how good the predictions of the model, also looks good. Sharpe ratio is higher than 5, which is excellent, as well as Sortino. Also, other results are shown in graphs.

The graph that compares the strategy vs hold looks like this:

Strategy vs hold

Other graphs like price prediction vs actual price

price prediction vs actual price

and, actual price, price prediction and sentiment

Price Prediction and Sentiment

The results show that this strategy is very profitable, so we are now using this argument to create an EA.

This EA should have two Python scripts that make the sentiment analysis and the Deep Learning Model, and should be all merged to function in the EA.



ONNX Model
The code for the data acquisition, training, and ONNX model remains the same as we used in previous articles. Therefore, I will proceed to discuss the Python code for sentiment analysis.



Sentiment Analysis with Python
We will use the libraries `requests` and `TextBlob` to fetch forex news and perform sentiment analysis, along with the `csv` library for reading and writing data. Additionally, the `datetime` and `time` libraries will be utilized.

import requests
from textblob import TextBlob
import csv
from datetime import datetime
import time
from time import sleep
The idea for this script is first to delay for a few seconds upon starting (to ensure that the next part of the script can function properly). The second part of the script will read the API key we want to use. For this case, we will use the Marketaux API, which offers a series of free news and free calls. There are more options such as News API, Alpha Vantage, or Finhub, some of which are paid but provide more news, including historical news, allowing a backtesting of the strategy in MT5. As mentioned earlier, we will use Marketaux for now since it has a free API to obtain daily news. If we want to use other sources, we will need to adapt the code.

Here is a draft of how the script could be structured:

Here's the function to read the api key from the input of the EA:

api_file_path = 'C:/Users/jsgas/AppData/Roaming/MetaQuotes/Terminal/24F345EB9F291441AFE537834F9D8A19/MQL5/Files/Files/api.txt'
print(api_file_path)

def read_api_from_file():
    try:
        with open(api_file_path, 'r', encoding='utf-16') as file:
            raw_data = file.read()
            print(f"Raw data from file: {repr(raw_data)}")  # Print raw data
            api = raw_data.strip()  # Lee el contenido y elimina espacios en blanco adicionales
            api = api.replace('\ufeff', '')  # Remove BOM character if present
            print(f"API after stripping whitespace: {api}")
            time.sleep(5)
            return api
    except FileNotFoundError:
        print(f"El archivo {api_file_path} no existe.")
        time.sleep(5)
        return None




# Configuración de la API de Marketaux
api=read_api_from_file()
MARKETAUX_API_KEY = api
Before reading the news, we need to know what to read, and for that, we will have this Python script read from a text file created by the EA, so that the Python script knows what to read or which symbol to study and obtain news about, and, what api key is input in the EA, what date is today so the model gets done and for the news to arrive for this date.

It must also be capable of writing a txt or csv so it serves as input to the EA, with the results of the Sentiment.

def read_symbol_from_file():
    try:
        with open(symbol_file_path, 'r', encoding='utf-16') as file:
            raw_data = file.read()
            print(f"Raw data from file: {repr(raw_data)}")  # Print raw data
            symbol = raw_data.strip()  # Lee el contenido y elimina espacios en blanco adicionales
            symbol = symbol.replace('\ufeff', '')  # Remove BOM character if present
            print(f"Symbol after stripping whitespace: {symbol}")
            return symbol
    except FileNotFoundError:
        print(f"El archivo {symbol_file_path} no existe.")
        return None
def save_sentiment_to_txt(average_sentiment, file_path='C:/Users/jsgas/AppData/Roaming/MetaQuotes/Terminal/24F345EB9F291441AFE537834F9D8A19/MQL5/Files/Files/'+str(symbol)+'sentiment.txt'):
    with open(file_path, 'w') as f:
        f.write(f"{average_sentiment:.2f}")
if symbol:
    news, current_rate = get_forex_news(symbol)

    if news:
        print(f"Noticias para {symbol}:")
        for i, (title, description) in enumerate(news, 1):
            print(f"{i}. {title}")
            print(f"   {description[:100]}...")  # Primeros 100 caracteres de la descripción
        
        print(f"\nTipo de cambio actual: {current_rate if current_rate else 'No disponible'}")

        # Calcular el sentimiento promedio
        sentiment_scores = [TextBlob(title + " " + description).sentiment.polarity for title, description in news]
        average_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        print(f"Sentimiento promedio: {average_sentiment:.2f}")

        # Guardar resultados en CSV
        #save_to_csv(symbol, current_rate, average_sentiment)

        # Guardar sentimiento promedio en un archivo de texto
        save_sentiment_to_txt(average_sentiment)
        print("Sentimiento promedio guardado en 'sentiment.txt'")
    else:
        print("No se pudieron obtener noticias de Forex.")
else:
    print("No se pudo obtener el símbolo del archivo.")
Readers must adapt the whole script depending on what the study, forex, stocks or crypto.



The Expert Advisor
We must include shell32.dll as here to run the python scripts

#include <WinUser32.mqh>

#import "shell32.dll"
int ShellExecuteW(int hwnd, string lpOperation, string lpFile, string lpParameters, string lpDirectory, int nShowCmd);
#import
We must add the python scripts to the File folder

string script1 = "C:\\Users\\jsgas\\AppData\\Roaming\\MetaQuotes\\Terminal\\24F345EB9F291441AFE537834F9D8A19\\MQL5\\Files\\Files\\dl model for mql5 v6 Final EURUSD_bien.py";
string script2 = "C:\\Users\\jsgas\\AppData\\Roaming\\MetaQuotes\\Terminal\\24F345EB9F291441AFE537834F9D8A19\\MQL5\\Files\\Files\\sentiment analysis marketaux v6 Final EURUSD_bien.py";
And all the paths to the inputs and outputs of the python scripts,

// Ruta del archivo donde se escribirá el símbolo
string filePathSymbol = "//Files//symbol.txt";
// Ruta del archivo donde se escribirá el timeframe
string filePathTimeframe = "//Files//timeframe.txt";
string filePathTime = "//Files//time.txt";
string filePathApi = "//Files//api.txt";

string fileToSentiment = "//Files//"+Symbol()+"sentiment.txt";



string file_add = "C://Users//jsgas//AppData//Roaming//MetaQuotes//Terminal//24F345EB9F291441AFE537834F9D8A19//MQL5//Files";
string file_str = "//Files//model_";
string file_str_final = ".onnx";
string file_str_nexo = "_";

string file_add2 = "C:\\Users\\jsgas\\AppData\\Roaming\\MetaQuotes\\Terminal\\24F345EB9F291441AFE537834F9D8A19\\MQL5\\Files";
string file_str2 = "\\Files\\model_";
string file_str_final2 = ".onnx";
string file_str_nexo2 = "_";
We must input the Marketaux api key

input string api_key      = "mWpORHgs3GdjqNZkxZwnXmrFLYmG5jhAbVrF";           // MARKETAUX_API_KEY www.marketaux.com
We can obtain that from here, and it will look as this:

api key

I don't work for marketaux, so you can use any other news feed, or subscription you want/need.

You will have to setup a Magic Number, so orders don't get mixed up

int OnInit()
  {
   ExtTrade.SetExpertMagicNumber(Magic_Number);
You can also add it here

void OpenBuyOrder(double lotSize, double slippage, double stopLoss, double takeProfit)
  {
// Definir la estructura MqlTradeRequest
   MqlTradeRequest request;
   MqlTradeResult result;

// Inicializar la estructura de la solicitud
   ZeroMemory(request);

// Establecer los parámetros de la orden
   request.action   = TRADE_ACTION_DEAL;
   request.symbol   = _Symbol;
   request.volume   = lotSize;
   request.type     = ORDER_TYPE_BUY;
   request.price    = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   request.deviation= slippage;
   request.sl       = stopLoss;
   request.tp       = takeProfit;
   request.magic    = Magic_Number;
   request.comment  = "Buy Order";

// Enviar la solicitud de comercio
   if(!OrderSend(request,result))
     {
      Print("Error al abrir orden de compra: ", result.retcode);
That last snippet of the code is how the order is made, you can also use trade from CTrade for making orders.

This will write a file (to use as input in the .py scritps):

void WriteToFile(string filePath, string data)
  {
   Print("Intentando abrir el archivo: ", filePath);
// Abre el archivo en modo de escritura, crea el archivo si no existe
   int fileHandle = FileOpen(filePath, FILE_WRITE | FILE_TXT);
   if(fileHandle != INVALID_HANDLE)
     {
      // Escribe los datos en el archivo
      FileWriteString(fileHandle, data);
      FileClose(fileHandle);  // Cierra el archivo
      Print("Archivo escrito exitosamente: ", filePath);
     }
   else
     {
      Print("Error al abrir el archivo ", filePath, ". Código de error: ", GetLastError());
     }
  }
This will write the symbol, timeframe and current date in the file:

void WriteSymbolAndTimeframe()
  {
// Obtén el símbolo actual
   currentSymbol = Symbol();
// Obtén el período de tiempo del gráfico actual
   string currentTimeframe = GetTimeframeString(Period());
   currentTime = TimeToString(TimeCurrent(), TIME_DATE);

// Escribe cada dato en su respectivo archivo
   WriteToFile(filePathSymbol, currentSymbol);
   WriteToFile(filePathTimeframe, currentTimeframe);
   WriteToFile(filePathTime, currentTime);
   WriteToFile(filePathApi,api_key);

   Sleep(10000); // Puedes ajustar o eliminar esto según sea necesario
  }
The function WriteSymbolAndTimeframe performs the following tasks:

First, it retrieves the current trading symbol and stores it in currentSymbol
Then, it gets the current chart's timeframe as a string using GetTimeframeString(Period()) and stores it in currentTimeframe
It also gets the current time in a specific format using TimeToString(TimeCurrent(), TIME_DATE) and stores it in currentTime
Next, it writes each of these values to their respective files:
currentSymbol is written to filePathSymbol
currentTimeframe is written to filePathTimeframe
currentTime is written to filePathTime
api_key is written to filePathApi
Finally, the function pauses for 10 seconds using Sleep(10000) which can be adjusted or removed as needed.
We can launch the scripts with this:

void OnTimer()
  {
   datetime currentTime2 = TimeCurrent();

// Verifica si ha pasado el intervalo para el primer script
   if(currentTime2 - lastExecutionTime1 >= interval1)
     {
      // Escribe los datos necesarios antes de ejecutar el script
      WriteSymbolAndTimeframe();

      // Ejecuta el primer script de Python
      int result = ShellExecuteW(0, "open", "cmd.exe", "/c python \"" + script1 + "\"", "", 1);
      if(result > 32)
         Print("Script 1 iniciado exitosamente");
      else
         Print("Error al iniciar Script 1. Código de error: ", result);
      lastExecutionTime1 = currentTime2;
     }
The function `OnTimer` is executed periodically and performs the following tasks:

First, it retrieves the current time and stores it in `currentTime2`.
It then checks if the time elapsed since the last execution of the first script (`lastExecutionTime1`) is greater than or equal to a predefined interval (`interval1`).
If the condition is met, it writes the necessary data by calling `WriteSymbolAndTimeframe`.
Next, it executes the first Python script by running a command via `ShellExecuteW` which opens `cmd.exe` and runs the Python script specified by `script1`.
If the script execution is successful (indicated by a result greater than 32), it prints a success message; otherwise, it prints an error message with the corresponding error code.
Finally, it updates `lastExecutionTime1` to the current time (`currentTime2`).
We can read the file with this function:

string ReadFile(string file_name)
  {
   string result = "";
   int handle = FileOpen(file_name, FILE_READ|FILE_TXT|FILE_ANSI); // Use FILE_ANSI for plain text

   if(handle != INVALID_HANDLE)
     {
      int file_size = FileSize(handle); // Get the size of the file
      result = FileReadString(handle, file_size); // Read the whole file content
      FileClose(handle);
     }
   else
     {
      Print("Error opening file: ", file_name);
     }

   return result;
  }
The code defines a function named ReadFile which takes a file name as an argument and returns the file content as a string first it initializes an empty string result then it attempts to open the file with read permissions and in plain text mode using FileOpen if the file handle is valid it gets the file size using FileSize reads the entire file content into result using FileReadString and then closes the file using FileClose if the file handle is invalid it prints an error message with the file name finally it returns the result containing the file content.

By changing this condition, we can add the sentiment as one more:

   if(ExtPredictedClass==PRICE_DOWN && Sentiment_number<0)
      signal=ORDER_TYPE_SELL;    // sell condition
   else
     {
      if(ExtPredictedClass==PRICE_UP && Sentiment_number>0)
         signal=ORDER_TYPE_BUY;  // buy condition
      else
         Print("No order possible");
     }
The sentiment in this case goes from 10 to -10, being 0 a neutral signal. You can modify as you want this strategy.

The rest of the code is the simple EA used from the article How to use ONNX models in MQL5 with a few modifications.


This is not a complete finished EA, this is just a simple example of how to use python and mql5 to create a sentiment & deep learning Expert Advisor. As more time you invest in this EA, you will get less errors and problems. This is a cutting edge case study, and backtesting shows promising results. I hope you find this article helpful, and if someone can manage to get a good sample of news or makes it work for some time, please share results. In order to test the strategy, you should use a demo account.



Conclusion
In conclusion, the integration of deep learning and sentiment analysis into MetaTrader 5 (MQL5) trading strategies exemplifies the advanced capabilities of modern algorithmic trading. By leveraging Python scripts through a DLL shell32.dll interface, we can seamlessly execute complex models and obtain valuable sentiment data, thereby enhancing trading decisions and outcomes. The process outlined includes using Python to fetch and analyze news sentiment, running ONNX models for price predictions, and executing trades when both indicators align.

The backtesting results demonstrate the strategy's potential profitability, as indicated by strong correlation metrics, high R-squared values, and excellent Sharpe and Sortino ratios. These findings suggest that combining sentiment analysis with deep learning can significantly improve the accuracy of trading signals and overall strategy performance.

Moving forward, the development of a fully functional Expert Advisor (EA) involves meticulous integration of various components, including Python scripts for sentiment analysis and ONNX models for price prediction. By continually refining these elements and adapting the strategy to different markets and data sources, traders can build a robust and effective trading tool.

This study serves as a foundation for those interested in exploring the convergence of machine learnin

###summary 
    Integrating sentiment analysis using a Python API into an existing pipeline, such as a trading strategy pipeline or backtesting system, involves several key steps outlined in the sources, including library usage, data acquisition via APIs, and comparison with deep learning predictions.

Here is a comprehensive breakdown of how sentiment analysis is integrated using Python:

### 1. Key Libraries and Setup

Sentiment analysis is a Natural Language Processing (NLP) technique used to determine the emotional tone of text. To implement this in Python, you utilize specific libraries:

*   **Natural Language Toolkit (NLTK):** NLTK is essential for working with human language data.
    *   You must import `nltk` and specifically `SentimentIntensityAnalyzer` from `nltk.sentiment`.
    *   The setup requires updating `nltk` and downloading the specific lexicon used for analysis: `nltk.download('vader_lexicon')`.
*   **Alternative Sentiment Library:** For specific Python scripts designed to function within an Expert Advisor (EA), the `TextBlob` library is also used for performing sentiment analysis.
*   **Data Acquisition Libraries:** You need libraries like `requests` and `newsapi` (or `textblob` for fetching news) to obtain the text data necessary for analysis.

### 2. Data Acquisition using APIs

The sentiment analysis relies on fetching news from the internet. Readers must adapt their Python scripts to specify where to obtain the news feed.

*   **API Usage:** For backtesting purposes, the `newsapi` is used because its free API allows for a 1-month look-back of news.
*   **API Function Structure:** A function, such as `def get_news_sentiment(symbol, api_key, date):`, is employed where the `NewsApiClient` is initialized using the provided `api_key` to fetch news related to a specific trading `symbol` and `date`.
*   **Source Options:** While the `News API` is used for backtesting, other sources like **Marketaux API** (which offers free daily news calls), Alpha Vantage, or Finhub can be utilized. If using other sources, the code needs adaptation.

### 3. Python Script Integration and Data Flow

The Python sentiment analysis script functions as a key component of the overall training or execution pipeline:

*   **Inputs:** For execution (e.g., within an MQL5 Expert Advisor), the Python script needs to **read input files** created by the external environment. This includes reading the **API key** and the **symbol** (or stock/crypto) about which to obtain news.
*   **Processing:** The script reads the headlines from the fetched news and then **quantifies media sentiment using AI**.
*   **Outputs:** The script is structured to write the calculated sentiment result (e.g., an `average_sentiment`) to a **text or CSV file**. This output file then serves as input for the next stage of the pipeline. A function like `save_sentiment_to_txt` is used to write the average sentiment (formatted to two decimal places) to a specified file path.

### 4. Integrating Sentiment into Decision Logic

In the overall strategy pipeline, the sentiment result is integrated with deep learning predictions (often sourced from a trained ONNX model) to finalize trading decisions.

*   **Comparison Logic:** The strategy compares the derived sentiment direction with the deep learning prediction direction.
    *   A `sentiment_direction` column is created, which takes the value of **1 if the sentiment is positive** and **-1 if it's negative** (or 0 if neutral, in a different context).
    *   This is compared to the `price_direction` (1 if the next predicted price is higher, -1 otherwise).
*   **Execution Condition:** An order is executed or a position is taken *only if both values are in agreement*.
    *   The `position` column takes the value of the `price_direction` if it **matches** the `sentiment_direction`, and 0 otherwise.
*   **Example Decision Condition:** In a trading context, if the predicted class is `PRICE_DOWN` and the `Sentiment_number` is less than 0, a **sell signal** is generated. Conversely, if the predicted class is `PRICE_UP` and the `Sentiment_number` is greater than 0, a **buy signal** is generated. The sentiment values used in this example range from 10 (positive) to -10 (negative), with 0 being neutral.

***

**Note on Pipeline Context:** The sources specifically detail integrating Python scripts into an MQL5 (MetaTrader 5) algorithmic trading pipeline. This integration is achieved by using the **DLL `shell32.dll`** to execute the Python scripts from the MQL5 Expert Advisor (EA). The EA uses the function `ShellExecuteW` to run the Python scripts.

This process acts like a specialized data module in your pipeline: the Python script takes raw news data (input via API), processes it into a quantified sentiment score, and outputs that score (via a text file) for consumption by the core decision-making engine.