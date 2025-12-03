//+------------------------------------------------------------------+
//|                                      Correlation Relationship.mq5 |
//|                                                            Auron |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Auron Automations"
#property link      "https://www.auronautomations.app"
#property version   "1.00"

#include <Math\Stat\Math.mqh>

input datetime Start = D'2015.01.01 00:00:00';
input datetime End   = D'2025.11.25 23:59:00';

input ENUM_TIMEFRAMES selectTimeframe = PERIOD_M5;
input int shift = 1;            // number of shifts to test (integer)

//+------------------------------------------------------------------+
//| Function to calculate correlation coefficient between two arrays |
//+------------------------------------------------------------------+
double CalculateCorrelation(const double &x[], const double &y[])
  {
   int n = ArraySize(x);
   if(n != ArraySize(y) || n < 2) return 0.0;

   double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0, sum_y2 = 0.0;

   for(int i = 0; i < n; i++)
     {
      sum_x += x[i];
      sum_y += y[i];
      sum_xy += x[i] * y[i];
      sum_x2 += x[i] * x[i];
      sum_y2 += y[i] * y[i];
     }

   double numerator = n * sum_xy - sum_x * sum_y;
   double denominator = sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));

   if(denominator == 0.0) return 0.0;

   return numerator / denominator;
  }

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
   //--- create indicator handles for training pipeline indicators
   int h_EMA50  = iMA(_Symbol, selectTimeframe, 50, 0, MODE_EMA, PRICE_CLOSE);
   int h_EMA100 = iMA(_Symbol, selectTimeframe, 100, 0, MODE_EMA, PRICE_CLOSE);
   int h_EMA200 = iMA(_Symbol, selectTimeframe, 200, 0, MODE_EMA, PRICE_CLOSE);
   int h_RSI    = iRSI(_Symbol, selectTimeframe, 14, PRICE_CLOSE);
   int h_ATR    = iATR(_Symbol, selectTimeframe, 14);
   int h_STOCH  = iStochastic(_Symbol, selectTimeframe, 14, 3, 3, MODE_LWMA, STO_LOWHIGH);
   int h_MACD   = iMACD(_Symbol, selectTimeframe, 12, 26, 9, PRICE_CLOSE);
   int h_BB     = iBands(_Symbol, selectTimeframe, 20, 0, 2, PRICE_CLOSE);

   //--- get OHLC via CopyRates
   MqlRates rates[];
   int copiedRates = CopyRates(_Symbol, selectTimeframe, Start, End, rates);
   if(copiedRates <= 0)
     {
      PrintFormat("CopyRates failed or returned 0 bars (%s %d)", _Symbol, selectTimeframe);
      if(h_EMA50 != INVALID_HANDLE) IndicatorRelease(h_EMA50);
      if(h_EMA100 != INVALID_HANDLE) IndicatorRelease(h_EMA100);
      if(h_EMA200 != INVALID_HANDLE) IndicatorRelease(h_EMA200);
      if(h_RSI != INVALID_HANDLE) IndicatorRelease(h_RSI);
      if(h_ATR != INVALID_HANDLE) IndicatorRelease(h_ATR);
      if(h_STOCH != INVALID_HANDLE) IndicatorRelease(h_STOCH);
      if(h_MACD != INVALID_HANDLE) IndicatorRelease(h_MACD);
      if(h_BB != INVALID_HANDLE) IndicatorRelease(h_BB);
      return;
     }

   //--- prepare buffers for indicators
   int total = ArraySize(rates);
   double ema50[], ema100[], ema200[], rsi[], atr[], stoch_k[], stoch_d[], macd_main[], macd_signal[], bb_main[], bb_up[], bb_low[];

   ArrayResize(ema50, total);
   ArrayResize(ema100, total);
   ArrayResize(ema200, total);
   ArrayResize(rsi, total);
   ArrayResize(atr, total);
   ArrayResize(stoch_k, total);
   ArrayResize(stoch_d, total);
   ArrayResize(macd_main, total);
   ArrayResize(macd_signal, total);
   ArrayResize(bb_main, total);
   ArrayResize(bb_up, total);
   ArrayResize(bb_low, total);

   //--- copy buffers from indicator handles
   if(h_EMA50 == INVALID_HANDLE || CopyBuffer(h_EMA50, 0, 0, total, ema50) <= 0 ||
      h_EMA100 == INVALID_HANDLE || CopyBuffer(h_EMA100, 0, 0, total, ema100) <= 0 ||
      h_EMA200 == INVALID_HANDLE || CopyBuffer(h_EMA200, 0, 0, total, ema200) <= 0 ||
      h_RSI == INVALID_HANDLE || CopyBuffer(h_RSI, 0, 0, total, rsi) <= 0 ||
      h_ATR == INVALID_HANDLE || CopyBuffer(h_ATR, 0, 0, total, atr) <= 0 ||
      h_STOCH == INVALID_HANDLE || CopyBuffer(h_STOCH, MAIN_LINE, 0, total, stoch_k) <= 0 ||
      CopyBuffer(h_STOCH, SIGNAL_LINE, 0, total, stoch_d) <= 0 ||
      h_MACD == INVALID_HANDLE || CopyBuffer(h_MACD, MAIN_LINE, 0, total, macd_main) <= 0 ||
      CopyBuffer(h_MACD, SIGNAL_LINE, 0, total, macd_signal) <= 0 ||
      h_BB == INVALID_HANDLE || CopyBuffer(h_BB, BASE_LINE, 0, total, bb_main) <= 0 ||
      CopyBuffer(h_BB, UPPER_BAND, 0, total, bb_up) <= 0 ||
      CopyBuffer(h_BB, LOWER_BAND, 0, total, bb_low) <= 0)
     {
      Print("One or more indicator buffers failed to copy. Check indicator handles / parameters.");
      if(h_EMA50 != INVALID_HANDLE) IndicatorRelease(h_EMA50);
      if(h_EMA100 != INVALID_HANDLE) IndicatorRelease(h_EMA100);
      if(h_EMA200 != INVALID_HANDLE) IndicatorRelease(h_EMA200);
      if(h_RSI != INVALID_HANDLE) IndicatorRelease(h_RSI);
      if(h_ATR != INVALID_HANDLE) IndicatorRelease(h_ATR);
      if(h_STOCH != INVALID_HANDLE) IndicatorRelease(h_STOCH);
      if(h_MACD != INVALID_HANDLE) IndicatorRelease(h_MACD);
      if(h_BB != INVALID_HANDLE) IndicatorRelease(h_BB);
      return;
     }

   //--- list of indicators and their arrays
   string indicator_names[] = {"EMA50", "EMA100", "EMA200", "RSI", "ATR", "Stoch_K", "Stoch_D", "MACD_Main", "MACD_Signal", "BB_Main", "BB_Upper", "BB_Lower"};
   double indicator_arrays[][1]; // dummy, we'll use pointers or something, but in MQL5, better to use array of arrays
   // Actually, MQL5 doesn't support array of arrays easily, so we'll hardcode the correlations

   //--- open CSV file for writing correlations
   int handle = FileOpen("training_indicators_correlations.csv",
                         FILE_WRITE | FILE_CSV | FILE_ANSI,
                         ',',
                         CP_UTF8);

   if(handle == INVALID_HANDLE)
     {
      Print("Failed to open training_indicators_correlations.csv for writing");
      if(h_EMA50 != INVALID_HANDLE) IndicatorRelease(h_EMA50);
      if(h_EMA100 != INVALID_HANDLE) IndicatorRelease(h_EMA100);
      if(h_EMA200 != INVALID_HANDLE) IndicatorRelease(h_EMA200);
      if(h_RSI != INVALID_HANDLE) IndicatorRelease(h_RSI);
      if(h_ATR != INVALID_HANDLE) IndicatorRelease(h_ATR);
      if(h_STOCH != INVALID_HANDLE) IndicatorRelease(h_STOCH);
      if(h_MACD != INVALID_HANDLE) IndicatorRelease(h_MACD);
      if(h_BB != INVALID_HANDLE) IndicatorRelease(h_BB);
      return;
     }

   //--- write header
   FileWrite(handle, "Indicator1", "Indicator2", "Correlation");

   //--- compute and write correlations
   // EMA50 vs others
   FileWrite(handle, "EMA50", "EMA100", DoubleToString(CalculateCorrelation(ema50, ema100), 5));
   FileWrite(handle, "EMA50", "EMA200", DoubleToString(CalculateCorrelation(ema50, ema200), 5));
   FileWrite(handle, "EMA50", "RSI", DoubleToString(CalculateCorrelation(ema50, rsi), 5));
   FileWrite(handle, "EMA50", "ATR", DoubleToString(CalculateCorrelation(ema50, atr), 5));
   FileWrite(handle, "EMA50", "Stoch_K", DoubleToString(CalculateCorrelation(ema50, stoch_k), 5));
   FileWrite(handle, "EMA50", "Stoch_D", DoubleToString(CalculateCorrelation(ema50, stoch_d), 5));
   FileWrite(handle, "EMA50", "MACD_Main", DoubleToString(CalculateCorrelation(ema50, macd_main), 5));
   FileWrite(handle, "EMA50", "MACD_Signal", DoubleToString(CalculateCorrelation(ema50, macd_signal), 5));
   FileWrite(handle, "EMA50", "BB_Main", DoubleToString(CalculateCorrelation(ema50, bb_main), 5));
   FileWrite(handle, "EMA50", "BB_Upper", DoubleToString(CalculateCorrelation(ema50, bb_up), 5));
   FileWrite(handle, "EMA50", "BB_Lower", DoubleToString(CalculateCorrelation(ema50, bb_low), 5));

   // EMA100 vs others (skip EMA50)
   FileWrite(handle, "EMA100", "EMA200", DoubleToString(CalculateCorrelation(ema100, ema200), 5));
   FileWrite(handle, "EMA100", "RSI", DoubleToString(CalculateCorrelation(ema100, rsi), 5));
   FileWrite(handle, "EMA100", "ATR", DoubleToString(CalculateCorrelation(ema100, atr), 5));
   FileWrite(handle, "EMA100", "Stoch_K", DoubleToString(CalculateCorrelation(ema100, stoch_k), 5));
   FileWrite(handle, "EMA100", "Stoch_D", DoubleToString(CalculateCorrelation(ema100, stoch_d), 5));
   FileWrite(handle, "EMA100", "MACD_Main", DoubleToString(CalculateCorrelation(ema100, macd_main), 5));
   FileWrite(handle, "EMA100", "MACD_Signal", DoubleToString(CalculateCorrelation(ema100, macd_signal), 5));
   FileWrite(handle, "EMA100", "BB_Main", DoubleToString(CalculateCorrelation(ema100, bb_main), 5));
   FileWrite(handle, "EMA100", "BB_Upper", DoubleToString(CalculateCorrelation(ema100, bb_up), 5));
   FileWrite(handle, "EMA100", "BB_Lower", DoubleToString(CalculateCorrelation(ema100, bb_low), 5));

   // EMA200 vs others
   FileWrite(handle, "EMA200", "RSI", DoubleToString(CalculateCorrelation(ema200, rsi), 5));
   FileWrite(handle, "EMA200", "ATR", DoubleToString(CalculateCorrelation(ema200, atr), 5));
   FileWrite(handle, "EMA200", "Stoch_K", DoubleToString(CalculateCorrelation(ema200, stoch_k), 5));
   FileWrite(handle, "EMA200", "Stoch_D", DoubleToString(CalculateCorrelation(ema200, stoch_d), 5));
   FileWrite(handle, "EMA200", "MACD_Main", DoubleToString(CalculateCorrelation(ema200, macd_main), 5));
   FileWrite(handle, "EMA200", "MACD_Signal", DoubleToString(CalculateCorrelation(ema200, macd_signal), 5));
   FileWrite(handle, "EMA200", "BB_Main", DoubleToString(CalculateCorrelation(ema200, bb_main), 5));
   FileWrite(handle, "EMA200", "BB_Upper", DoubleToString(CalculateCorrelation(ema200, bb_up), 5));
   FileWrite(handle, "EMA200", "BB_Lower", DoubleToString(CalculateCorrelation(ema200, bb_low), 5));

   // RSI vs others
   FileWrite(handle, "RSI", "ATR", DoubleToString(CalculateCorrelation(rsi, atr), 5));
   FileWrite(handle, "RSI", "Stoch_K", DoubleToString(CalculateCorrelation(rsi, stoch_k), 5));
   FileWrite(handle, "RSI", "Stoch_D", DoubleToString(CalculateCorrelation(rsi, stoch_d), 5));
   FileWrite(handle, "RSI", "MACD_Main", DoubleToString(CalculateCorrelation(rsi, macd_main), 5));
   FileWrite(handle, "RSI", "MACD_Signal", DoubleToString(CalculateCorrelation(rsi, macd_signal), 5));
   FileWrite(handle, "RSI", "BB_Main", DoubleToString(CalculateCorrelation(rsi, bb_main), 5));
   FileWrite(handle, "RSI", "BB_Upper", DoubleToString(CalculateCorrelation(rsi, bb_up), 5));
   FileWrite(handle, "RSI", "BB_Lower", DoubleToString(CalculateCorrelation(rsi, bb_low), 5));

   // ATR vs others
   FileWrite(handle, "ATR", "Stoch_K", DoubleToString(CalculateCorrelation(atr, stoch_k), 5));
   FileWrite(handle, "ATR", "Stoch_D", DoubleToString(CalculateCorrelation(atr, stoch_d), 5));
   FileWrite(handle, "ATR", "MACD_Main", DoubleToString(CalculateCorrelation(atr, macd_main), 5));
   FileWrite(handle, "ATR", "MACD_Signal", DoubleToString(CalculateCorrelation(atr, macd_signal), 5));
   FileWrite(handle, "ATR", "BB_Main", DoubleToString(CalculateCorrelation(atr, bb_main), 5));
   FileWrite(handle, "ATR", "BB_Upper", DoubleToString(CalculateCorrelation(atr, bb_up), 5));
   FileWrite(handle, "ATR", "BB_Lower", DoubleToString(CalculateCorrelation(atr, bb_low), 5));

   // Stoch_K vs others
   FileWrite(handle, "Stoch_K", "Stoch_D", DoubleToString(CalculateCorrelation(stoch_k, stoch_d), 5));
   FileWrite(handle, "Stoch_K", "MACD_Main", DoubleToString(CalculateCorrelation(stoch_k, macd_main), 5));
   FileWrite(handle, "Stoch_K", "MACD_Signal", DoubleToString(CalculateCorrelation(stoch_k, macd_signal), 5));
   FileWrite(handle, "Stoch_K", "BB_Main", DoubleToString(CalculateCorrelation(stoch_k, bb_main), 5));
   FileWrite(handle, "Stoch_K", "BB_Upper", DoubleToString(CalculateCorrelation(stoch_k, bb_up), 5));
   FileWrite(handle, "Stoch_K", "BB_Lower", DoubleToString(CalculateCorrelation(stoch_k, bb_low), 5));

   // Stoch_D vs others
   FileWrite(handle, "Stoch_D", "MACD_Main", DoubleToString(CalculateCorrelation(stoch_d, macd_main), 5));
   FileWrite(handle, "Stoch_D", "MACD_Signal", DoubleToString(CalculateCorrelation(stoch_d, macd_signal), 5));
   FileWrite(handle, "Stoch_D", "BB_Main", DoubleToString(CalculateCorrelation(stoch_d, bb_main), 5));
   FileWrite(handle, "Stoch_D", "BB_Upper", DoubleToString(CalculateCorrelation(stoch_d, bb_up), 5));
   FileWrite(handle, "Stoch_D", "BB_Lower", DoubleToString(CalculateCorrelation(stoch_d, bb_low), 5));

   // MACD_Main vs others
   FileWrite(handle, "MACD_Main", "MACD_Signal", DoubleToString(CalculateCorrelation(macd_main, macd_signal), 5));
   FileWrite(handle, "MACD_Main", "BB_Main", DoubleToString(CalculateCorrelation(macd_main, bb_main), 5));
   FileWrite(handle, "MACD_Main", "BB_Upper", DoubleToString(CalculateCorrelation(macd_main, bb_up), 5));
   FileWrite(handle, "MACD_Main", "BB_Lower", DoubleToString(CalculateCorrelation(macd_main, bb_low), 5));

   // MACD_Signal vs others
   FileWrite(handle, "MACD_Signal", "BB_Main", DoubleToString(CalculateCorrelation(macd_signal, bb_main), 5));
   FileWrite(handle, "MACD_Signal", "BB_Upper", DoubleToString(CalculateCorrelation(macd_signal, bb_up), 5));
   FileWrite(handle, "MACD_Signal", "BB_Lower", DoubleToString(CalculateCorrelation(macd_signal, bb_low), 5));

   // BB_Main vs others
   FileWrite(handle, "BB_Main", "BB_Upper", DoubleToString(CalculateCorrelation(bb_main, bb_up), 5));
   FileWrite(handle, "BB_Main", "BB_Lower", DoubleToString(CalculateCorrelation(bb_main, bb_low), 5));

   // BB_Upper vs BB_Lower
   FileWrite(handle, "BB_Upper", "BB_Lower", DoubleToString(CalculateCorrelation(bb_up, bb_low), 5));

   //--- flush and close file
   FileFlush(handle);
   int bytes = (int)FileSize(handle);
   FileClose(handle);

   PrintFormat("Training indicators correlations export complete. Wrote %d bytes.", bytes);

   //--- release indicator handles
   if(h_EMA50 != INVALID_HANDLE) IndicatorRelease(h_EMA50);
   if(h_EMA100 != INVALID_HANDLE) IndicatorRelease(h_EMA100);
   if(h_EMA200 != INVALID_HANDLE) IndicatorRelease(h_EMA200);
   if(h_RSI != INVALID_HANDLE) IndicatorRelease(h_RSI);
   if(h_ATR != INVALID_HANDLE) IndicatorRelease(h_ATR);
   if(h_STOCH != INVALID_HANDLE) IndicatorRelease(h_STOCH);
   if(h_MACD != INVALID_HANDLE) IndicatorRelease(h_MACD);
   if(h_BB != INVALID_HANDLE) IndicatorRelease(h_BB);
  }
//+------------------------------------------------------------------+