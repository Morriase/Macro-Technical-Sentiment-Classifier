//+------------------------------------------------------------------+
//|                                                 initial_data.mq5 |
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
input int shift = 10;           // maximum shift (prediction horizon) to test
// ZigZag parameters removed; using explicit example resource call per user instruction

// Global file handle for runtime logging so progress is visible on disk
int g_logHandle = INVALID_HANDLE;

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
  PrintFormat("Starting correlation export: %s %s -> %s (TF=%d)", _Symbol, TimeToString(Start), TimeToString(End), selectTimeframe);
  // open a logfile so the user can tail progress from disk while script runs
  g_logHandle = FileOpen("correlation.log", FILE_WRITE | FILE_ANSI, 0, CP_UTF8);
  if(g_logHandle == INVALID_HANDLE)
    PrintFormat("Warning: failed to open correlation.log for writing");
  else
    {
     FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "START", _Symbol);
     FileFlush(g_logHandle);
    }
  //--- create indicator handles
   int h_EMA50 = iMA(_Symbol, selectTimeframe, 50, 0, MODE_EMA, PRICE_CLOSE);
   int h_EMA100 = iMA(_Symbol, selectTimeframe, 100, 0, MODE_EMA, PRICE_CLOSE);
   int h_EMA200 = iMA(_Symbol, selectTimeframe, 200, 0, MODE_EMA, PRICE_CLOSE);
   int h_RSI   = iRSI(_Symbol, selectTimeframe, 14, PRICE_CLOSE);
   int h_ATR   = iATR(_Symbol, selectTimeframe, 14);
   int h_Stoh  = iStochastic(_Symbol, selectTimeframe, 14, 3, 3, MODE_LWMA, STO_LOWHIGH);
   int h_MACD  = iMACD(_Symbol, selectTimeframe, 12, 26, 9, PRICE_CLOSE);
   int h_BB    = iBands(_Symbol, selectTimeframe, 20, 0, 2, PRICE_CLOSE);
  // ZigZag handle (use explicit call provided by user)
  int h_ZZ = iCustom(_Symbol, PERIOD_M5, "Examples\\ZigZag.ex5", 48, 1, 47);

   //--- get OHLC via CopyRates (robust when working with datetime ranges)
   MqlRates rates[];
   int copiedRates = CopyRates(_Symbol, selectTimeframe, Start, End, rates);
   if(copiedRates <= 0)
     {
      PrintFormat("CopyRates failed or returned 0 bars (%s %d)", _Symbol, selectTimeframe);
      // release handles before returning
      if(h_EMA50 != INVALID_HANDLE) IndicatorRelease(h_EMA50);
      if(h_EMA100 != INVALID_HANDLE) IndicatorRelease(h_EMA100);
      if(h_EMA200 != INVALID_HANDLE) IndicatorRelease(h_EMA200);
      if(h_RSI != INVALID_HANDLE) IndicatorRelease(h_RSI);
      if(h_ATR != INVALID_HANDLE) IndicatorRelease(h_ATR);
      if(h_Stoh != INVALID_HANDLE) IndicatorRelease(h_Stoh);
      if(h_MACD != INVALID_HANDLE) IndicatorRelease(h_MACD);
      if(h_BB != INVALID_HANDLE) IndicatorRelease(h_BB);
      return;
     }
   
     PrintFormat("CopyRates succeeded: %d bars retrieved", copiedRates);
     if(g_logHandle != INVALID_HANDLE)
       {
        FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "CopyRates", copiedRates);
        FileFlush(g_logHandle);
       }
   //--- create separate OHLC arrays (series-oriented: most recent at index 0)
   int total = ArraySize(rates);
   PrintFormat("Preparing buffers of length %d", total);
   if(g_logHandle != INVALID_HANDLE)
     {
      FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "Preparing buffers", total);
      FileFlush(g_logHandle);
     }
   // no need to ArrayResize(rates, total) — already sized by CopyRates

   double close[], open[], high[], low[];
   ArrayResize(close, total);
   ArrayResize(open, total);
   ArrayResize(high, total);
   ArrayResize(low, total);

   // Copy MqlRates into arrays in the same order as built-in Copy* functions (series, index 0 = current)
   for(int i=0; i<total; i++)
     {
      open[i]  = rates[i].open;
      high[i]  = rates[i].high;
      low[i]   = rates[i].low;
      close[i] = rates[i].close;
     }

   //--- prepare buffers for indicators
   int needed = total;
   int extra = 50;
   double ema50[], ema100[], ema200[], rsi[], macd_main[], macd_signal[], atr[], bands_medium[], bands_up[], bands_low[];
   double stoch[], ssig[];

   ArrayResize(ema50, needed);
   ArrayResize(ema100, needed);
   ArrayResize(ema200, needed);
   ArrayResize(rsi, needed);
   ArrayResize(macd_main, needed);
   ArrayResize(macd_signal, needed);
   ArrayResize(atr, needed);
   ArrayResize(bands_medium, needed);
   ArrayResize(bands_up, needed);
   ArrayResize(bands_low, needed);
   ArrayResize(stoch, needed);
   ArrayResize(ssig, needed);

   //--- copy buffers from indicator handles (start position 0, count = needed or needed+extra)
   if(h_EMA50 == INVALID_HANDLE || CopyBuffer(h_EMA50, 0, 0, needed, ema50) <= 0 ||
      h_EMA100 == INVALID_HANDLE || CopyBuffer(h_EMA100, 0, 0, needed, ema100) <= 0 ||
      h_EMA200 == INVALID_HANDLE || CopyBuffer(h_EMA200, 0, 0, needed, ema200) <= 0 ||
      h_RSI == INVALID_HANDLE || CopyBuffer(h_RSI, 0, 0, needed, rsi) <= 0 ||
      h_MACD == INVALID_HANDLE || CopyBuffer(h_MACD, MAIN_LINE, 0, needed, macd_main) <= 0 ||
      CopyBuffer(h_MACD, SIGNAL_LINE, 0, needed, macd_signal) <= 0 ||
      h_ATR == INVALID_HANDLE || CopyBuffer(h_ATR, 0, 0, needed, atr) <= 0 ||
      h_BB == INVALID_HANDLE || CopyBuffer(h_BB, BASE_LINE, 0, needed, bands_medium) <= 0 ||
      CopyBuffer(h_BB, UPPER_BAND, 0, needed, bands_up) <= 0 ||
      CopyBuffer(h_BB, LOWER_BAND, 0, needed, bands_low) <= 0 ||
      h_Stoh == INVALID_HANDLE || CopyBuffer(h_Stoh, MAIN_LINE, 0, needed, stoch) <= 0 ||
      CopyBuffer(h_Stoh, SIGNAL_LINE, 0, needed, ssig) <= 0)
     {
      Print("One or more indicator buffers failed to copy. Check indicator handles / parameters.");
      // not returning here — we will attempt to proceed with what we have
     }
   
     Print("Indicator buffer copy attempted (check earlier message for failures if any)");
     if(g_logHandle != INVALID_HANDLE)
       {
        FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "Indicator buffers copied (attempted)");
        FileFlush(g_logHandle);
       }
   //--- prepare derived signals
   double oc[], hc[], lc[], macd_delta[], bb_main_dist[], bb_low_dist[], bb_up_dist[];
   ArrayResize(oc, needed);
   ArrayResize(hc, needed);
   ArrayResize(lc, needed);
   ArrayResize(macd_delta, needed);
   ArrayResize(bb_main_dist, needed);
   ArrayResize(bb_low_dist, needed);
   ArrayResize(bb_up_dist, needed);

   //--- compute derived arrays
   for(int i = 0; i < needed; i++)
     {
      oc[i] = close[i] - open[i];
      hc[i] = high[i] - close[i];
      lc[i] = close[i] - low[i];

      if(ArraySize(macd_main) > i && ArraySize(macd_signal) > i)
         macd_delta[i] = macd_main[i] - macd_signal[i];
      else
         macd_delta[i] = 0.0;

      if(ArraySize(bands_medium) > i) bb_main_dist[i] = bands_medium[i] - close[i];
      if(ArraySize(bands_low) > i) bb_low_dist[i] = close[i] - bands_low[i];
      if(ArraySize(bands_up) > i) bb_up_dist[i] = bands_up[i] - close[i];
     }
   
   Print("Derived signal arrays computed");
   if(g_logHandle != INVALID_HANDLE)
     {
      FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "Derived arrays computed");
      FileFlush(g_logHandle);
     }
   //--- open CSV file (use proper error check) with comma delimiter
   int handle = FileOpen("correlation.csv",
                         FILE_WRITE | FILE_CSV | FILE_ANSI,
                         ',',
                         CP_UTF8);

   if(handle == INVALID_HANDLE)
     {
      Print("Failed to open correlation.csv for writing");
      // release handles
      if(h_EMA50 != INVALID_HANDLE) IndicatorRelease(h_EMA50);
      if(h_EMA100 != INVALID_HANDLE) IndicatorRelease(h_EMA100);
      if(h_EMA200 != INVALID_HANDLE) IndicatorRelease(h_EMA200);
      if(h_RSI != INVALID_HANDLE) IndicatorRelease(h_RSI);
      if(h_ATR != INVALID_HANDLE) IndicatorRelease(h_ATR);
      if(h_Stoh != INVALID_HANDLE) IndicatorRelease(h_Stoh);
      if(h_MACD != INVALID_HANDLE) IndicatorRelease(h_MACD);
      if(h_BB != INVALID_HANDLE) IndicatorRelease(h_BB);
      return;
     }

  //--- write header (CSV columns) - include Pearson and Spearman versions
  FileWrite(handle, "Indicator", "Shift", "Correlation_Direction_Pearson", "Correlation_Return_Pearson", "Correlation_Direction_Spearman", "Correlation_Return_Spearman");

   //--- call ShiftCorrelation over the signals we have and collect total lines written
  int totalLines = 0;
  int lw = 0;
  lw = ShiftCorrelation(close, open, high, low, ema50, "EMA50", shift, handle);
  totalLines += lw; PrintFormat("ShiftCorrelation: %s wrote %d lines", "EMA50", lw);
  if(g_logHandle != INVALID_HANDLE) { FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ShiftCorrelation", "EMA50", lw); FileFlush(g_logHandle); }
  lw = ShiftCorrelation(close, open, high, low, ema100, "EMA100", shift, handle);
  totalLines += lw; PrintFormat("ShiftCorrelation: %s wrote %d lines", "EMA100", lw);
  if(g_logHandle != INVALID_HANDLE) { FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ShiftCorrelation", "EMA100", lw); FileFlush(g_logHandle); }
  lw = ShiftCorrelation(close, open, high, low, ema200, "EMA200", shift, handle);
  totalLines += lw; PrintFormat("ShiftCorrelation: %s wrote %d lines", "EMA200", lw);
  if(g_logHandle != INVALID_HANDLE) { FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ShiftCorrelation", "EMA200", lw); FileFlush(g_logHandle); }
  lw = ShiftCorrelation(close, open, high, low, rsi, "RSI", shift, handle);
  totalLines += lw; PrintFormat("ShiftCorrelation: %s wrote %d lines", "RSI", lw);
  if(g_logHandle != INVALID_HANDLE) { FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ShiftCorrelation", "RSI", lw); FileFlush(g_logHandle); }
  lw = ShiftCorrelation(close, open, high, low, atr, "ATR", shift, handle);
  totalLines += lw; PrintFormat("ShiftCorrelation: %s wrote %d lines", "ATR", lw);
  if(g_logHandle != INVALID_HANDLE) { FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ShiftCorrelation", "ATR", lw); FileFlush(g_logHandle); }
  lw = ShiftCorrelation(close, open, high, low, stoch, "Stochastic Main", shift, handle);
  totalLines += lw; PrintFormat("ShiftCorrelation: %s wrote %d lines", "Stochastic Main", lw);
  if(g_logHandle != INVALID_HANDLE) { FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ShiftCorrelation", "Stochastic Main", lw); FileFlush(g_logHandle); }
  lw = ShiftCorrelation(close, open, high, low, ssig, "Stochastic Signal", shift, handle);
  totalLines += lw; PrintFormat("ShiftCorrelation: %s wrote %d lines", "Stochastic Signal", lw);
  if(g_logHandle != INVALID_HANDLE) { FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ShiftCorrelation", "Stochastic Signal", lw); FileFlush(g_logHandle); }
  lw = ShiftCorrelation(close, open, high, low, macd_main, "MACD Main", shift, handle);
  totalLines += lw; PrintFormat("ShiftCorrelation: %s wrote %d lines", "MACD Main", lw);
  if(g_logHandle != INVALID_HANDLE) { FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ShiftCorrelation", "MACD Main", lw); FileFlush(g_logHandle); }
  lw = ShiftCorrelation(close, open, high, low, macd_signal, "MACD Signal", shift, handle);
  totalLines += lw; PrintFormat("ShiftCorrelation: %s wrote %d lines", "MACD Signal", lw);
  if(g_logHandle != INVALID_HANDLE) { FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ShiftCorrelation", "MACD Signal", lw); FileFlush(g_logHandle); }
  lw = ShiftCorrelation(close, open, high, low, macd_delta, "MACD Main-Signal", shift, handle);
  totalLines += lw; PrintFormat("ShiftCorrelation: %s wrote %d lines", "MACD Main-Signal", lw);
  if(g_logHandle != INVALID_HANDLE) { FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ShiftCorrelation", "MACD Main-Signal", lw); FileFlush(g_logHandle); }
  lw = ShiftCorrelation(close, open, high, low, bb_main_dist, "BB Main", shift, handle);
  totalLines += lw; PrintFormat("ShiftCorrelation: %s wrote %d lines", "BB Main", lw);
  if(g_logHandle != INVALID_HANDLE) { FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ShiftCorrelation", "BB Main", lw); FileFlush(g_logHandle); }
  lw = ShiftCorrelation(close, open, high, low, bb_low_dist, "BB Low", shift, handle);
  totalLines += lw; PrintFormat("ShiftCorrelation: %s wrote %d lines", "BB Low", lw);
  if(g_logHandle != INVALID_HANDLE) { FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ShiftCorrelation", "BB Low", lw); FileFlush(g_logHandle); }
  lw = ShiftCorrelation(close, open, high, low, bb_up_dist, "BB Up", shift, handle);
  totalLines += lw; PrintFormat("ShiftCorrelation: %s wrote %d lines", "BB Up", lw);
  if(g_logHandle != INVALID_HANDLE) { FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ShiftCorrelation", "BB Up", lw); FileFlush(g_logHandle); }
  lw = ShiftCorrelation(close, open, high, low, oc, "OC", shift, handle);
  totalLines += lw; PrintFormat("ShiftCorrelation: %s wrote %d lines", "OC", lw);
  if(g_logHandle != INVALID_HANDLE) { FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ShiftCorrelation", "OC", lw); FileFlush(g_logHandle); }
  lw = ShiftCorrelation(close, open, high, low, hc, "HC", shift, handle);
  totalLines += lw; PrintFormat("ShiftCorrelation: %s wrote %d lines", "HC", lw);
  if(g_logHandle != INVALID_HANDLE) { FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ShiftCorrelation", "HC", lw); FileFlush(g_logHandle); }
  lw = ShiftCorrelation(close, open, high, low, lc, "LC", shift, handle);
  totalLines += lw; PrintFormat("ShiftCorrelation: %s wrote %d lines", "LC", lw);
  if(g_logHandle != INVALID_HANDLE) { FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ShiftCorrelation", "LC", lw); FileFlush(g_logHandle); }

  //--- compute ZigZag buffer and nearest-future-extremum correlations
  double zzbuf[];
  ArrayResize(zzbuf, needed);
  if(h_ZZ == INVALID_HANDLE || CopyBuffer(h_ZZ, 0, 0, needed, zzbuf) <= 0)
    {
     Print("ZigZag buffer failed to copy; skipping ZigZag-based correlations");
     if(g_logHandle != INVALID_HANDLE) { FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ZigZag buffer failed to copy"); FileFlush(g_logHandle); }
    }
  else
   {
    int zl = 0;
    zl = ZigZagCorrelations(close, ema50, zzbuf, atr, "EMA50", shift, handle); totalLines += zl; if(g_logHandle!=INVALID_HANDLE){ FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ZigZag", "EMA50", zl); FileFlush(g_logHandle); }
    zl = ZigZagCorrelations(close, ema100, zzbuf, atr, "EMA100", shift, handle); totalLines += zl; if(g_logHandle!=INVALID_HANDLE){ FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ZigZag", "EMA100", zl); FileFlush(g_logHandle); }
    zl = ZigZagCorrelations(close, ema200, zzbuf, atr, "EMA200", shift, handle); totalLines += zl; if(g_logHandle!=INVALID_HANDLE){ FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ZigZag", "EMA200", zl); FileFlush(g_logHandle); }
    zl = ZigZagCorrelations(close, rsi, zzbuf, atr, "RSI", shift, handle); totalLines += zl; if(g_logHandle!=INVALID_HANDLE){ FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ZigZag", "RSI", zl); FileFlush(g_logHandle); }
    zl = ZigZagCorrelations(close, atr, zzbuf, atr, "ATR", shift, handle); totalLines += zl; if(g_logHandle!=INVALID_HANDLE){ FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ZigZag", "ATR", zl); FileFlush(g_logHandle); }
    zl = ZigZagCorrelations(close, stoch, zzbuf, atr, "Stochastic Main", shift, handle); totalLines += zl; if(g_logHandle!=INVALID_HANDLE){ FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ZigZag", "Stochastic Main", zl); FileFlush(g_logHandle); }
    zl = ZigZagCorrelations(close, ssig, zzbuf, atr, "Stochastic Signal", shift, handle); totalLines += zl; if(g_logHandle!=INVALID_HANDLE){ FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ZigZag", "Stochastic Signal", zl); FileFlush(g_logHandle); }
    zl = ZigZagCorrelations(close, macd_main, zzbuf, atr, "MACD Main", shift, handle); totalLines += zl; if(g_logHandle!=INVALID_HANDLE){ FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ZigZag", "MACD Main", zl); FileFlush(g_logHandle); }
    zl = ZigZagCorrelations(close, macd_signal, zzbuf, atr, "MACD Signal", shift, handle); totalLines += zl; if(g_logHandle!=INVALID_HANDLE){ FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ZigZag", "MACD Signal", zl); FileFlush(g_logHandle); }
    zl = ZigZagCorrelations(close, macd_delta, zzbuf, atr, "MACD Main-Signal", shift, handle); totalLines += zl; if(g_logHandle!=INVALID_HANDLE){ FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ZigZag", "MACD Main-Signal", zl); FileFlush(g_logHandle); }
    zl = ZigZagCorrelations(close, bb_main_dist, zzbuf, atr, "BB Main", shift, handle); totalLines += zl; if(g_logHandle!=INVALID_HANDLE){ FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ZigZag", "BB Main", zl); FileFlush(g_logHandle); }
    zl = ZigZagCorrelations(close, bb_low_dist, zzbuf, atr, "BB Low", shift, handle); totalLines += zl; if(g_logHandle!=INVALID_HANDLE){ FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ZigZag", "BB Low", zl); FileFlush(g_logHandle); }
    zl = ZigZagCorrelations(close, bb_up_dist, zzbuf, atr, "BB Up", shift, handle); totalLines += zl; if(g_logHandle!=INVALID_HANDLE){ FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ZigZag", "BB Up", zl); FileFlush(g_logHandle); }
    zl = ZigZagCorrelations(close, oc, zzbuf, atr, "OC", shift, handle); totalLines += zl; if(g_logHandle!=INVALID_HANDLE){ FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ZigZag", "OC", zl); FileFlush(g_logHandle); }
    zl = ZigZagCorrelations(close, hc, zzbuf, atr, "HC", shift, handle); totalLines += zl; if(g_logHandle!=INVALID_HANDLE){ FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ZigZag", "HC", zl); FileFlush(g_logHandle); }
    zl = ZigZagCorrelations(close, lc, zzbuf, atr, "LC", shift, handle); totalLines += zl; if(g_logHandle!=INVALID_HANDLE){ FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ZigZag", "LC", zl); FileFlush(g_logHandle); }
   }

   //--- flush and get file size BEFORE closing
   FileFlush(handle);
   int bytes = (int)FileSize(handle);
   FileClose(handle);

   //--- release indicator handles
   if(h_EMA50 != INVALID_HANDLE) IndicatorRelease(h_EMA50);
   if(h_EMA100 != INVALID_HANDLE) IndicatorRelease(h_EMA100);
   if(h_EMA200 != INVALID_HANDLE) IndicatorRelease(h_EMA200);
   if(h_RSI != INVALID_HANDLE) IndicatorRelease(h_RSI);
   if(h_ATR != INVALID_HANDLE) IndicatorRelease(h_ATR);
   if(h_Stoh != INVALID_HANDLE) IndicatorRelease(h_Stoh);
   if(h_MACD != INVALID_HANDLE) IndicatorRelease(h_MACD);
   if(h_BB != INVALID_HANDLE) IndicatorRelease(h_BB);
  if(h_ZZ != INVALID_HANDLE) IndicatorRelease(h_ZZ);

   PrintFormat("Correlation export complete. Wrote %d bytes, %d data lines.", bytes, totalLines);
  if(g_logHandle != INVALID_HANDLE)
    {
     FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "COMPLETE", bytes, totalLines);
     FileFlush(g_logHandle);
     FileClose(g_logHandle);
     g_logHandle = INVALID_HANDLE;
    }

   if(totalLines == 0)
      Print("Warning: no correlation rows were written. Possible causes: dataset too small, shift is too large, or len <= 10 sample check filtered all shifts.");
  }

//+------------------------------------------------------------------+
// Simple ranking function for Spearman: ranks start at 1..n, ties get average rank
void RankArray(const double &arr[], double &ranks[], int n)
  {
  for(int i=0;i<n;i++)
    {
    int count_less = 0;
    int count_equal = 0;
    for(int j=0;j<n;j++)
      {
      if(arr[j] < arr[i]) count_less++;
      else if(arr[j] == arr[i]) count_equal++;
      }
    // average rank for ties
    ranks[i] = (double)count_less + ((double)count_equal + 1.0) / 2.0;
    }
  }

int ShiftCorrelation(double &close[], double &open[], double &high[], double &low[],
              double &signal[], string name,
              int max_shift, int handle)
  {
  int total = ArraySize(close);
  if(total <= 0 || ArraySize(signal) <= 0)
    return(0);

  if(max_shift <= 0)
    max_shift = 1;
  if(max_shift > total - 2)
    max_shift = total - 2;
  if(max_shift < 1)
    return(0);

  int lines_written = 0;

  for(int s = 0; s <= max_shift; s++)
    {
    int len = total - s;
    if(len <= 10) // need enough samples for correlation
      continue;

    double t1[], t2[], sarr[];
    ArrayResize(t1, len);
    ArrayResize(t2, len);
    ArrayResize(sarr, len);

    // copy shifted arrays: signal from [0 .. len-1] into sarr[0..len-1]
    ArrayCopy(sarr, signal, 0, 0, len);

    // compute targets: for each j, target at j+s
    for(int j = 0; j < len; j++)
      {
      t2[j] = close[j + s] - close[j];
      t1[j] = (t2[j] > 0.0) ? 1.0 : 0.0;
      }

    double corr_dir = 0.0, corr_ret = 0.0;
    bool ok_dir = MathCorrelationPearson(sarr, t1, corr_dir);
    bool ok_ret = MathCorrelationPearson(sarr, t2, corr_ret);

    double c_dir = ok_dir ? corr_dir : 0.0;
    double c_ret = ok_ret ? corr_ret : 0.0;

    // Spearman: rank-transform sarr, t1, t2 then Pearson on ranks
    double rank_s[], rank_t1[], rank_t2[];
    ArrayResize(rank_s, len);
    ArrayResize(rank_t1, len);
    ArrayResize(rank_t2, len);
    RankArray(sarr, rank_s, len);
    RankArray(t1, rank_t1, len);
    RankArray(t2, rank_t2, len);

    double corr_dir_s = 0.0, corr_ret_s = 0.0;
    bool ok_dir_s = MathCorrelationPearson(rank_s, rank_t1, corr_dir_s);
    bool ok_ret_s = MathCorrelationPearson(rank_s, rank_t2, corr_ret_s);

    double c_dir_s = ok_dir_s ? corr_dir_s : 0.0;
    double c_ret_s = ok_ret_s ? corr_ret_s : 0.0;

    FileWrite(handle, name, s, DoubleToString(c_dir,5), DoubleToString(c_ret,5), DoubleToString(c_dir_s,5), DoubleToString(c_ret_s,5));
    lines_written++;
    }

  PrintFormat("ShiftCorrelation: %s shifts=%d wrote %d lines", name, max_shift, lines_written);
  if(g_logHandle != INVALID_HANDLE)
    {
     FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ShiftSummary", name, max_shift, lines_written);
     FileFlush(g_logHandle);
    }
  return(lines_written);
  }
//+------------------------------------------------------------------+
// Correlate a signal array with the nearest future ZigZag extremum
int ZigZagCorrelations(double &close[], double &signal[], double &zz[], double &atr[], string name, int max_lookahead, int handle)
  {
   int total = ArraySize(close);
   if(total <= 0 || ArraySize(signal) <= 0 || ArraySize(zz) <= 0)
      return(0);

   if(max_lookahead <= 0)
      max_lookahead = 1;

   // Prepare compact arrays only for bars that have a future extremum within lookahead
   double sarr[];
   double targ_dir[];
   double targ_mag[];
   ArrayResize(sarr, total);
   ArrayResize(targ_dir, total);
   ArrayResize(targ_mag, total);

   int cnt = 0;
   for(int i = 0; i < total; i++)
     {
      double future_ext = EMPTY_VALUE;
      int maxj = MathMin(total-1, i + max_lookahead);
      for(int j = i+1; j <= maxj; j++)
        {
         if(zz[j] != EMPTY_VALUE)
           {
            future_ext = zz[j];
            break; // nearest future extremum
           }
        }

      if(future_ext != EMPTY_VALUE)
        {
         double mag = future_ext - close[i];
         // normalize magnitude by ATR when available
         double mag_norm = mag;
         if(ArraySize(atr) > i && atr[i] > 0.0)
            mag_norm = mag / atr[i];

         // convert direction to binary (1 = up, 0 = down)
         double dir_bin = (mag > 0.0) ? 1.0 : 0.0;

         sarr[cnt] = signal[i];
         targ_dir[cnt] = dir_bin;
         targ_mag[cnt] = mag_norm;
         cnt++;
        }
     }

   if(cnt <= 10)
     return(0);

   // resize to actual count
   ArrayResize(sarr, cnt);
   ArrayResize(targ_dir, cnt);
   ArrayResize(targ_mag, cnt);

  double corr_dir = 0.0, corr_mag = 0.0;
  bool ok_dir = MathCorrelationPearson(sarr, targ_dir, corr_dir);
  bool ok_mag = MathCorrelationPearson(sarr, targ_mag, corr_mag);

  double c_dir = ok_dir ? corr_dir : 0.0;
  double c_mag = ok_mag ? corr_mag : 0.0;

  // Compute Spearman via ranks
  double rank_s[], rank_dir[], rank_mag[];
  ArrayResize(rank_s, cnt);
  ArrayResize(rank_dir, cnt);
  ArrayResize(rank_mag, cnt);
  RankArray(sarr, rank_s, cnt);
  RankArray(targ_dir, rank_dir, cnt);
  RankArray(targ_mag, rank_mag, cnt);

  double corr_dir_s = 0.0, corr_mag_s = 0.0;
  bool ok_dir_s = MathCorrelationPearson(rank_s, rank_dir, corr_dir_s);
  bool ok_mag_s = MathCorrelationPearson(rank_s, rank_mag, corr_mag_s);

  double c_dir_s = ok_dir_s ? corr_dir_s : 0.0;
  double c_mag_s = ok_mag_s ? corr_mag_s : 0.0;
  // log summary for debugging/progress
  PrintFormat("ZigZagCorrelations: %s cnt=%d pearson_dir=%.5f pearson_mag=%.5f spearman_dir=%.5f spearman_mag=%.5f", name, cnt, c_dir, c_mag, c_dir_s, c_mag_s);
  if(g_logHandle != INVALID_HANDLE)
    {
     FileWrite(g_logHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), "ZigZagSummary", name, cnt, DoubleToString(c_dir,5), DoubleToString(c_mag,5), DoubleToString(c_dir_s,5), DoubleToString(c_mag_s,5));
     FileFlush(g_logHandle);
    }

  // write one line using Shift=-1 to denote ZigZag-based target
  FileWrite(handle, name + " (ZigZag)", -1, DoubleToString(c_dir,5), DoubleToString(c_mag,5), DoubleToString(c_dir_s,5), DoubleToString(c_mag_s,5));
  PrintFormat("ZigZagCorrelations: %s wrote 1 line (cnt=%d)", name, cnt);
  return(1);
   
  }
//+------------------------------------------------------------------+
