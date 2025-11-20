//+------------------------------------------------------------------+
//|                                         BlackIce_REST_EA_V2.mq5  |
//|                          Optimized for 64% RandomForest Model    |
//|                          Single Model (RF Only) - Stable & Profitable |
//+------------------------------------------------------------------+
#property copyright "Auron Automations" 
#property link "https://www.auronautomations.app"
#property version   "1.00"
#property description "Guardian AI"
#property strict
#include <Trade\Trade.mqh>

CTrade         trade; 
CPositionInfo  posinfo; 
COrderInfo     orderinfo; 


enum    RiskType        {  systemRisk = 0,      // System defined risk (By confidence)
                           userRisk = 1,        // Risk defined By You
                           hardCodedRisk = 2    // Fixed 1% risk per trade (You cannot change)
                        };

enum    StopLossMode    {  atrBased = 0,      // ATR-based (Adaptive to volatility)
                           fixedPips = 1,     // Fixed pips (Manual)
                           percentage = 2     // Percentage of price
                        };

enum    TrType          {  Highlow=1,        // Previous low or high 
                           FixedPips=2,      // Defined no of pips 
                           PctofPrice = 3    // Trail stoploss as % of distance from the TP 
                        }; 
                           
enum    SLType          {  Yes=0,            // Use Trailing stoploss
                           No=1              // Use fixed stoploss 
                        }; 
                        
enum    IntervalTime     {
                           newBar = 0,       // Once every new bar
                           xSeconds = 1      // Every X Seconds   
                        };

enum    AutoCloseTime    {
                           Disabled = 0,     // Disabled (no auto-close)
                           EndOfDay = 1,     // Close at end of trading day
                           CustomTime = 2    // Close at custom time
                        }; 
                        
                        
enum SeparatorSymbol    {
                          comma = 0,         // Comma
                          semiColon = 1      // Semi-colon 
                        };                                                 
input group "/--- Input Parameters ---/"

   input string            RestServerURL =  "https://morriase-forex-live-server.hf.space/predict";   // REST API URL
   input long              inpMagic = 123456;                                 // EA magic number
   input IntervalTime      UpdateIntervalSeconds = 0;                         // Update interval
   input int               updateSeconds = 10;                                // Update interval in Seconds if "Every X Seconds" selected
   input double            MinConfidence = 0.55;                              // Minimum confidence
   input StopLossMode      SLMode = atrBased;                                 // Stop loss calculation mode
   input double            ATRMultiplier = 2.5;                               // ATR multiplier for SL (if atrBased)
   input int               ATRPeriod = 14;                                    // ATR period
   input int               StopLossPips = 50;                                 // Stop loss in pips (if fixedPips mode)
   input double            SLPercentage = 0.5;                                // SL as % of price (if percentage mode)
   input double            MinRiskReward = 2.0;                               // Minimum R:R ratio
   input bool              EnableTrading = false;                             // Enable actual trading
   input bool              ShowDebugInfo = true;                              // Show debug logs
   input bool              PredictOnStart = true;                             // Make prediction on EA start
   input ENUM_TIMEFRAMES   AnalysisTimeframe = PERIOD_M15;                    // Timeframe for analysis (ATR, trailing stops)

input group "/--- Risk & Order Management Inputs ---/"
   
   input RiskType RiskMode = systemRisk;                             // Risk calculation mode
   input double   UserRiskPercent = 1.0;                             // Manual risk % (if userRisk mode)
   input bool     UseConfidenceScaling = true;                       // Scale risk by confidence (systemRisk mode only)
   input bool     EnableCompensatory = true;                         // Enable recovery multiplier after losing trades
   input int      RecoveryDepth = 3;                                 // Max consecutive recovery steps
   input double   RecoveryMultiplier = 1.5;                          // Multiplier per recovery step
   input bool     EnableMarginCheck = false;                         // Enable margin safety check
   input double   MaxMarginPercent = 50.0;                           // Maximum margin usage (% of balance, if enabled)
   input double   ConfidenceHigh = 0.80;                             // High confidence threshold
   input double   ConfidenceMed = 0.70;                              // Medium confidence threshold
   input double   ConfidenceLow = 0.55;                              // Low confidence threshold

// Hard-coded risk parameters (for safety)
#define BASE_RISK_PERCENT 0.5    // Base risk: 0.5% per trade (systemRisk mode)
#define FIXED_RISK_PERCENT 1.0   // Fixed risk: 1.0% per trade (hardCodedRisk mode)
#define MAX_RISK_PERCENT 5.0     // Maximum risk: 5% per trade (safety cap)

input group "/--- Core functions news & trailing configuration ---/" 

   input bool                  NewsFilterOn = false;  // Enable news filtering
   input int                   StartTradingMin = 30;  // Minimum minutes to wait after news
   input SeparatorSymbol       separator = 0;        // 0=comma,1=semicolon for news keys
   input string                keyNews = "NFP,Non-Farm,Interest Rate,FOMC,CPI,GDP,Unemployment,Retail Sales,PMI,Central Bank,Fed,ECB,BoE,BoJ";        // Comma/semicolon separated list of news keywords
   input string                NewsCurrencies = "USD,EUR,GBP,JPY,AUD,CAD,CHF,NZD";                                                                      // Comma-separated currencies to monitor
   input int                   DaysNewsLookup = 2;   // How many days ahead to look for calendar events
   input int                   StopBeforeMin = 30;   // Minutes before news to avoid trading
   ushort sep_code;

input group "/--- Trailing params ---/"  
   input SLType   SLT = 0;                         // Use Trailing stoploss (TSL)? (No = Fixed Stoploss)
   input TrType   trailType = 1;                   // If using TSL, what type of TSL? 
   input int      BarsN = 20;                      // Number of bars to scan for highs and lows
   input int      HighLowBuffer = 3;               // Buffer from prev low or high to trail (If selected) 
   input int      trailFixedpips = 10;             // Number of pips to trail SL (If option is selected)
   input double   TslPercent = 1.0;                // Percentage of ptice to TSL
   input double   TslPercentTP = 50.0;             // start TSL at x% from TP

input group "/--- Auto-Close Settings ---/"
   input AutoCloseTime AutoClose = Disabled;       // Auto-close all trades
   input int      CloseHour = 23;                  // Hour to close (0-23, server time)
   input int      CloseMinute = 55;                // Minute to close (0-59)

// Include modular core functions (after all input declarations)
#include "core_functions.mqh"

//--- Global Variables
datetime lastBarTime = 0;
datetime lastUpdateTime = 0;
int requestCount = 0;
int successCount = 0;
bool firstRun = true;
int consecutiveLosses = 0;
double lastTradeProfit = 0.0;
string newsStatus = ""; // Global news status for chart display

//--- SMC Context (extracted from response)
string smcOrderBlocks = "";
string smcFairValueGaps = "";
string smcStructure = "";
string smcRegime = "";

//--- Trade logging
string logFileName = "BlackIce_Trades.csv";


//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{

   ChartSetInteger(0, CHART_SHOW_GRID, false); 
   
   Print("========================================");
   Print("HYBRID ENSEMBLE EA - MACRO-TECHNICAL SENTIMENT");
   Print("========================================");
   Print("Server: ", RestServerURL);
   Print("Timeframe: M5 (5-minute candles)");
   Print("Bars: 250 (for feature engineering)");
   Print("Features: 58 (55 Technical + 3 Macro)");
   Print("Chart Timeframe: ", EnumToString(PERIOD_CURRENT));
   
   if(UpdateIntervalSeconds == 0)
      Print("Update Mode: New M5 Bar Only");
   else
      Print("Update Mode: Every ", updateSeconds, " seconds");
   
   Print("Min Confidence: ", MinConfidence);
   Print("Trading: ", EnableTrading ? "ENABLED" : "DEMO MODE");
   Print("========================================");
   
   // Check WebRequest permission
   if(!TerminalInfoInteger(TERMINAL_DLLS_ALLOWED))
   {
      Alert("DLL imports must be allowed!");
      return(INIT_FAILED);
   }
   
   // Initialize trade log file
   InitializeTradeLog();
   
   lastBarTime = iTime(_Symbol, PERIOD_M5, 0);  // Use M5 as base timeframe
   lastUpdateTime = TimeCurrent();
   
   // Make initial prediction if enabled
   if(PredictOnStart)
   {
      Print("Making initial prediction...");
      MakePrediction();
   }
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // Clean up narrative objects if they exist
   ObjectDelete(0, "Narrative_Box");
   ObjectDelete(0, "Narrative_Text");
   ObjectDelete(0, "Narrative_Panel");
   
   Print("========================================");
   Print("EA Stopped. Stats:");
   Print("  Requests: ", requestCount);
   Print("  Success: ", successCount);
   Print("  Success Rate: ", (requestCount > 0 ? (double)successCount/requestCount*100 : 0), "%");
   Print("========================================");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Make prediction on first tick
   if(firstRun)
   {
      firstRun = false;
      if(!PredictOnStart)  // Only if not already done in OnInit
      {
         Print("Making first prediction...");
         MakePrediction();
      }
      return;
   }
   
   // Two update modes:
   // 1. New M5 bar only (UpdateIntervalSeconds = 0)
   // 2. Timer-based (UpdateIntervalSeconds > 0)
   
   bool shouldUpdate = false;
   datetime currentBarTime = iTime(_Symbol, PERIOD_M5, 0);  // Use M5 as base timeframe
   datetime currentTime = TimeCurrent();
   switch(UpdateIntervalSeconds){
      case 0:
         // Mode 1: Update on new M5 bar only
         if(currentBarTime != lastBarTime){
            lastBarTime = currentBarTime;
            shouldUpdate = true;
         }
         break;
      case 1:
         //Mode 2: Update every X seconds
         if(currentTime - lastUpdateTime >= updateSeconds){
            lastUpdateTime = currentTime;
            shouldUpdate = true;
            
            // Also update lastBarTime to track bar changes
            lastBarTime = iTime(_Symbol, PERIOD_M5, 0);
         }
         break;
   }
   
   // Update news status periodically (every minute)
   static datetime lastNewsCheck = 0;
   static bool newsInitialized = false;
   
   if(NewsFilterOn)
   {
      // Check immediately on first run, then every 60 seconds
      if(!newsInitialized || TimeCurrent() - lastNewsCheck >= 60)
      {
         lastNewsCheck = TimeCurrent();
         newsInitialized = true;
         IsUpcomingNews(); // This updates the global newsStatus variable
      }
   }
   
   // Check for auto-close time
   static bool closedToday = false;
   static int lastDay = 0;
   
   if(AutoClose != Disabled && EnableTrading)
   {
      MqlDateTime dt;
      TimeToStruct(TimeCurrent(), dt);
      
      // Reset flag on new day
      if(dt.day != lastDay)
      {
         closedToday = false;
         lastDay = dt.day;
      }
      
      // Check if it's time to close
      bool shouldClose = false;
      
      switch(AutoClose)
      {
         case EndOfDay:
            // Close at 23:55 server time by default
            if(dt.hour == 23 && dt.min >= 55 && !closedToday)
               shouldClose = true;
            break;
            
         case CustomTime:
            // Close at user-defined time
            if(dt.hour == CloseHour && dt.min >= CloseMinute && !closedToday)
               shouldClose = true;
            break;
      }
      
      if(shouldClose)
      {
         Print("⏰ Auto-close triggered at ", TimeToString(TimeCurrent(), TIME_DATE|TIME_MINUTES));
         CloseandResetAll();
         closedToday = true;
      }
   }
   
   // Apply trailing stop loss if enabled (from core_functions.mqh)
   if(SLT == Yes && EnableTrading)
   {
      TrailSL();
   }
   
   if(shouldUpdate)
   {
      MakePrediction();
   }
}

//+------------------------------------------------------------------+
//| Make prediction via REST API                                     |
//+------------------------------------------------------------------+
void MakePrediction()
{
   requestCount++;
   
   // Prepare OHLCV data
   string jsonData = PrepareOHLCVData();
   if(jsonData == "")
   {
      Print("ERROR: Failed to prepare OHLCV data");
      return;
   }
   
   // Send request
   char post[];
   char result[];
   string headers = "Content-Type: application/json\r\n";
   
   // If debug info requested, add debug flag to payload (server will return full result)
   if(ShowDebugInfo)
   {
      // Insert debug flag before closing brace of JSON
      int insertPos = StringFind(jsonData, "}", StringLen(jsonData) - 2);
      if(insertPos > 0)
      {
         string jsonWithDebug = StringSubstr(jsonData, 0, insertPos);
         jsonWithDebug += ",\"debug\":true";
         jsonWithDebug += StringSubstr(jsonData, insertPos, StringLen(jsonData) - insertPos);
         jsonData = jsonWithDebug;
      }
   }

   StringToCharArray(jsonData, post, 0, StringLen(jsonData));
   
   if(ShowDebugInfo)
      Print("Sending request to: ", RestServerURL);
   
   ResetLastError();
   int res = WebRequest(
      "POST",
      RestServerURL,
      headers,
      5000,  // timeout
      post,
      result,
      headers
   );
   
   if(res == -1)
   {
      int error = GetLastError();
      Print("ERROR: WebRequest failed. Error: ", error);
      if(error == 4060)
         Print("  URL not allowed. Add to Tools > Options > Expert Advisors");
      return;
   }
   
   // Parse response
   string response = CharArrayToString(result);
   
   if(ShowDebugInfo)
      Print("Response received: ", StringSubstr(response, 0, 200), "...");
   
   // Save full response to debug file for later inspection
   if(ShowDebugInfo)
   {
      int h = FileOpen("BlackIce_Response_Debug.jsonl", FILE_WRITE|FILE_READ|FILE_COMMON);
      if(h != INVALID_HANDLE)
      {
         FileSeek(h, 0, SEEK_END);
         FileWriteString(h, response + "\n");
         FileClose(h);
      }
   }
   
   ParseAndExecute(response);
   successCount++;
}

//+------------------------------------------------------------------+
//| Prepare M5 OHLCV JSON data for inference server                  |
//+------------------------------------------------------------------+
string PrepareOHLCVData()
{
   // Optimized bar counts to prevent server crashes
   // Server needs 250+ bars minimum, but sending 100k+ causes memory issues
   int barsM5 = 500;   // 500 M5 bars = ~41 hours of data
   int barsH1 = 300;   // 300 H1 bars = ~12.5 days of data
   int barsH4 = 250;   // 250 H4 bars = ~41 days of data
   
   // Check if we have enough bars available
   int availableM5 = Bars(_Symbol, PERIOD_M5);
   int availableH1 = Bars(_Symbol, PERIOD_H1);
   int availableH4 = Bars(_Symbol, PERIOD_H4);
   
   if(availableM5 < barsM5)
   {
      Print("ERROR: Insufficient M5 bars. Need ", barsM5, ", have ", availableM5);
      return "";
   }
   
   if(availableH1 < barsH1)
   {
      Print("ERROR: Insufficient H1 bars. Need ", barsH1, ", have ", availableH1);
      return "";
   }
   
   if(availableH4 < barsH4)
   {
      Print("ERROR: Insufficient H4 bars. Need ", barsH4, ", have ", availableH4);
      return "";
   }
   
   // Convert symbol format: EURUSD → EUR_USD (match training data format)
   string pairFormatted = ConvertSymbolFormat(_Symbol);
   
   if(ShowDebugInfo)
      Print("Symbol: ", _Symbol, " → Formatted: ", pairFormatted);
   
   // Build JSON structure with M5, H1, and H4 data
   // Format: {"pair": "EUR_USD", "ohlcv_m5": [...], "ohlcv_h1": [...], "ohlcv_h4": [...], "events": [...]}
   string json = "{";
   json += "\"pair\":\"" + pairFormatted + "\",";
   
   // M5 data (500 bars)
   json += "\"ohlcv_m5\":[";
   json += CollectTimeframeData(PERIOD_M5, barsM5);
   json += "],";
   
   // H1 data (300 bars)
   json += "\"ohlcv_h1\":[";
   json += CollectTimeframeData(PERIOD_H1, barsH1);
   json += "],";
   
   // H4 data (250 bars)
   json += "\"ohlcv_h4\":[";
   json += CollectTimeframeData(PERIOD_H4, barsH4);
   json += "],";
   
   // Add calendar events for macro features (if news filter enabled)
   if(NewsFilterOn)
   {
      json += "\"events\":";
      json += GetCalendarEventsJSON(pairFormatted);
   }
   else
   {
      json += "\"events\":[]";  // Empty events array if news filter disabled
   }
   
   json += "}";
   
   if(ShowDebugInfo)
      Print("📊 Sending ", barsM5, " M5 bars, ", barsH1, " H1 bars, ", barsH4, " H4 bars to server");
   
   return json;
}

//+------------------------------------------------------------------+
//| Collect OHLCV data for any timeframe                             |
//+------------------------------------------------------------------+
string CollectTimeframeData(ENUM_TIMEFRAMES timeframe, int bars)
{
   string data = "";
   
   for(int i = bars - 1; i >= 0; i--)
   {
      datetime time = iTime(_Symbol, timeframe, i);
      double open = iOpen(_Symbol, timeframe, i);
      double high = iHigh(_Symbol, timeframe, i);
      double low = iLow(_Symbol, timeframe, i);
      double close = iClose(_Symbol, timeframe, i);
      long volume = iVolume(_Symbol, timeframe, i);
      
      data += "{";
      data += "\"timestamp\":\"" + TimeToString(time, TIME_DATE|TIME_MINUTES) + "\",";
      data += "\"open\":" + DoubleToString(open, _Digits) + ",";
      data += "\"high\":" + DoubleToString(high, _Digits) + ",";
      data += "\"low\":" + DoubleToString(low, _Digits) + ",";
      data += "\"close\":" + DoubleToString(close, _Digits) + ",";
      data += "\"volume\":" + IntegerToString(volume);
      data += "}";
      
      if(i > 0) data += ",";
   }
   
   return data;
}

//+------------------------------------------------------------------+
//| Convert symbol format: EURUSD → EUR_USD                          |
//+------------------------------------------------------------------+
string ConvertSymbolFormat(string symbol)
{
   // Handle standard forex symbols
   if(StringLen(symbol) >= 6)
   {
      // Extract first 6 characters for forex pairs (ignores suffix like .ecn, .raw, etc.)
      string base = StringSubstr(symbol, 0, 3);
      string quote = StringSubstr(symbol, 3, 3);
      return base + "_" + quote;
   }
   
   // Fallback: return as-is
   return symbol;
}

//+------------------------------------------------------------------+
//| Parse response and execute trade                                 |
//+------------------------------------------------------------------+
void ParseAndExecute(string response)
{
   // ============================================================================
   // SERVER-SIDE FILTERING (v2.0 - Quality + Session Filters)
   // ============================================================================
   // The server now applies TWO layers of filtering before sending signals:
   //
   // 1. QUALITY FILTER:
   //    - OB_Displacement > 1.5 ATR (strong institutional move)
   //    - OB_Quality > 0.3 (good structure characteristics)
   //    - Increases win rate from 51% → 63%
   //
   // 2. SESSION FILTER:
   //    - Optimal hours: 7, 9, 10, 13, 14, 19, 20 UTC (>55% historical WR)
   //    - Avoids: 0, 1, 8, 15, 17 UTC (<45% historical WR)
   //    - Increases win rate from 63% → 68%
   //
   // RESULT: Server only sends BUY/SELL for high-probability setups
   //         Most signals will be HOLD (70-80% filtered)
   //         Expected win rate: ~68% (vs 51% unfiltered)
   //
   // EA's role: Execute server signals with proper risk management
   // ============================================================================
   
   // Check for error in response
   if(StringFind(response, "\"error\":") >= 0)
   {
      string errorMsg = ExtractString(response, "\"message\":\"", "\"");
      Print("SERVER ERROR: ", errorMsg);
      return;
   }
   
   // Parse inference server response format:
   // {"prediction": "BUY", "confidence": 0.85, "probabilities": {"SELL": 0.05, "HOLD": 0.10, "BUY": 0.85}}
   
   string signal = "";
   double confidence = 0.0;
   double prob_sell = 0.0;
   double prob_hold = 0.0;
   double prob_buy = 0.0;
   
   // Extract prediction (signal)
   signal = ExtractString(response, "\"prediction\":\"", "\"");
   
   if(ShowDebugInfo && signal != "")
      Print("Extracted prediction: ", signal);
   
   // Extract confidence
   int confPos = StringFind(response, "\"confidence\":");
   if(confPos >= 0)
   {
      int confEnd = StringFind(response, ",", confPos);
      if(confEnd < 0) confEnd = StringFind(response, "}", confPos);
      string confStr = StringSubstr(response, confPos + 13, confEnd - confPos - 13);
      confidence = StringToDouble(confStr);
      
      if(ShowDebugInfo)
         Print("Extracted confidence: ", confidence);
   }
   
   // Extract probabilities from nested object
   int probStartPos = StringFind(response, "\"probabilities\":{");
   if(probStartPos >= 0)
   {
      // Find the end of probabilities object
      int probEndPos = StringFind(response, "}", probStartPos);
      string probSection = StringSubstr(response, probStartPos, probEndPos - probStartPos + 1);
      
      // Extract SELL probability
      int sellPos = StringFind(probSection, "\"SELL\":");
      if(sellPos >= 0)
      {
         int sellEnd = StringFind(probSection, ",", sellPos);
         if(sellEnd < 0) sellEnd = StringFind(probSection, "}", sellPos);
         string sellStr = StringSubstr(probSection, sellPos + 7, sellEnd - sellPos - 7);
         prob_sell = StringToDouble(sellStr);
      }
      
      // Extract HOLD probability
      int holdPos = StringFind(probSection, "\"HOLD\":");
      if(holdPos >= 0)
      {
         int holdEnd = StringFind(probSection, ",", holdPos);
         if(holdEnd < 0) holdEnd = StringFind(probSection, "}", holdPos);
         string holdStr = StringSubstr(probSection, holdPos + 7, holdEnd - holdPos - 7);
         prob_hold = StringToDouble(holdStr);
      }
      
      // Extract BUY probability
      int buyPos = StringFind(probSection, "\"BUY\":");
      if(buyPos >= 0)
      {
         int buyEnd = StringFind(probSection, ",", buyPos);
         if(buyEnd < 0) buyEnd = StringFind(probSection, "}", buyPos);
         string buyStr = StringSubstr(probSection, buyPos + 6, buyEnd - buyPos - 6);
         prob_buy = StringToDouble(buyStr);
      }
      
      if(ShowDebugInfo)
         Print("Extracted probabilities - SELL: ", prob_sell, ", HOLD: ", prob_hold, ", BUY: ", prob_buy);
   }
   
   // Display results
   string predLabel = signal;
   if(predLabel == "") predLabel = "HOLD"; // Default if parsing failed
   
   Print("========================================");
   Print("HYBRID ENSEMBLE PREDICTION: ", predLabel);
   Print("Confidence: ", DoubleToString(confidence * 100, 1), "%");
   Print("Probabilities:");
   Print("  SELL: ", DoubleToString(prob_sell * 100, 1), "%");
   Print("  HOLD: ", DoubleToString(prob_hold * 100, 1), "%");
   Print("  BUY:  ", DoubleToString(prob_buy * 100, 1), "%");
   Print("========================================");
   
   // Display on chart with color coding
   DisplayPredictionInfo(predLabel, confidence, prob_sell, prob_hold, prob_buy);
   
   // Convert string signal to numeric prediction for trading logic
   // BUY = 1, SELL = -1, HOLD = 0
   int prediction = 0;
   if(predLabel == "BUY") prediction = 1;
   else if(predLabel == "SELL") prediction = -1;
   else if(predLabel == "HOLD") prediction = 0;
   
   // Check for exit signals first (always active)
   if(EnableTrading)
   {
      CheckExitSignals(prediction, confidence);
   }
   
   // Execute trade if confidence is sufficient AND conditions are good
   // NOTE: With server's quality+session filters, you should see:
   //       - 70-80% HOLD signals (filtered setups)
   //       - 20-30% BUY/SELL signals (high-quality setups only)
   //       - ~68% win rate on executed trades (vs 51% without filters)
   
   if(confidence >= MinConfidence && EnableTrading)
   {
      if(IsGoodTradingCondition())
      {
         Print("✅ EXECUTING: Server-approved high-quality setup");
         ExecuteTrade(prediction, confidence);
      }
   }
   else if(confidence < MinConfidence)
   {
      Print("SKIPPED: Confidence ", DoubleToString(confidence * 100, 1), "% below minimum ", DoubleToString(MinConfidence * 100, 0), "%");
      Print("   (Consider lowering MinConfidence to 0.50 with new server filters)");
   }
}

//+------------------------------------------------------------------+
//| Extract string value from JSON                                   |
//+------------------------------------------------------------------+
string ExtractString(string json, string key, string terminator)
{
   int startPos = StringFind(json, key);
   if(startPos < 0) return "";
   
   startPos += StringLen(key);
   
   // For JSON strings, we need to find the closing quote that's not escaped
   // Simple approach: find the next unescaped quote
   int endPos = startPos;
   bool found = false;
   
   while(endPos < StringLen(json))
   {
      endPos = StringFind(json, terminator, endPos);
      if(endPos < 0) break;
      
      // Check if this quote is escaped (preceded by \)
      if(endPos > 0 && StringGetCharacter(json, endPos - 1) == '\\')
      {
         // This quote is escaped, keep looking
         endPos++;
         continue;
      }
      
      // Found unescaped terminator
      found = true;
      break;
   }
   
   if(!found || endPos < 0) return "";
   
   string result = StringSubstr(json, startPos, endPos - startPos);
   
   // Unescape common JSON escape sequences
   StringReplace(result, "\\\"", "\"");
   StringReplace(result, "\\n", "\n");
   StringReplace(result, "\\t", "\t");
   StringReplace(result, "\\\\", "\\");
   
   return result;
}

//+------------------------------------------------------------------+
//| Extract model predictions from response                          |
//+------------------------------------------------------------------+
string ExtractModelPredictions(string response)
{
   string result = "";
   
   // Find models section
   int modelsPos = StringFind(response, "\"models\":");
   if(modelsPos < 0) return "N/A";
   
   // Extract RandomForest prediction
   // IMPORTANT: Models predict OUTCOME (WIN/LOSS/TIMEOUT), not trading signals!
   // 1 = WIN, 0 = TIMEOUT, -1 = LOSS
   int rfPos = StringFind(response, "\"RandomForest\":", modelsPos);
   if(rfPos >= 0)
   {
      string rfPred = StringSubstr(response, rfPos + 15, 2);  // Get 2 chars to handle negative
      int rfVal = (int)StringToInteger(rfPred);
      result += "RF:" + (rfVal == -1 ? "LOSS" : (rfVal == 0 ? "TIMEOUT" : "WIN"));
   }
   
   // Extract XGBoost prediction
   int xgbPos = StringFind(response, "\"XGBoost\":", modelsPos);
   if(xgbPos >= 0)
   {
      string xgbPred = StringSubstr(response, xgbPos + 10, 2);  // Get 2 chars to handle negative
      int xgbVal = (int)StringToInteger(xgbPred);
      if(result != "") result += " | ";
      result += "XGB:" + (xgbVal == -1 ? "LOSS" : (xgbVal == 0 ? "TIMEOUT" : "WIN"));
   }
   
   // Extract NeuralNetwork prediction
   int nnPos = StringFind(response, "\"NeuralNetwork\":", modelsPos);
   if(nnPos >= 0)
   {
      string nnPred = StringSubstr(response, nnPos + 16, 2);  // Get 2 chars to handle negative
      int nnVal = (int)StringToInteger(nnPred);
      if(result != "") result += " | ";
      result += "NN:" + (nnVal == -1 ? "LOSS" : (nnVal == 0 ? "TIMEOUT" : "WIN"));
   }
   
   return result;
}

//+------------------------------------------------------------------+
//| Execute trade based on prediction                                |
//+------------------------------------------------------------------+
void ExecuteTrade(int prediction, double confidence)
{
   // Close opposite positions first
   CloseOppositePositions(prediction);
   
   // Check if we already have a position in this direction
   if(HasPosition(prediction))
   {
      Print("Already have position in this direction");
      return;
   }
   
   // Calculate SL distance using the selected mode (ATR/Fixed/Percentage)
   double slPoints = CalculateSLDistance();
   
   // Calculate position size based on confidence, regime and recovery
   double lots = ComputeLotSize(prediction, confidence);
   
   // Normalize lot size
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   lots = MathMax(minLot, MathMin(maxLot, MathRound(lots / lotStep) * lotStep));
   
   // Calculate SL/TP
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   double sl = 0, tp = 0;
   
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   ConfigureTradeRequest(request);
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = lots;
   request.deviation = 10;
   request.magic = 123456;
   
   if(prediction == 1)  // BUY (server sends 1 for BUY)
   {
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      request.type = ORDER_TYPE_BUY;
      request.price = ask;
      
      // Use calculated SL distance in points
      sl = ask - slPoints * point;
      tp = ask + (slPoints * MinRiskReward) * point;
      
      request.sl = NormalizeDouble(sl, digits);
      request.tp = NormalizeDouble(tp, digits);
      
      Print("EXECUTING BUY: ", lots, " lots at ", ask, " (SL:", sl, " [", slPoints, " pts] TP:", tp, ")");
      LogTrade("BUY", confidence, ask, sl, tp);
   }
   else if(prediction == -1)  // SELL (server sends -1 for SELL)
   {
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      request.type = ORDER_TYPE_SELL;
      request.price = bid;
      
      // Use calculated SL distance in points
      sl = bid + slPoints * point;
      tp = bid - (slPoints * MinRiskReward) * point;
      
      request.sl = NormalizeDouble(sl, digits);
      request.tp = NormalizeDouble(tp, digits);
      
      Print("EXECUTING SELL: ", lots, " lots at ", bid, " (SL:", sl, " [", slPoints, " pts] TP:", tp, ")");
      LogTrade("SELL", confidence, bid, sl, tp);
   }
   else  // prediction == 0 (HOLD)
   {
      Print("HOLD signal - no trade");
      return;
   }
   
   // Send order
   if(!OrderSend(request, result))
   {
      Print("ERROR: OrderSend failed. Error: ", GetLastError());
      Print("  Retcode: ", result.retcode);
   }
   else
   {
      Print("SUCCESS: Order placed. Ticket: ", result.order);
   }
}


//+------------------------------------------------------------------+
//| Calculate Stop Loss distance in points based on mode             |
//+------------------------------------------------------------------+
double CalculateSLDistance()
{
   double slDistance = 0.0;
   
   switch(SLMode)
   {
      case atrBased:
         // ATR-based SL (adaptive to volatility)
         {
            int atr_handle = iATR(_Symbol, AnalysisTimeframe, ATRPeriod);
            if(atr_handle == INVALID_HANDLE)
            {
               Print("⚠️ ATR indicator failed, using fixed pips fallback");
               slDistance = StopLossPips * 10; // Fallback to fixed pips
               break;
            }
            
            double atr_array[];
            ArraySetAsSeries(atr_array, true);
            
            if(CopyBuffer(atr_handle, 0, 0, 1, atr_array) > 0)
            {
               double atr = atr_array[0];
               double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
               slDistance = (atr * ATRMultiplier) / point; // Convert ATR to points
               
               if(ShowDebugInfo)
                  PrintFormat("📏 ATR SL: ATR=%.5f × %.1f = %.1f points", atr, ATRMultiplier, slDistance);
            }
            else
            {
               Print("⚠️ Failed to get ATR value, using fixed pips");
               slDistance = StopLossPips * 10;
            }
            
            IndicatorRelease(atr_handle);
         }
         break;
         
      case fixedPips:
         // Fixed pips (manual)
         slDistance = StopLossPips * 10; // Convert pips to points
         if(ShowDebugInfo)
            PrintFormat("📏 Fixed SL: %d pips = %.1f points", StopLossPips, slDistance);
         break;
         
      case percentage:
         // Percentage of current price
         {
            double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
            double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
            slDistance = (price * SLPercentage / 100.0) / point;
            
            if(ShowDebugInfo)
               PrintFormat("📏 Percentage SL: %.1f%% of %.5f = %.1f points", SLPercentage, price, slDistance);
         }
         break;
         
      default:
         slDistance = StopLossPips * 10;
         break;
   }
   
   // Ensure minimum SL distance
   double minDistance = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   if(slDistance < minDistance)
   {
      if(ShowDebugInfo)
         PrintFormat("⚠️ SL too small: %.1f < %.1f, using minimum", slDistance, minDistance);
      slDistance = minDistance;
   }
   
   return slDistance;
}

//+------------------------------------------------------------------+
//| Compute lot size based on confidence, regime and recovery logic  |
//+------------------------------------------------------------------+
double ComputeLotSize(int prediction, double confidence)
{
   double riskPercent = 0.0;
   string modeName = "";
   
   // Determine risk based on RiskMode enum using switch
   switch(RiskMode)
   {
      case systemRisk:
         // System calculates risk based on confidence
         if(UseConfidenceScaling)
         {
            if(confidence >= ConfidenceHigh)
            {
               riskPercent = BASE_RISK_PERCENT * 4.0;  // 0.5% * 4 = 2.0%
               modeName = "System+Conf[HIGH]";
            }
            else if(confidence >= ConfidenceMed)
            {
               riskPercent = BASE_RISK_PERCENT * 3.0;  // 0.5% * 3 = 1.5%
               modeName = "System+Conf[MED]";
            }
            else if(confidence >= ConfidenceLow)
            {
               riskPercent = BASE_RISK_PERCENT * 2.0;  // 0.5% * 2 = 1.0%
               modeName = "System+Conf[LOW]";
            }
            else if(confidence >= MinConfidence)
            {
               riskPercent = BASE_RISK_PERCENT;        // 0.5%
               modeName = "System+Conf[MIN]";
            }
            else
            {
               riskPercent = 0.0;
               modeName = "System+Conf[SKIP]";
            }
            
            if(ShowDebugInfo)
               PrintFormat("🎯 Confidence Scaling: %.2f%% → Tier: %s (High≥%.0f%%, Med≥%.0f%%, Low≥%.0f%%, Min≥%.0f%%)", 
                          confidence*100, modeName, ConfidenceHigh*100, ConfidenceMed*100, ConfidenceLow*100, MinConfidence*100);
         }
         else
         {
            riskPercent = BASE_RISK_PERCENT;  // Fixed 0.5%
            modeName = "System";
         }
         break;
         
      case userRisk:
         // User defines their own risk
         riskPercent = UserRiskPercent;
         modeName = "Manual";
         break;
         
      case hardCodedRisk:
         // Hard-coded 1% risk (cannot be changed)
         riskPercent = FIXED_RISK_PERCENT;
         modeName = "Fixed1%";
         break;
         
      default:
         riskPercent = BASE_RISK_PERCENT;
         modeName = "Default";
         break;
   }
   
   // Apply recovery multiplier (all modes)
   if(EnableCompensatory && consecutiveLosses > 0)
   {
      int steps = MathMin(consecutiveLosses, RecoveryDepth);
      double recoveryMult = MathPow(RecoveryMultiplier, steps);
      riskPercent *= recoveryMult;
      
      if(ShowDebugInfo)
         PrintFormat("Recovery: %d losses × %.2f", consecutiveLosses, recoveryMult);
   }
   
   // Safety cap (all modes)
   if(riskPercent > MAX_RISK_PERCENT)
   {
      if(ShowDebugInfo)
         PrintFormat("Risk capped: %.2f%% → %.2f%%", riskPercent, MAX_RISK_PERCENT);
      riskPercent = MAX_RISK_PERCENT;
   }
   
   // Calculate SL distance using adaptive method
   double slPoints = CalculateSLDistance();
   double lots = CalculateLotsWithRisk(slPoints, riskPercent);
   
   // Normalize
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   
   lots = MathRound(lots / lotStep) * lotStep;
   lots = MathMax(minLot, MathMin(maxLot, lots));
   
   if(ShowDebugInfo)
      PrintFormat("💰 [%s] SL=%.1f pts | Risk=%.2f%% | Conf=%.0f%% | Lots=%.2f", 
                  modeName, slPoints, riskPercent, confidence*100, lots);
   
   return lots;
}

//+------------------------------------------------------------------+
//| Calculate lots with specific risk percentage                     |
//+------------------------------------------------------------------+
double CalculateLotsWithRisk(double slPoints, double riskPercent)
{
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double minvolume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxvolume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lotstep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   
   if(balance < 10)
      return minvolume;
   
   if(slPoints <= 0) slPoints = 0.001;
   
   // Calculate risk amount in account currency
   double riskAmount = balance * riskPercent / 100.0;
   
   // Get symbol properties
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double ticksize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double tickvalue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double contractsize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE);
   
   // Calculate money lost per lot for the SL distance
   // Standard formula: (SL in points) × (point value per lot)
   // Point value per lot = (point size) × (tick value) / (tick size)
   double pointValue = (point / ticksize) * tickvalue;
   double moneyPerLot = slPoints * pointValue;
   
   // Debug: show the calculation breakdown
   if(ShowDebugInfo)
      PrintFormat("🔍 Point Calc: Point=%.5f | TickSize=%.5f | TickVal=%.2f | PointVal=%.5f", 
                  point, ticksize, tickvalue, pointValue);
   
   // Calculate lot size: risk amount / money lost per lot
   double lots = 0.0;
   if(moneyPerLot > 0)
      lots = riskAmount / moneyPerLot;
   else
      lots = minvolume;
   
   // Normalize to lot step
   lots = MathFloor(lots / lotstep) * lotstep;
   
   // Apply min/max constraints
   if(lots < minvolume) lots = minvolume;
   if(lots > maxvolume) lots = maxvolume;
   
   // Optional margin safety check
   if(EnableMarginCheck)
   {
      double marginRequired = lots * SymbolInfoDouble(_Symbol, SYMBOL_MARGIN_INITIAL);
      double maxMargin = balance * (MaxMarginPercent / 100.0);
      
      if(ShowDebugInfo)
         PrintFormat("🔍 Margin Check: Lots=%.2f | MarginReq=$%.2f | MaxMargin=$%.2f (%.0f%% of balance)", 
                     lots, marginRequired, maxMargin, MaxMarginPercent);
      
      if(marginRequired > maxMargin)
      {
         double originalLots = lots;
         lots = maxMargin / SymbolInfoDouble(_Symbol, SYMBOL_MARGIN_INITIAL);
         lots = MathFloor(lots / lotstep) * lotstep;
         if(lots < minvolume) lots = minvolume;
         
         if(ShowDebugInfo)
            PrintFormat("⚠️ MARGIN LIMIT: Reduced from %.2f to %.2f lots (%.0f%% margin safety)", originalLots, lots, MaxMarginPercent);
      }
   }
   else if(ShowDebugInfo)
   {
      double marginRequired = lots * SymbolInfoDouble(_Symbol, SYMBOL_MARGIN_INITIAL);
      PrintFormat("🔍 Margin Check: DISABLED | Lots=%.2f | MarginReq=$%.2f", lots, marginRequired);
   }
   
   if(ShowDebugInfo)
      PrintFormat("💵 Lot Calc: Balance=%.2f | Risk=%.2f%% ($%.2f) | SL=%.1f pts | TickVal=%.2f | Contract=%.0f | MoneyPerLot=$%.2f | Lots=%.2f", 
                  balance, riskPercent, riskAmount, slPoints, tickvalue, contractsize, moneyPerLot, lots);
   
   return lots;
}

//+------------------------------------------------------------------+
//| Check if we have a position in this direction                    |
//+------------------------------------------------------------------+
bool HasPosition(int prediction)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionSelectByTicket(PositionGetTicket(i)))
      {
         if(PositionGetString(POSITION_SYMBOL) == _Symbol && 
            PositionGetInteger(POSITION_MAGIC) == inpMagic)
         {
            ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
            
            // FIXED: Correct mapping (server sends 1=BUY, -1=SELL, 0=HOLD)
            if(prediction == 1 && posType == POSITION_TYPE_BUY)
               return true;
            if(prediction == -1 && posType == POSITION_TYPE_SELL)
               return true;
         }
      }
   }
   return false;
}

//+------------------------------------------------------------------+
//| Close opposite positions                                         |
//+------------------------------------------------------------------+
void CloseOppositePositions(int prediction)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionSelectByTicket(PositionGetTicket(i)))
      {
         if(PositionGetString(POSITION_SYMBOL) == _Symbol)
         {
            ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
            
            // FIXED: Correct mapping (server sends 1=BUY, -1=SELL, 0=HOLD)
            bool shouldClose = false;
            if(prediction == 1 && posType == POSITION_TYPE_SELL)
               shouldClose = true;
            if(prediction == -1 && posType == POSITION_TYPE_BUY)
               shouldClose = true;
            
            if(shouldClose)
            {
               MqlTradeRequest request = {};
               MqlTradeResult result = {};
               ConfigureTradeRequest(request);
               
               request.action = TRADE_ACTION_DEAL;
               request.position = PositionGetInteger(POSITION_TICKET);
               request.symbol = _Symbol;
               request.volume = PositionGetDouble(POSITION_VOLUME);
               request.deviation = 10;
               request.type = (posType == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
               request.price = (posType == POSITION_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
               
               if(OrderSend(request, result))
                  Print("Closed opposite position: ", request.position);
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Extract SMC context from JSON response                           |
//+------------------------------------------------------------------+
void ExtractSMCContext(string response)
{
   // Reset SMC context
   smcOrderBlocks = "";
   smcFairValueGaps = "";
   smcStructure = "";
   smcRegime = "";
   
   // Find SMC context section
   int smcPos = StringFind(response, "\"smc_context\":");
   if(smcPos < 0)
   {
      smcOrderBlocks = "N/A";
      smcFairValueGaps = "N/A";
      smcStructure = "N/A";
      smcRegime = "N/A";
      return;
   }
   
   string smcSection = StringSubstr(response, smcPos, 1000);
   
   // Extract Order Blocks
   int obPos = StringFind(smcSection, "\"order_blocks\":");
   if(obPos >= 0)
   {
      string obSection = StringSubstr(smcSection, obPos, 200);
      bool bullishOB = (StringFind(obSection, "\"bullish_present\":true") >= 0);
      bool bearishOB = (StringFind(obSection, "\"bearish_present\":true") >= 0);
      
      // Extract quality
      int qualityPos = StringFind(obSection, "\"quality\":");
      double quality = 0.0;
      if(qualityPos >= 0)
      {
         int qualityEnd = StringFind(obSection, ",", qualityPos);
         if(qualityEnd < 0) qualityEnd = StringFind(obSection, "}", qualityPos);
         string qualityStr = StringSubstr(obSection, qualityPos + 10, qualityEnd - qualityPos - 10);
         quality = StringToDouble(qualityStr);
      }
      
      if(bullishOB && bearishOB)
         smcOrderBlocks = StringFormat("Bull+Bear OB (Q:%.2f)", quality);
      else if(bullishOB)
         smcOrderBlocks = StringFormat("Bullish OB (Q:%.2f)", quality);
      else if(bearishOB)
         smcOrderBlocks = StringFormat("Bearish OB (Q:%.2f)", quality);
      else
         smcOrderBlocks = "No OB";
   }
   
   // Extract Fair Value Gaps
   int fvgPos = StringFind(smcSection, "\"fair_value_gaps\":");
   if(fvgPos >= 0)
   {
      string fvgSection = StringSubstr(smcSection, fvgPos, 200);
      bool bullishFVG = (StringFind(fvgSection, "\"bullish_present\":true") >= 0);
      bool bearishFVG = (StringFind(fvgSection, "\"bearish_present\":true") >= 0);
      
      // Extract depth_atr
      int depthPos = StringFind(fvgSection, "\"depth_atr\":");
      double depth = 0.0;
      if(depthPos >= 0)
      {
         int depthEnd = StringFind(fvgSection, ",", depthPos);
         if(depthEnd < 0) depthEnd = StringFind(fvgSection, "}", depthPos);
         string depthStr = StringSubstr(fvgSection, depthPos + 12, depthEnd - depthPos - 12);
         depth = StringToDouble(depthStr);
      }
      
      if(bullishFVG && bearishFVG)
         smcFairValueGaps = StringFormat("Bull+Bear FVG (D:%.1f)", depth);
      else if(bullishFVG)
         smcFairValueGaps = StringFormat("Bullish FVG (D:%.1f)", depth);
      else if(bearishFVG)
         smcFairValueGaps = StringFormat("Bearish FVG (D:%.1f)", depth);
      else
         smcFairValueGaps = "No FVG";
   }
   
   // Extract Structure
   int structPos = StringFind(smcSection, "\"structure\":");
   if(structPos >= 0)
   {
      string structSection = StringSubstr(smcSection, structPos, 200);
      bool bosWick = (StringFind(structSection, "\"bos_wick_confirmed\":true") >= 0);
      bool bosClose = (StringFind(structSection, "\"bos_close_confirmed\":true") >= 0);
      bool choch = (StringFind(structSection, "\"choch_detected\":true") >= 0);
      
      if(bosClose)
         smcStructure = "BOS Close";
      else if(bosWick)
         smcStructure = "BOS Wick";
      else if(choch)
         smcStructure = "CHOCH";
      else
         smcStructure = "No Break";
   }
   
   // Extract Regime
   int regimePos = StringFind(smcSection, "\"regime\":");
   if(regimePos >= 0)
   {
      string regimeSection = StringSubstr(smcSection, regimePos, 200);
      
      // Extract regime label
      int labelPos = StringFind(regimeSection, "\"regime_label\":\"");
      if(labelPos >= 0)
      {
         int labelEnd = StringFind(regimeSection, "\"", labelPos + 16);
         smcRegime = StringSubstr(regimeSection, labelPos + 16, labelEnd - labelPos - 16);
      }
      
      // Extract trend bias
      int biasPos = StringFind(regimeSection, "\"trend_bias\":");
      if(biasPos >= 0)
      {
         int biasEnd = StringFind(regimeSection, ",", biasPos);
         if(biasEnd < 0) biasEnd = StringFind(regimeSection, "}", biasPos);
         string biasStr = StringSubstr(regimeSection, biasPos + 13, biasEnd - biasPos - 13);
         double bias = StringToDouble(biasStr);
         
         // Add bias value to regime
         smcRegime += StringFormat(" (%.2f)", bias);
      }
      
      // Extract volatility
      int volPos = StringFind(regimeSection, "\"volatility\":\"");
      if(volPos >= 0)
      {
         int volEnd = StringFind(regimeSection, "\"", volPos + 14);
         string volatility = StringSubstr(regimeSection, volPos + 14, volEnd - volPos - 14);
         smcRegime += " " + volatility + " Vol";
      }
   }
}

//+------------------------------------------------------------------+
//| Extract Indicators from response                                 |
//+------------------------------------------------------------------+
string ExtractIndicators(string response)
{
   string result = "";
   
   // Find indicators section
   int indPos = StringFind(response, "\"indicators\":");
   if(indPos < 0) return "";
   
   string indSection = StringSubstr(response, indPos, 300);
   
   // Extract RSI
   int rsiPos = StringFind(indSection, "\"rsi\":");
   if(rsiPos >= 0)
   {
      int rsiEnd = StringFind(indSection, ",", rsiPos);
      if(rsiEnd < 0) rsiEnd = StringFind(indSection, "}", rsiPos);
      string rsiStr = StringSubstr(indSection, rsiPos + 6, rsiEnd - rsiPos - 6);
      double rsi = StringToDouble(rsiStr);
      result += StringFormat("RSI:%.1f", rsi);
   }
   
   // Extract MACD Histogram
   int macdHistPos = StringFind(indSection, "\"macd_hist\":");
   if(macdHistPos >= 0)
   {
      int macdHistEnd = StringFind(indSection, ",", macdHistPos);
      if(macdHistEnd < 0) macdHistEnd = StringFind(indSection, "}", macdHistPos);
      string macdHistStr = StringSubstr(indSection, macdHistPos + 12, macdHistEnd - macdHistPos - 12);
      double macdHist = StringToDouble(macdHistStr);
      if(result != "") result += " | ";
      result += StringFormat("MACD:%s%.3f", (macdHist >= 0 ? "+" : ""), macdHist);
   }
   
   // Extract ATR
   int atrPos = StringFind(indSection, "\"atr\":");
   if(atrPos >= 0)
   {
      int atrEnd = StringFind(indSection, ",", atrPos);
      if(atrEnd < 0) atrEnd = StringFind(indSection, "}", atrPos);
      string atrStr = StringSubstr(indSection, atrPos + 6, atrEnd - atrPos - 6);
      double atr = StringToDouble(atrStr);
      if(result != "") result += " | ";
      result += StringFormat("ATR:%.5f", atr);
   }
   
   // Extract Momentum
   int momPos = StringFind(indSection, "\"momentum\":");
   if(momPos >= 0)
   {
      int momEnd = StringFind(indSection, ",", momPos);
      if(momEnd < 0) momEnd = StringFind(indSection, "}", momPos);
      string momStr = StringSubstr(indSection, momPos + 11, momEnd - momPos - 11);
      double momentum = StringToDouble(momStr);
      if(result != "") result += " | ";
      result += StringFormat("Mom:%s%.3f", (momentum >= 0 ? "+" : ""), momentum);
   }
   
   // Extract Volume MA Ratio
   int volPos = StringFind(indSection, "\"volume_ma_ratio\":");
   if(volPos >= 0)
   {
      int volEnd = StringFind(indSection, ",", volPos);
      if(volEnd < 0) volEnd = StringFind(indSection, "}", volPos);
      string volStr = StringSubstr(indSection, volPos + 18, volEnd - volPos - 18);
      double volRatio = StringToDouble(volStr);
      if(result != "") result += " | ";
      result += StringFormat("Vol:%.2fx", volRatio);
   }
   
   return result;
}

//+------------------------------------------------------------------+
//| Display prediction info on chart with color-coded indicators     |
//+------------------------------------------------------------------+
//| Color Scheme:                                                    |
//|   🟢 BUY      = Bright Green (High confidence bullish signal)   |
//|   🔴 SELL     = Bright Red (High confidence bearish signal)      |
//|   🟠 HOLD     = Orange (Neutral/no clear direction)             |
//|   🔵 SKIP     = Blue (Below minimum confidence threshold)       |
//+------------------------------------------------------------------+
void DisplayPredictionInfo(string predLabel, double confidence, double prob_sell, double prob_hold, double prob_buy)
{
   // Color scheme implementation
   // BUY = Bright Green (clrLime)
   // SELL = Red (clrRed) 
   // HOLD = Orange (clrOrange)
   // Below Confidence = Blue (clrDodgerBlue)
   
   color predColor = clrDodgerBlue;
   if(confidence >= MinConfidence)
   {
      if(predLabel == "BUY")
         predColor = clrLime;           // Bright Green
      else if(predLabel == "SELL")
         predColor = clrRed;            // Red
      else
         predColor = clrOrange;          // Orange (HOLD)
   }
   else
   {
      predColor = clrDodgerBlue;         // Blue (below confidence)
   }
   
   // Confidence color
   color confColor = clrDodgerBlue;
   if(confidence >= MinConfidence)
   {
      if(confidence >= 0.80)
         confColor = clrLime;            // Very high confidence
      else if(confidence >= 0.70)
         confColor = clrYellow;          // High confidence
      else
         confColor = clrOrange;          // Medium confidence
   }
   else
   {
      confColor = clrDodgerBlue;         // Below minimum
   }
   
   // Build comment with color indicators
   string comment = "";
   comment += "╔════════════════════════════════════════╗\n";
   comment += "║  HYBRID ENSEMBLE - MACRO-TECHNICAL     ║\n";
   comment += "║         LSTM + XGBoost Model           ║\n";
   comment += "╚════════════════════════════════════════╝\n";
   comment += "\n";
   
   // System info
   comment += "📊 Analysis: Single Timeframe (M5)\n";
   comment += "📈 Candles: 250 bars for feature engineering\n";
   comment += "🔧 Features: 58 (55 Technical + 3 Macro)\n";
   comment += "🌐 Server: " + RestServerURL + "\n";
   comment += "\n";
   comment += "💡 Note: Regime and Multi-Timeframe Analysis active\n";
   comment += "\n";
   
   // Update mode
   string updateMode = "";
   if(UpdateIntervalSeconds == 0)
      updateMode = "New M5 Bar Only";
   else
      updateMode = "Every " + IntegerToString(updateSeconds) + " seconds";
   comment += "⏱️ Update: " + updateMode + "\n";
   comment += "\n";
   
   comment += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
   
   // Prediction with visual indicator and color coding
   string predIcon = "";
   string predStatus = "";
   
   if(confidence >= MinConfidence)
   {
      if(predLabel == "BUY")
      {
         predIcon = "🟢 ▲";              // Bright Green
         predStatus = " [STRONG BUY]";
      }
      else if(predLabel == "SELL")
      {
         predIcon = "🔴 ▼";              // Bright Crimson
         predStatus = " [STRONG SELL]";
      }
      else
      {
         predIcon = "🟠 ●";              // Orange
         predStatus = " [NEUTRAL]";
      }
   }
   else
   {
      predIcon = "🔵 ○";                 // Blue (below confidence)
      predStatus = " [LOW CONFIDENCE]";
   }
   
   comment += "PREDICTION: " + predIcon + " " + predLabel + predStatus + "\n";
   
   // Confidence with bar
   string confBar = "";
   int confBars = (int)MathRound(confidence * 20);
   for(int i = 0; i < 20; i++)
   {
      if(i < confBars)
         confBar += "█";
      else
         confBar += "░";
   }
   comment += "Confidence: " + DoubleToString(confidence * 100, 1) + "% [" + confBar + "]\n";
   comment += "\n";
   
   comment += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
   comment += "PROBABILITIES:\n";
   comment += "\n";
   
   // SELL probability with bar
   string sellBar = "";
   int sellBars = (int)MathRound(prob_sell * 20);
   for(int i = 0; i < 20; i++)
      sellBar += (i < sellBars) ? "█" : "░";
   comment += "🔴 SELL:  " + StringFormat("%5.1f%%", prob_sell * 100) + " [" + sellBar + "]\n";
   
   // HOLD probability with bar
   string holdBar = "";
   int holdBars = (int)MathRound(prob_hold * 20);
   for(int i = 0; i < 20; i++)
      holdBar += (i < holdBars) ? "█" : "░";
   comment += "🟠 HOLD:  " + StringFormat("%5.1f%%", prob_hold * 100) + " [" + holdBar + "]\n";
   
   // BUY probability with bar
   string buyBar = "";
   int buyBars = (int)MathRound(prob_buy * 20);
   for(int i = 0; i < 20; i++)
      buyBar += (i < buyBars) ? "█" : "░";
   comment += "🟢 BUY:   " + StringFormat("%5.1f%%", prob_buy * 100) + " [" + buyBar + "]\n";
   comment += "\n";
   
   comment += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
   comment += "STATUS:\n";
   comment += "\n";
   
   // Trading status with color indicators
   if(EnableTrading)
   {
      if(confidence >= MinConfidence)
      {
         if(predLabel == "BUY")
            comment += "✅ 🟢 READY TO BUY\n";
         else if(predLabel == "SELL")
            comment += "✅ 🔴 READY TO SELL\n";
         else
            comment += "✅ 🟠 HOLD POSITION\n";
      }
      else
      {
         comment += "🔵 CONFIDENCE TOO LOW - NO TRADE\n";
      }
      comment += "🎯 Trading: ENABLED\n";
   }
   else
   {
      comment += "📊 Trading: DEMO MODE\n";
      if(confidence >= MinConfidence)
      {
         if(predLabel == "BUY")
            comment += "   Would signal: 🟢 BUY\n";
         else if(predLabel == "SELL")
            comment += "   Would signal: 🔴 SELL\n";
         else
            comment += "   Would signal: 🟠 HOLD\n";
      }
      else
      {
         comment += "   Would signal: 🔵 SKIP (Low Confidence)\n";
      }
   }
   
   comment += "📈 Min Confidence: " + DoubleToString(MinConfidence * 100, 0) + "%\n";
   comment += "\n";
   
   // News filter status
   if(NewsFilterOn)
   {
      comment += "📰 News Filter: " + (newsStatus != "" ? newsStatus : "Active") + "\n";
   }
   
   // Statistics
   double successRate = (requestCount > 0) ? (double)successCount / requestCount * 100.0 : 0.0;
   comment += "📊 Requests: " + IntegerToString(requestCount) + " | Success: " + IntegerToString(successCount);
   if(requestCount > 0)
      comment += " (" + DoubleToString(successRate, 1) + "%)";
   comment += "\n";
   
   // Recovery status
   if(EnableCompensatory && consecutiveLosses > 0)
   {
      comment += "⚠️ Recovery Mode: " + IntegerToString(consecutiveLosses) + " losses\n";
   }
   
   comment += "\n";
   comment += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
   
   // Time info
   comment += "🕐 Last Update: " + TimeToString(TimeCurrent(), TIME_DATE|TIME_MINUTES) + "\n";
   
   // Set chart foreground color based on prediction BEFORE Comment()
   ChartSetInteger(0, CHART_COLOR_FOREGROUND, predColor);
   
   // Display comment - text will inherit foreground color
   Comment(comment);
}

//+------------------------------------------------------------------+
//| Display colored information on chart                             |
//+------------------------------------------------------------------+
void DisplayColoredInfo(string predLabel, double confidence, double prob_sell, double prob_hold, double prob_buy, bool consensus, string modelPredictions, string explanation, string narrative)
{
   // Determine colors based on prediction
   color predColor = clrDodgerBlue;
   if(predLabel == "BUY")
      predColor = clrLime;
   else if(predLabel == "SELL")
      predColor = clrRed;
   else
      predColor = clrOrange;
   
   // Confidence color
   color confColor = clrDodgerBlue;
   if(confidence >= 0.70)
      confColor = clrLime;
   else if(confidence >= 0.60)
      confColor = clrYellow;
   else if(confidence >= 0.55)
      confColor = clrOrange;
   else
      confColor = clrRed;
   
   // Regime color
   color regimeColor = clrDodgerBlue;
   if(StringFind(smcRegime, "Bullish") >= 0)
      regimeColor = clrLime;
   else if(StringFind(smcRegime, "Bearish") >= 0)
      regimeColor = clrRed;
   else
      regimeColor = clrYellow;
   
   // Build comment - enhanced format with multi-timeframe info
   string comment = "";
   comment += "BLACK ICE V2 - MULTI-TIMEFRAME SMC\n";
   comment += "═══════════════════════════════════════\n";
   comment += "\n";
   comment += "Timeframes: M15 + H1 + H4 (250 bars each)\n";
   
   switch(UpdateIntervalSeconds){
      case 0:
         comment += "Update: New M15 Bar\n";
         break;
      case 1:
         comment += "Update: Every " + IntegerToString(updateSeconds) + "s\n";
         break;
   }
   
   comment += "\n";
   comment += "PREDICTION: " + predLabel;
   if(consensus) 
      comment += " [MAJORITY 2/3]";
   else
      comment += " [SPLIT VOTE]";
   comment += "\n";
   comment += "Confidence: " + DoubleToString(confidence * 100, 1) + "%\n";
   
   // Add explanation if available
   if(explanation != "")
   {
      comment += "\n";
      comment += "EXPLANATION:\n";
      comment += explanation + "\n";
   }
   
   comment += "───────────────────────────────────────\n";
   comment += "Probabilities:\n";
   comment += "  SELL: " + DoubleToString(prob_sell * 100, 1) + "%\n";
   comment += "  HOLD: " + DoubleToString(prob_hold * 100, 1) + "%\n";
   comment += "  BUY:  " + DoubleToString(prob_buy * 100, 1) + "%\n";
   comment += "\n";
   comment += "Model Predictions:\n";
   comment += "  " + modelPredictions + "\n";
   comment += "\n";
   comment += "SMC CONTEXT\n";
   comment += "───────────────────────────────────────\n";
   comment += "Order Blocks: " + smcOrderBlocks + "\n";
   comment += "Fair Value Gaps: " + smcFairValueGaps + "\n";
   comment += "Structure: " + smcStructure + "\n";
   comment += "Regime: " + smcRegime + "\n";
   comment += "\n";
   comment += "───────────────────────────────────────\n";
   comment += "Min Confidence: " + DoubleToString(MinConfidence * 100, 0) + "%\n";
   comment += "Trading: " + (EnableTrading ? "ENABLED" : "DEMO") + "\n";
   comment += "Requests: " + IntegerToString(requestCount) + " | Success: " + IntegerToString(successCount) + "\n";
   
   // Add news filter status
   if(NewsFilterOn)
   {
      comment += "\n";
      comment += "NEWS FILTER\n";
      comment += "───────────────────────────────────────\n";
      if(newsStatus != "")
         comment += newsStatus + "\n";
      else
         comment += "📰 Checking calendar...\n";
   }
   
   // Add signal status with color indicators
   comment += "\n";
   if(confidence >= MinConfidence)
   {
      if(predLabel == "BUY")
      {
         comment += ">>> SIGNAL: READY TO BUY <<<\n";
         if(StringFind(smcRegime, "Bullish") >= 0)
            comment += "    [ALIGNED WITH REGIME]\n";
         else if(StringFind(smcRegime, "Bearish") >= 0)
            comment += "    [COUNTER-TREND - CAUTION]\n";
      }
      else if(predLabel == "SELL")
      {
         comment += ">>> SIGNAL: READY TO SELL <<<\n";
         if(StringFind(smcRegime, "Bearish") >= 0)
            comment += "    [ALIGNED WITH REGIME]\n";
         else if(StringFind(smcRegime, "Bullish") >= 0)
            comment += "    [COUNTER-TREND - CAUTION]\n";
      }
      else
      {
         comment += ">>> SIGNAL: HOLD POSITION <<<\n";
      }
   }
   else
   {
      comment += ">>> CONFIDENCE TOO LOW - WAITING <<<\n";
   }
   
   // Set chart foreground color based on signal BEFORE Comment()
   if(confidence >= MinConfidence)
   {
      if(predLabel == "BUY")
         ChartSetInteger(0, CHART_COLOR_FOREGROUND, clrLime);
      else if(predLabel == "SELL")
         ChartSetInteger(0, CHART_COLOR_FOREGROUND, clrRed);
      else
         ChartSetInteger(0, CHART_COLOR_FOREGROUND, clrOrange);
   }
   else
   {
      ChartSetInteger(0, CHART_COLOR_FOREGROUND, clrDodgerBlue);
   }
   
   Comment(comment);
}
//+------------------------------------------------------------------+




//+------------------------------------------------------------------+
//| Initialize trade log file                                        |
//+------------------------------------------------------------------+
void InitializeTradeLog()
{
   int handle = FileOpen(logFileName, FILE_WRITE|FILE_CSV|FILE_COMMON);
   if(handle != INVALID_HANDLE)
   {
      // Write header
      FileWrite(handle, "Timestamp", "Symbol", "Action", "Confidence", "Entry", "SL", "TP", 
                "OB_Context", "FVG_Context", "Regime", "ATR", "Spread");
      FileClose(handle);
      Print("✅ Trade log initialized: ", logFileName);
   }
   else
   {
      Print("❌ Failed to create trade log");
   }
}

//+------------------------------------------------------------------+
//| Configure default fields for trade requests                      |
//+------------------------------------------------------------------+
void ConfigureTradeRequest(MqlTradeRequest &request)
{
   // Set order time type to GTC by default
   request.type_time = ORDER_TIME_GTC;
   
   // Auto-detect the correct filling mode for this broker/symbol
   int filling = (int)SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);
   
   // Select the first available filling mode
   if((filling & SYMBOL_FILLING_FOK) == SYMBOL_FILLING_FOK)
      request.type_filling = ORDER_FILLING_FOK;
   else if((filling & SYMBOL_FILLING_IOC) == SYMBOL_FILLING_IOC)
      request.type_filling = ORDER_FILLING_IOC;
   else
      request.type_filling = ORDER_FILLING_RETURN;  // Fallback for market execution
}

//+------------------------------------------------------------------+
//| Log trade to CSV                                                 |
//+------------------------------------------------------------------+
void LogTrade(string action, double confidence, double entry, double sl, double tp)
{
   int handle = FileOpen(logFileName, FILE_WRITE|FILE_READ|FILE_CSV|FILE_COMMON);
   if(handle != INVALID_HANDLE)
   {
      FileSeek(handle, 0, SEEK_END);
      
      int atr_handle = iATR(_Symbol, AnalysisTimeframe, 14);
      double atr = 0;
      if(atr_handle != INVALID_HANDLE)
      {
         double atr_buffer[1];
         if(CopyBuffer(atr_handle, 0, 0, 1, atr_buffer) == 1)
            atr = atr_buffer[0];
         IndicatorRelease(atr_handle);
      }
      
      double spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * SymbolInfoDouble(_Symbol, SYMBOL_POINT);
      
      FileWrite(handle,
         TimeToString(TimeCurrent(), TIME_DATE|TIME_MINUTES),
         _Symbol,
         action,
         DoubleToString(confidence, 3),
         DoubleToString(entry, _Digits),
         DoubleToString(sl, _Digits),
         DoubleToString(tp, _Digits),
         smcOrderBlocks,
         smcFairValueGaps,
         smcRegime,
         DoubleToString(atr, _Digits),
         DoubleToString(spread, _Digits)
      );
      
      FileClose(handle);
      Print("📝 Trade logged: ", action, " at ", entry);
   }
}

//+------------------------------------------------------------------+
//| Check if trading conditions are good                             |
//+------------------------------------------------------------------+
bool IsGoodTradingCondition()
{
   // Check if market is open (from core_functions.mqh)
   if(!IsMarketOpen())
   {
      Print("⚠️ Skipping trade: Market is closed");
      return false;
   }
   
   // Check for upcoming news (from core_functions.mqh)
   if(IsUpcomingNews())
   {
      Print("⚠️ Skipping trade: High-impact news approaching");
      return false;
   }
   
   // Check volatility
   int atr_handle = iATR(_Symbol, AnalysisTimeframe, 14);
   double atr = 0;
   double atr_avg = 0;
   
   if(atr_handle == INVALID_HANDLE)
      return true;  // If can't check, allow trade
   
   double atr_array[];
   ArraySetAsSeries(atr_array, true);
   
   // Get current ATR and 100-period average
   if(CopyBuffer(atr_handle, 0, 0, 100, atr_array) == 100)
   {
      atr = atr_array[0];
      
      double atr_sum = 0;
      for(int i = 0; i < 100; i++)
         atr_sum += atr_array[i];
      atr_avg = atr_sum / 100;
   }
   
   IndicatorRelease(atr_handle);
   
   if(atr < atr_avg * 0.5)
   {
      Print("⚠️ Skipping trade: Volatility too low (ATR: ", atr, " < ", atr_avg * 0.5, ")");
      return false;
   }
   
   // Check spread
   double spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   if(spread > atr * 0.3)
   {
      Print("⚠️ Skipping trade: Spread too wide (", spread, " > ", atr * 0.3, ")");
      return false;
   }
   
   // NOTE: Server now handles session filtering (optimal hours: 7, 9, 10, 13, 14, 19, 20 UTC)
   // Server avoids: 0, 1, 8, 15, 17 UTC (low win rate hours)
   // EA only does basic sanity check here
   
   MqlDateTime dt;
   TimeToStruct(TimeGMT(), dt);
   int hour = dt.hour;
   
   // Only block extreme dead hours (server handles the rest)
   if(hour >= 2 && hour <= 5)
   {
      Print("⚠️ Skipping trade: Extreme low liquidity hours (", hour, ":00 UTC)");
      Print("   Server session filter handles optimal hour selection");
      return false;
   }
   
   // If we get here, trust the server's session + quality filters
   // Server only sends BUY/SELL signals for:
   // 1. High-quality setups (OB_Displacement > 1.5 ATR, Quality > 0.3)
   // 2. Optimal trading hours (based on 63%+ historical win rate)
   
   return true;
}


//+------------------------------------------------------------------+
//| Check for exit signals on open positions                         |
//+------------------------------------------------------------------+
void CheckExitSignals(int prediction, double confidence)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionSelectByTicket(PositionGetTicket(i)))
      {
         if(PositionGetString(POSITION_SYMBOL) == _Symbol)
         {
            ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
            ulong ticket = PositionGetInteger(POSITION_TICKET);
            
            bool shouldExit = false;
            string exitReason = "";
            
            // FIXED: Correct mapping (server sends 1=BUY, -1=SELL, 0=HOLD)
            // Exit long if strong SELL signal
            if(posType == POSITION_TYPE_BUY && prediction == -1 && confidence >= 0.60)
            {
               shouldExit = true;
               exitReason = "AI Exit: Strong SELL signal";
            }
            
            // Exit short if strong BUY signal
            if(posType == POSITION_TYPE_SELL && prediction == 1 && confidence >= 0.60)
            {
               shouldExit = true;
               exitReason = "AI Exit: Strong BUY signal";
            }
            
            if(shouldExit)
            {
               MqlTradeRequest request = {};
               MqlTradeResult result = {};
               ConfigureTradeRequest(request);
               
               request.action = TRADE_ACTION_DEAL;
               request.position = ticket;
               request.symbol = _Symbol;
               request.volume = PositionGetDouble(POSITION_VOLUME);
               request.deviation = 10;
               request.type = (posType == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
               request.price = (posType == POSITION_TYPE_BUY) ? 
                  SymbolInfoDouble(_Symbol, SYMBOL_BID) : 
                  SymbolInfoDouble(_Symbol, SYMBOL_ASK);
               
               if(OrderSend(request, result))
               {
                  Print("✅ ", exitReason, " - Ticket: ", ticket);
                  LogTrade("EXIT_" + (posType == POSITION_TYPE_BUY ? "BUY" : "SELL"), 
                          confidence, request.price, 0, 0);
                 // Bookkeeping: assume exit was taken; we can't know pnl here without OnTradeTransaction
                 consecutiveLosses = MathMax(0, consecutiveLosses - 1); // conservatively reduce recovery
               }
            }
         }
      }
   }
}

// Placeholder: this should be hooked to OnTradeTransaction to get actual closed trade pnl
void OnTradeClosed(ulong ticket, double profit)
{
   lastTradeProfit = profit;
   if(profit < 0)
      consecutiveLosses++;
   else
      consecutiveLosses = MathMax(0, consecutiveLosses - 1);

   if(ShowDebugInfo)
      PrintFormat("Trade closed: ticket=%I64u profit=%.2f consecutiveLosses=%d", ticket, profit, consecutiveLosses);
}

//+------------------------------------------------------------------+
//| Draw Order Block zones on chart                                  |
//+------------------------------------------------------------------+
void DrawOrderBlockZones(string response)
{
   // Clear old OB rectangles
   ObjectDelete(0, "OB_Bullish");
   ObjectDelete(0, "OB_Bearish");
   
   // Find SMC context section
   int smcPos = StringFind(response, "\"smc_context\":");
   if(smcPos < 0) return;
   
   string smcSection = StringSubstr(response, smcPos, 1500);
   
   // Extract Order Blocks section
   int obPos = StringFind(smcSection, "\"order_blocks\":");
   if(obPos < 0) return;
   
   string obSection = StringSubstr(smcSection, obPos, 500);
   
   // Check if bullish OB present
   bool bullishOB = (StringFind(obSection, "\"bullish_present\":true") >= 0);
   if(bullishOB)
   {
      // Extract bullish_high
      int highPos = StringFind(obSection, "\"bullish_high\":");
      double bullishHigh = 0.0;
      if(highPos >= 0)
      {
         int highEnd = StringFind(obSection, ",", highPos);
         if(highEnd < 0) highEnd = StringFind(obSection, "}", highPos);
         string highStr = StringSubstr(obSection, highPos + 15, highEnd - highPos - 15);
         bullishHigh = StringToDouble(highStr);
      }
      
      // Extract bullish_low
      int lowPos = StringFind(obSection, "\"bullish_low\":");
      double bullishLow = 0.0;
      if(lowPos >= 0)
      {
         int lowEnd = StringFind(obSection, ",", lowPos);
         if(lowEnd < 0) lowEnd = StringFind(obSection, "}", lowPos);
         string lowStr = StringSubstr(obSection, lowPos + 14, lowEnd - lowPos - 14);
         bullishLow = StringToDouble(lowStr);
      }
      
      // Extract age
      int agePos = StringFind(obSection, "\"age\":");
      int age = 0;
      if(agePos >= 0)
      {
         int ageEnd = StringFind(obSection, ",", agePos);
         if(ageEnd < 0) ageEnd = StringFind(obSection, "}", agePos);
         string ageStr = StringSubstr(obSection, agePos + 6, ageEnd - agePos - 6);
         age = (int)StringToInteger(ageStr);
      }
      
      // Draw bullish OB rectangle if we have valid prices
      if(bullishHigh > 0 && bullishLow > 0 && bullishHigh > bullishLow)
      {
         datetime startTime = iTime(_Symbol, PERIOD_M15, age);
         datetime endTime = TimeCurrent() + PeriodSeconds(PERIOD_M15) * 100;
         
         ObjectCreate(0, "OB_Bullish", OBJ_RECTANGLE, 0, startTime, bullishHigh, endTime, bullishLow);
         ObjectSetInteger(0, "OB_Bullish", OBJPROP_COLOR, clrDodgerBlue);
         ObjectSetInteger(0, "OB_Bullish", OBJPROP_STYLE, STYLE_SOLID);
         ObjectSetInteger(0, "OB_Bullish", OBJPROP_WIDTH, 1);
         ObjectSetInteger(0, "OB_Bullish", OBJPROP_FILL, true);
         ObjectSetInteger(0, "OB_Bullish", OBJPROP_BACK, true);
         ObjectSetInteger(0, "OB_Bullish", OBJPROP_SELECTABLE, false);
         ObjectSetInteger(0, "OB_Bullish", OBJPROP_SELECTED, false);
         ObjectSetString(0, "OB_Bullish", OBJPROP_TOOLTIP, "Bullish Order Block");
      }
   }
   
   // Check if bearish OB present
   bool bearishOB = (StringFind(obSection, "\"bearish_present\":true") >= 0);
   if(bearishOB)
   {
      // Extract bearish_high
      int highPos = StringFind(obSection, "\"bearish_high\":");
      double bearishHigh = 0.0;
      if(highPos >= 0)
      {
         int highEnd = StringFind(obSection, ",", highPos);
         if(highEnd < 0) highEnd = StringFind(obSection, "}", highPos);
         string highStr = StringSubstr(obSection, highPos + 15, highEnd - highPos - 15);
         bearishHigh = StringToDouble(highStr);
      }
      
      // Extract bearish_low
      int lowPos = StringFind(obSection, "\"bearish_low\":");
      double bearishLow = 0.0;
      if(lowPos >= 0)
      {
         int lowEnd = StringFind(obSection, ",", lowPos);
         if(lowEnd < 0) lowEnd = StringFind(obSection, "}", lowPos);
         string lowStr = StringSubstr(obSection, lowPos + 14, lowEnd - lowPos - 14);
         bearishLow = StringToDouble(lowStr);
      }
      
      // Extract age
      int agePos = StringFind(obSection, "\"age\":");
      int age = 0;
      if(agePos >= 0)
      {
         int ageEnd = StringFind(obSection, ",", agePos);
         if(ageEnd < 0) ageEnd = StringFind(obSection, "}", agePos);
         string ageStr = StringSubstr(obSection, agePos + 6, ageEnd - agePos - 6);
         age = (int)StringToInteger(ageStr);
      }
      
      // Draw bearish OB rectangle if we have valid prices
      if(bearishHigh > 0 && bearishLow > 0 && bearishHigh > bearishLow)
      {
         datetime startTime = iTime(_Symbol, PERIOD_M15, age);
         datetime endTime = TimeCurrent() + PeriodSeconds(PERIOD_M15) * 100;
         
         ObjectCreate(0, "OB_Bearish", OBJ_RECTANGLE, 0, startTime, bearishHigh, endTime, bearishLow);
         ObjectSetInteger(0, "OB_Bearish", OBJPROP_COLOR, clrCrimson);
         ObjectSetInteger(0, "OB_Bearish", OBJPROP_STYLE, STYLE_SOLID);
         ObjectSetInteger(0, "OB_Bearish", OBJPROP_WIDTH, 1);
         ObjectSetInteger(0, "OB_Bearish", OBJPROP_FILL, true);
         ObjectSetInteger(0, "OB_Bearish", OBJPROP_BACK, true);
         ObjectSetInteger(0, "OB_Bearish", OBJPROP_SELECTABLE, false);
         ObjectSetInteger(0, "OB_Bearish", OBJPROP_SELECTED, false);
         ObjectSetString(0, "OB_Bearish", OBJPROP_TOOLTIP, "Bearish Order Block");
      }
   }
   
   // Draw mitigation marker if present
   ObjectDelete(0, "OB_Mitigation");
   
   // Extract mitigation_time
   int mitTimePos = StringFind(obSection, "\"mitigation_time\":\"");
   if(mitTimePos >= 0)
   {
      int mitTimeEnd = StringFind(obSection, "\"", mitTimePos + 19);
      string mitTimeStr = StringSubstr(obSection, mitTimePos + 19, mitTimeEnd - mitTimePos - 19);
      
      // Extract mitigation_price
      int mitPricePos = StringFind(obSection, "\"mitigation_price\":");
      double mitPrice = 0.0;
      if(mitPricePos >= 0)
      {
         int mitPriceEnd = StringFind(obSection, ",", mitPricePos);
         if(mitPriceEnd < 0) mitPriceEnd = StringFind(obSection, "}", mitPricePos);
         string mitPriceStr = StringSubstr(obSection, mitPricePos + 19, mitPriceEnd - mitPricePos - 19);
         mitPrice = StringToDouble(mitPriceStr);
      }
      
      // Draw X marker if we have valid data
      if(mitTimeStr != "" && mitPrice > 0)
      {
         datetime mitTime = StringToTime(mitTimeStr);
         if(mitTime > 0)
         {
            ObjectCreate(0, "OB_Mitigation", OBJ_ARROW, 0, mitTime, mitPrice);
            ObjectSetInteger(0, "OB_Mitigation", OBJPROP_ARROWCODE, 251); // Wingdings X
            ObjectSetInteger(0, "OB_Mitigation", OBJPROP_COLOR, clrYellow);
            ObjectSetInteger(0, "OB_Mitigation", OBJPROP_WIDTH, 3);
            ObjectSetInteger(0, "OB_Mitigation", OBJPROP_SELECTABLE, false);
            ObjectSetString(0, "OB_Mitigation", OBJPROP_TOOLTIP, "OB Mitigation Point");
         }
      }
   }
}


//+------------------------------------------------------------------+

