//+------------------------------------------------------------------+
//| External parameters for script operation                         |
//+------------------------------------------------------------------+
// Beginning of the period of the general population
input datetime Start = D'2020.06.06 00:00:00';  
// End of the period of the general population
input datetime End = D'2025.06.06 00:00:00';    
// Timeframe for data loading
input ENUM_TIMEFRAMES TimeFrame = PERIOD_M5;    
// Number of historical bars in one pattern
input int BarsToLine = 40;                 
// File name for recording the training sample
input string   StudyFileName = "study_data.csv";
// File name for recording the test sample
input string   TestFileName  = "test_data.csv"; 
// Data normalization flag
input bool NormalizeData = true;            
//+------------------------------------------------------------------+
//| Beginning of the script program                                  |
//+------------------------------------------------------------------+
void OnStart(void)
{
//--- Connect indicators to the chart
int h_ZZ = iCustom(_Symbol, TimeFrame, "Examples\\ZigZag.ex5", 48, 1, 47);
int h_RSI = iRSI(_Symbol, TimeFrame, 12, PRICE_TYPICAL);
int h_MACD = iMACD(_Symbol, TimeFrame, 12, 48, 12, PRICE_TYPICAL);
double close[];
int copied = CopyClose(_Symbol, TimeFrame, Start, End, close);
if(copied <= 0)
  {
   PrintFormat("Error copying Close data: %d. Check if history exists for %s from %s to %s", GetLastError(), _Symbol, TimeToString(Start), TimeToString(End));
   return;
  }
PrintFormat("Successfully loaded %d bars of Close data", copied);
//--- Load indicator data into dynamic arrays
double zz[], macd_main[], macd_signal[], rsi[];
datetime end_zz = End + PeriodSeconds(TimeFrame) * 500;
if(h_ZZ == INVALID_HANDLE || 
CopyBuffer(h_ZZ, 0, Start, end_zz, zz) <= 0)
{
      PrintFormat("Error loading indicator %s data", "ZigZag");
      return;
     }
   if(h_RSI == INVALID_HANDLE || 
      CopyBuffer(h_RSI, 0, Start, End, rsi) <= 0)
     {
      PrintFormat("Error loading indicator %s data", "RSI");
      return;
     }
   if(h_MACD == INVALID_HANDLE || 
      CopyBuffer(h_MACD, MAIN_LINE, Start, End, macd_main) <= 0 ||
      CopyBuffer(h_MACD, SIGNAL_LINE, Start, End, macd_signal) <= 0)
     {
      PrintFormat("Error loading indicator %s data", "MACD");
      return;
     }
   int total = ArraySize(close);

   // Adjust total to match the smallest indicator buffer
   if(ArraySize(rsi) < total) total = ArraySize(rsi);
   if(ArraySize(macd_main) < total) total = ArraySize(macd_main);
   if(ArraySize(macd_signal) < total) total = ArraySize(macd_signal);

   double target1[], target2[], macd_delta[], test[];
   if(ArrayResize(target1, total) <= 0 || 
      ArrayResize(target2, total) <= 0 ||
      ArrayResize(test, total) <= 0 || 
      ArrayResize(macd_delta, total) <= 0)
      return;
//--- Calculate targets: direction and distance 
//--- to the nearest extremum
   double extremum = -1;
   for(int i = ArraySize(zz) - 2; i >= 0; i--)
     {
      if(zz[i + 1] > 0 && zz[i + 1] != EMPTY_VALUE)
         extremum = zz[i + 1];
      if(i >= total)
         continue;
      target2[i] = extremum - close[i];
      target1[i] = (target2[i] >= 0 ? 1 : -1);
     }
  
  //--- Clean EMPTY_VALUE before calculating delta and normalization
   for(int i = 0; i < total; i++)
     {
      if(rsi[i] == EMPTY_VALUE || rsi[i] > 100 || rsi[i] < 0)
         rsi[i] = 50.0;  // Default to neutral
      if(macd_main[i] == EMPTY_VALUE || !MathIsValidNumber(macd_main[i]))
         macd_main[i] = 0.0;
      if(macd_signal[i] == EMPTY_VALUE || !MathIsValidNumber(macd_signal[i]))
         macd_signal[i] = 0.0;
     }
  
  //--- Calculate macd_delta AFTER cleaning
   for(int i = 0; i < total; i++)
     {
      macd_delta[i] = macd_main[i] - macd_signal[i];
     }
  
  //--- Data normalization
   if(NormalizeData)
     {
      double main_norm = MathMax(MathAbs(macd_main[ArrayMinimum(macd_main)]),
                                         macd_main[ArrayMaximum(macd_main)]);
      double sign_norm = MathMax(MathAbs(macd_signal[ArrayMinimum(macd_signal)]),
                                         macd_signal[ArrayMaximum(macd_signal)]);
      double delt_norm = MathMax(MathAbs(macd_delta[ArrayMinimum(macd_delta)]),
                                         macd_delta[ArrayMaximum(macd_delta)]);
      
      // Normalize target2 (price distance) to prevent huge loss values
      double target2_norm = MathMax(MathAbs(target2[ArrayMinimum(target2)]),
                                            MathAbs(target2[ArrayMaximum(target2)]));
      
      for(int i = 0; i < total; i++)
        {
         rsi[i] = (rsi[i] - 50.0) / 50.0;
         if(main_norm != 0)
            macd_main[i] /= main_norm;
         else
            macd_main[i] = 0;
            
         if(sign_norm != 0)
            macd_signal[i] /= sign_norm;
         else
            macd_signal[i] = 0;
            
         if(delt_norm != 0)
            macd_delta[i] /= delt_norm;
         else
            macd_delta[i] = 0;
         
         // Normalize target2 to [-1, 1] range
         if(target2_norm != 0)
            target2[i] /= target2_norm;
         else
            target2[i] = 0;
        }
      
      PrintFormat("Normalization factors: MACD_main=%.2f, MACD_signal=%.2f, MACD_delta=%.2f, Target2=%.2f", 
                  main_norm, sign_norm, delt_norm, target2_norm);
}
//--- Generate randomly the data indices for the test sample
ArrayInitialize(test, 0);
   int for_test = (int)((total - BarsToLine) * 0.2);
   for(int i = 0; i < for_test; i++)
     {
      int t = (int)((double)(MathRand() * MathRand()) / MathPow(32767.0, 2) * 
                    (total - 1 - BarsToLine)) + BarsToLine;
      if(test[t] == 1)
        {
         i--;
         continue;
        }
      test[t] = 1;
     }
    //--- Open the training sample file for writing
   int Study = FileOpen(StudyFileName, FILE_WRITE | 
                                       FILE_CSV | 
                                       FILE_ANSI, ",", CP_UTF8);
   if(Study == INVALID_HANDLE)
     {
      PrintFormat("Error opening file %s: %d", StudyFileName, GetLastError());
      return;
     }
//--- Open the test sample file for writing
   int Test = FileOpen(TestFileName, FILE_WRITE | 
                                     FILE_CSV | 
                                     FILE_ANSI, ",", CP_UTF8);
   if(Test == INVALID_HANDLE)
     {
      PrintFormat("Error opening file %s: %d", TestFileName, GetLastError());
      return;
     }
   //--- Write samples to files
   int skipped = 0;
   for(int i = BarsToLine - 1; i < total; i++)
     {
      Comment(StringFormat("%.2f%%", i * 100.0 / (double)(total - BarsToLine)));
      
      // Skip patterns where indicators still have EMPTY_VALUE
      bool has_valid_data = true;
      for(int j = i - BarsToLine + 1; j <= i; j++)
        {
         if(rsi[j] == EMPTY_VALUE)
           {
            has_valid_data = false;
            break;
           }
         if(macd_main[j] == EMPTY_VALUE)
           {
            has_valid_data = false;
            break;
           }
         if(macd_signal[j] == EMPTY_VALUE)
           {
            has_valid_data = false;
            break;
           }
        }
      
      if(!has_valid_data)
        {
         skipped++;
         continue;
        }
      
      if(!WriteData(target1, target2, rsi, macd_main, macd_signal, macd_delta, i,
                                      BarsToLine, (test[i] == 1 ? Test : Study)))
        {
         PrintFormat("Error to write data: %d", GetLastError());
         break;
        }
     }
   
   if(skipped > 0)
      PrintFormat("Skipped %d patterns with invalid indicator data", skipped);
//--- Close the files
   Comment("");
   FileFlush(Study);
   FileClose(Study);
   FileFlush(Test);
   FileClose(Test);
   PrintFormat("Study data saved to file %s\\MQL5\\Files\\%s",
               TerminalInfoString(TERMINAL_DATA_PATH), StudyFileName);
   PrintFormat("Test data saved to file %s\\MQL5\\Files\\%s",
               TerminalInfoString(TERMINAL_DATA_PATH), TestFileName);
  }
//+------------------------------------------------------------------+
//| Function for writing a pattern to a file                         |
//+------------------------------------------------------------------+
bool WriteData(double &target1[], // Buffer 1 of target values
               double &target2[], // Buffer 2 target values
               double &data1[],   // Buffer 1 of historical data
               double &data2[],   // Buffer 2 of historical data
               double &data3[],   // Buffer 2 of historical data
               double &data4[],   // Buffer 2 of historical data
               int cur_bar,       // Current bar of the end of the pattern
               int bars,          // Number of historical bars 
                                  // in one pattern
               int handle)        // Handle of the file to be written
  {
//--- check the file handle
   if(handle == INVALID_HANDLE)
     {
      Print("Invalid Handle");
      return false;
     }
//--- determine the index of the first record of the historical data of the pattern
   int start = cur_bar - bars + 1;
   if(start < 0)
     {
      Print("Too small current bar");
      return false;
     }
   //--- Check the correctness of the index of the data and the data written to the file 
   int size1 = ArraySize(data1);
   int size2 = ArraySize(data2);
   int size3 = ArraySize(data3);
   int size4 = ArraySize(data4);
   int sizet1 = ArraySize(target1);
   int sizet2 = ArraySize(target2);
   string pattern = (string)(start < size1 ? data1[start] : 0.0) + "," +
                    (string)(start < size2 ? data2[start] : 0.0) + "," +
                    (string)(start < size3 ? data3[start] : 0.0) + "," +
                    (string)(start < size4 ? data4[start] : 0.0);
   for(int i = start + 1; i <= cur_bar; i++)
     {
      pattern = pattern + "," + (string)(i < size1 ? data1[i] : 0.0) + "," +
                                (string)(i < size2 ? data2[i] : 0.0) + "," +
                                (string)(i < size3 ? data3[i] : 0.0) + "," +
                                (string)(i < size4 ? data4[i] : 0.0);
     }
   return (FileWrite(handle, pattern, 
                    (double)(cur_bar < sizet1 ? target1[cur_bar] : 0),
                    (double)(cur_bar < sizet2 ? target2[cur_bar] : 0)) > 0);
  }

   