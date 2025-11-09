//+------------------------------------------------------------------+
//| core_functions.mqh                                               |
//| Modularized core functions (news filter, trailing stops, etc.)   |
//+------------------------------------------------------------------+

// Required trading helper includes (PositionInfo and Trade)
#include <Trade\PositionInfo.mqh>
#include <Trade\Trade.mqh> 
#include "Auron AI.mq5"

bool           TrDisableNews;
datetime       LastNewsAvoided;
string         newsToAvoid[];
int            barsRangeStart, barsToCount, buyTotal, sellTotal; 

string TradingEnabledComm;

///+------------------------------------------------------------------+
//| Function to Find High                                            |
//+------------------------------------------------------------------+
double findHigh(){
   double highestHigh = 0; 
      for (int i = 0; i < 200; i++){
         double high = iHigh(_Symbol, AnalysisTimeframe, i ); 
         if(i > BarsN && iHighest(_Symbol, AnalysisTimeframe, MODE_HIGH, BarsN*2+1, i-BarsN) == i){
            if(high > highestHigh){
               return high; 
            }
         }
         highestHigh = MathMax(high, highestHigh); 
      }
    return -1; 
}
//+------------------------------------------------------------------+
//| Function to Find Low                                             |
//+------------------------------------------------------------------+
double findLow(){
   double lowestLow = DBL_MAX; 
      for (int i = 0; i < 200; i++){
         double low = iLow(_Symbol, AnalysisTimeframe, i); 
         if(i > BarsN && iLowest(_Symbol, AnalysisTimeframe, MODE_LOW, BarsN*2+1, i-BarsN) == i){
            if(low < lowestLow){
               return low; 
         }
      }
      lowestLow = MathMin(low, lowestLow); 
    }
    return -1; 
}
//+------------------------------------------------------------------+
//| Function to Check Upcoming News                                  |
//+------------------------------------------------------------------+
bool IsUpcomingNews(){
   
   if(NewsFilterOn == false) return (false); 
   
   if(TrDisableNews && TimeCurrent()-LastNewsAvoided < StartTradingMin*PeriodSeconds(PERIOD_M1)) {
      int minutesRemaining = (int)((StartTradingMin*60 - (TimeCurrent()-LastNewsAvoided))/60);
      newsStatus = "⏳ Waiting after news\n   Resume trading in " + IntegerToString(minutesRemaining) + " minutes";
      return true;
   }
   
   TrDisableNews = false; 
   
   string sep; 
   switch (separator){
      case 0: sep = ","; break; 
      case 1: sep = ";";
   }
   
   sep_code = StringGetCharacter(sep, 0); 
   
   int k = StringSplit(keyNews, sep_code,newsToAvoid); 
   
   if(k == 0) {
      newsStatus = "⚠️ No keywords configured";
      return false;
   }
   
   MqlCalendarValue values[]; 
   datetime start_time = TimeCurrent(); 
   datetime end_time = start_time + PeriodSeconds(PERIOD_D1)*DaysNewsLookup; 
   
   CalendarValueHistory(values, start_time, end_time, NULL, NULL); 
   
   if(ArraySize(values) == 0) {
      newsStatus = "📰 No calendar events found";
      return false;
   }
   
   // Silently scan events (only log if news found)
   int matchedEvents = 0;
   string nextNewsInfo = "";
   
   for (int i = 0; i < ArraySize(values); i++){
       MqlCalendarEvent event; 
       CalendarEventById(values[i].event_id, event); 
       MqlCalendarCountry country; 
       CalendarCountryById(event.country_id, country); 
       
       if(StringFind(NewsCurrencied, country.currency) < 0) continue; 
       
         for (int j = 0; j<k; j++){
            string currentevent = newsToAvoid[j]; 
            string currentnews = event.name; 
            if(StringFind(currentnews, currentevent) < 0) continue; 
            
            matchedEvents++;
            int minutesUntil = (int)((values[i].time - TimeCurrent()) / 60);
            
            // Get impact level
            string impactIcon = "";
            string impactText = "";
            switch(event.importance)
            {
               case CALENDAR_IMPORTANCE_NONE:
                  impactIcon = "⚪";
                  impactText = "Low";
                  break;
               case CALENDAR_IMPORTANCE_LOW:
                  impactIcon = "🟡";
                  impactText = "Medium";
                  break;
               case CALENDAR_IMPORTANCE_MODERATE:
                  impactIcon = "🟠";
                  impactText = "High";
                  break;
               case CALENDAR_IMPORTANCE_HIGH:
                  impactIcon = "🔴";
                  impactText = "Critical";
                  break;
               default:
                  impactIcon = "⚪";
                  impactText = "Unknown";
                  break;
            }
            
            // Only log if news is within 24 hours
            if(minutesUntil < 1440)
               Print("📰 Found: ", country.currency, " - ", event.name, " [", impactText, "] in ", minutesUntil, " minutes (", TimeToString(values[i].time, TIME_DATE|TIME_MINUTES), ")");
            
            // Store the first/closest news event for display
            if(nextNewsInfo == "") {
               string timeStr = "";
               if(minutesUntil < 60)
                  timeStr = IntegerToString(minutesUntil) + " min";
               else if(minutesUntil < 1440)
                  timeStr = IntegerToString(minutesUntil/60) + " hrs";
               else
                  timeStr = IntegerToString(minutesUntil/1440) + " days";
               
               nextNewsInfo = impactIcon + " " + country.currency + ": " + event.name + "\n   Impact: " + impactText + " | In " + timeStr;
            }
            
            if(values[i].time - TimeCurrent() < StopBeforeMin*PeriodSeconds(PERIOD_M1)){
               LastNewsAvoided = values[i].time; 
               TrDisableNews = true; 
               Print("⚠️ TRADING DISABLED: ", impactText, " impact news in ", minutesUntil, " minutes - ", event.name);
               
               string timeStr = "";
               if(minutesUntil < 60)
                  timeStr = IntegerToString(minutesUntil) + " min";
               else
                  timeStr = IntegerToString(minutesUntil/60) + " hrs";
               
               newsStatus = "⚠️ TRADING DISABLED\n   " + impactIcon + " " + country.currency + ": " + event.name + "\n   Impact: " + impactText + " | In " + timeStr;
               
               if(TradingEnabledComm=="" || TradingEnabledComm!="Printed"){
                  TradingEnabledComm = "Trading is disabled due to upcoming news: " + event.name; 
               }
               return true; 
            }
          // Don't return false here - continue checking other events
         }
       
   }
   
   // Update status based on what we found
   if(matchedEvents == 0) {
      newsStatus = "✅ No high-impact news in next " + IntegerToString(DaysNewsLookup) + " days";
   } else if(nextNewsInfo != "") {
      newsStatus = nextNewsInfo;
   } else {
      newsStatus = "📰 " + IntegerToString(matchedEvents) + " event(s) found";
   }
   
   return false;    
}
//+------------------------------------------------------------------+
// Get raw calendar events as JSON for inference server
string GetCalendarEventsJSON(string pair)
{
   // Extract currencies from pair (e.g., EUR_USD -> EUR, USD)
   string currencies[];
   StringSplit(pair, '_', currencies);
   
   if(ArraySize(currencies) != 2)
      return "[]";  // Empty array if invalid pair
   
   string curr1 = currencies[0];
   string curr2 = currencies[1];
   
   // Get events from last 48 hours (for tau_post) and next 48 hours (for tau_pre)
   MqlCalendarValue values[]; 
   datetime start_time = TimeCurrent() - PeriodSeconds(PERIOD_H1) * 48;  // 48 hours ago
   datetime end_time = TimeCurrent() + PeriodSeconds(PERIOD_H1) * 48;    // 48 hours ahead
   
   CalendarValueHistory(values, start_time, end_time, NULL, NULL); 
   
   if(ArraySize(values) == 0)
      return "[]";  // No events
   
   string jsonEvents = "[";
   int eventCount = 0;
   
   // Parse keywords for high-impact events
   string sep = (separator == 0) ? "," : ";";
   int k = StringSplit(keyNews, sep_code, newsToAvoid);
   
   for(int i = 0; i < ArraySize(values); i++)
   {
       MqlCalendarEvent event; 
       CalendarEventById(values[i].event_id, event); 
       MqlCalendarCountry country; 
       CalendarCountryById(event.country_id, country); 
       
       // Only include events for the pair's currencies
       if(country.currency != curr1 && country.currency != curr2)
          continue;
       
       // Only include high-impact events that match keywords
       bool matchesKeyword = false;
       for(int j = 0; j < k; j++)
       {
          if(StringFind(event.name, newsToAvoid[j]) >= 0)
          {
             matchesKeyword = true;
             break;
          }
       }
       
       if(!matchesKeyword)
          continue;
       
       // Get impact level
       string impactLevel = "low";
       switch(event.importance)
       {
          case CALENDAR_IMPORTANCE_HIGH:
             impactLevel = "high";
             break;
          case CALENDAR_IMPORTANCE_MODERATE:
             impactLevel = "medium";
             break;
          default:
             impactLevel = "low";
             break;
       }
       
       // Add comma between events
       if(eventCount > 0)
          jsonEvents += ",";
       
       // Build JSON object for this event
       jsonEvents += "{";
       jsonEvents += "\"timestamp\":\"" + TimeToString(values[i].time, TIME_DATE|TIME_MINUTES|TIME_SECONDS) + "\",";
       jsonEvents += "\"event_name\":\"" + event.name + "\",";
       jsonEvents += "\"country\":\"" + country.currency + "\",";
       jsonEvents += "\"actual\":" + DoubleToString(values[i].actual_value, 8) + ",";
       jsonEvents += "\"forecast\":" + DoubleToString(values[i].forecast_value, 8) + ",";
       jsonEvents += "\"previous\":" + DoubleToString(values[i].prev_value, 8) + ",";
       jsonEvents += "\"impact\":\"" + impactLevel + "\"";
       jsonEvents += "}";
       
       eventCount++;
   }
   
   jsonEvents += "]";
   
   if(eventCount > 0)
      Print("📊 Sending ", eventCount, " calendar events to server for ", pair);
   
   return jsonEvents;
}
//+------------------------------------------------------------------+
// Trailing stoploss

void TrailSL(){
   // Get broker requirements
   long stopLevel = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double minStopDistance = stopLevel * point;
   
   for(int i = PositionsTotal()-1; i>=0; i--){
      posinfo.SelectByIndex(i); 
      long magic = posinfo.Magic();
      ulong ticket = posinfo.Ticket(); 
      ENUM_POSITION_TYPE postype = posinfo.PositionType(); 
      string symbol = posinfo.Symbol(); 
      
      if(symbol == _Symbol && magic == inpMagic){
         double price = SymbolInfoDouble(_Symbol, SYMBOL_BID); 
         double currentSL = posinfo.StopLoss();
         double tp = posinfo.TakeProfit(); 
         double openPrice = posinfo.PriceOpen(); 
         double high = findHigh(); 
         double low  = findLow(); 
         double newSL = currentSL; // Initialize with current SL
         
         if (trailType==1){
            if(postype==POSITION_TYPE_BUY){
               if (price > posinfo.PriceOpen() && low>0){
                  newSL = low - HighLowBuffer*10*_Point; 
               }
            } else if(postype==POSITION_TYPE_SELL){
               if(price < posinfo.PriceOpen() && high>0){
                  newSL = high + HighLowBuffer*10*_Point; 
               } 
            }
         }
         if(trailType==2){
            if(postype == POSITION_TYPE_BUY){
               if(price > posinfo.PriceOpen()){
                  newSL = price - trailFixedpips*10*_Point; 
               }   
            } else if(postype == POSITION_TYPE_SELL){
               if(price < posinfo.PriceOpen()){
                  newSL = price + trailFixedpips*10*_Point; 
               }
            }
         }
         if(trailType == 3){
            if(postype==POSITION_TYPE_BUY){
               if (price > posinfo.PriceOpen()){
                  double tpCurrent = posinfo.TakeProfit();
                  double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID); 
                  if(tpCurrent > 0) {
                     double totalTpDistance = tpCurrent - posinfo.PriceOpen();
                     double currentProfit = bid - posinfo.PriceOpen();
                     double profitPercentage = (currentProfit / totalTpDistance) * 100;
                     double trailAmount = bid * (TslPercent/100);
                     if(profitPercentage >= TslPercentTP){
                        newSL = bid - trailAmount; 
                     }
                  }
               }
            } else if(postype==POSITION_TYPE_SELL){
               if(price < posinfo.PriceOpen()){
                  double tpCurrent = posinfo.TakeProfit();
                  double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK); 
                  if(tpCurrent > 0) {
                     double totalTpDistance = posinfo.PriceOpen() - tpCurrent;
                     double currentProfit = posinfo.PriceOpen() - ask;
                     double profitPercentage = (currentProfit / totalTpDistance) * 100;
                     double trailAmount = ask * (TslPercent/100);
                     if(profitPercentage >= TslPercentTP){
                        newSL = ask + trailAmount;
                     }
                  }
               }
            }
         }
         
         // Normalize and validate the new stop loss
         newSL = NormalizeDouble(newSL, _Digits);
         
         // Check if we need to modify the position
         bool shouldModify = false;
         if(postype == POSITION_TYPE_BUY && newSL > currentSL && newSL != currentSL) {
            // For buy positions, only move SL up (trailing)
            if(newSL <= price - minStopDistance) { // Ensure minimum distance from current price
               shouldModify = true;
            }
         } else if(postype == POSITION_TYPE_SELL && newSL < currentSL && newSL != currentSL) {
            // For sell positions, only move SL down (trailing)  
            if(newSL >= price + minStopDistance) { // Ensure minimum distance from current price
               shouldModify = true;
            }
         }
         
         // Only modify if conditions are met and values are different
         if(shouldModify && MathAbs(newSL - currentSL) > point) {
            if(trade.PositionModify(ticket, newSL, tp)) {
               printf("[TSL]: Position #%d modified successfully. Old SL: %.5f, New SL: %.5f", ticket, currentSL, newSL);
            } else {
               printf("[TSL]: Failed to modify position #%d. Error: %d", ticket, GetLastError());
            }
         }
      } // Closes if(symbol == _Symbol && magic == Magic)
   } // Closes for loop
} // Closes function
//+------------------------------------------------------------------+
//| Function to Reset positions                                      |
//+------------------------------------------------------------------+
void CloseandResetAll(){
   
   printf("Starting CloseandResetAll at %s", TimeToString(TimeCurrent()));
   
   // Close all positions for this EA with retry mechanism
   for(int i = PositionsTotal()-1; i>=0; i--){
      if(posinfo.SelectByIndex(i)) {
         if(posinfo.Symbol() == _Symbol && posinfo.Magic() == inpMagic){
            ulong ticket = posinfo.Ticket();
            
            // Try to close position with retries
            int attempts = 0;
            bool closed = false;
            while(attempts < 3 && !closed) {
               attempts++;
               
               // Refresh quotes before closing
               MqlTick tick;
               SymbolInfoTick(_Symbol, tick);
               Sleep(100); // Small delay
               
               if(trade.PositionClose(ticket)) {
                  printf("Position #%d closed successfully on attempt %d", ticket, attempts);
                  closed = true;
               } else {
                  int error = GetLastError();
                  printf("Failed to close position #%d on attempt %d, Error: %d", ticket, attempts, error);
                  
                  // If it's a "close to market" error, wait longer
                  if(error == 10015 || error == 10016 || error == 10017) {
                     Sleep(1000); // Wait 1 second
                  }
               }
            }
            
            if(!closed) {
               printf("Position #%d could not be closed after 3 attempts - will try again later", ticket);
            }
         }
      }
   }
   
   // Delete all pending orders for this EA
   for (int i = OrdersTotal()-1; i>=0; i--){
      if(orderinfo.SelectByIndex(i)) {
         if(orderinfo.Symbol() == _Symbol && orderinfo.Magic() == inpMagic){
            ulong ticket = orderinfo.Ticket();
            if(!trade.OrderDelete(ticket)) {
               printf("Failed to delete order #%d, Error: %d", ticket, GetLastError());
            } else {
               printf("Order #%d deleted successfully", ticket);
            }
         }
      }
   }
   
   barsRangeStart = 0; 
   buyTotal = 0;
   sellTotal = 0;
   
   
   printf("EA Reset completed at %s", TimeToString(TimeCurrent()));
   ChartSetInteger(0, CHART_COLOR_BACKGROUND, clrBlack); 
} 
//+------------------------------------------------------------------+
double calcLots(double slPoints){
   
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double minvolume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN); 
   
   // For very small accounts, return minimum lot immediately
   if(balance < 10) {
      printf("Very small account (%.2f) - using minimum lot size: %.2f", balance, minvolume);
      return minvolume;
   }
   
   double risk = balance * UserRiskPercent/100; 
   
   double ticksize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE); 
   double tickvalue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double lotstep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP); 
   double maxvolume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX); 
   double volumelimit = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_LIMIT); 
   
   if (slPoints < 0) slPoints = slPoints*-1; 
   if (slPoints == 0) slPoints = 0.001; // Prevent division by zero
   
   double moneyPerLotStep = slPoints / ticksize * tickvalue * lotstep;
   double lots = 0;
   
   if(moneyPerLotStep > 0) {
      lots = MathFloor(risk / moneyPerLotStep) * lotstep;
   } else {
      lots = minvolume; // Fallback to minimum
   }
   
   // Ensure we don't exceed what the account can afford
   double marginRequired = lots * SymbolInfoDouble(_Symbol, SYMBOL_MARGIN_INITIAL);
   if(marginRequired > balance * 0.5) { // Never use more than 50% of balance for margin
      lots = (balance * 0.5) / SymbolInfoDouble(_Symbol, SYMBOL_MARGIN_INITIAL);
      lots = MathFloor(lots / lotstep) * lotstep;
      
      // If still too small, use minimum
      if(lots < minvolume) {
         lots = minvolume;
      }
   }
   
   if (volumelimit!=0) lots = MathMin(lots, volumelimit); 
   if (maxvolume!=0) lots = MathMin(lots, maxvolume); 
   if (minvolume!=0) lots = MathMax(lots, minvolume); 
   lots = NormalizeDouble(lots, 2); 
   
   printf("Lot calculation: Balance=%.2f, Risk=%.2f, SL=%.5f, Calculated=%.2f, Margin=%.2f", 
          balance, risk, slPoints, lots, lots * SymbolInfoDouble(_Symbol, SYMBOL_MARGIN_INITIAL));
   
   return lots; 
} 
//+------------------------------------------------------------------+
//| Function to Check if Market is Open                              |
//+------------------------------------------------------------------+
bool IsMarketOpen(){
   // Check if we have recent tick data (within last 60 seconds)
   datetime lastTick = (datetime)SymbolInfoInteger(_Symbol, SYMBOL_TIME);
   if(TimeCurrent() - lastTick > 60) {
      Print("⚠️ Market check: No recent ticks (last tick: ", (TimeCurrent() - lastTick), " seconds ago)");
      return false;
   }
   
   // Check if spread is reasonable (not too wide indicating market closure)
   // Use dynamic spread check based on current price
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double spread = ask - bid;
   double spreadPercent = (spread / bid) * 100.0;
   
   // If spread is more than 1% of price, market might be closed or illiquid
   // This works for both forex (tiny spreads) and crypto (larger spreads)
   if(spreadPercent > 1.0) {
      Print("⚠️ Market check: Spread too wide (", DoubleToString(spreadPercent, 2), "% of price)");
      return false;
   }
   
   return true;
}
