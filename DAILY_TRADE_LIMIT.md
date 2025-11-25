# Daily Trade Limit System - 1 BUY + 1 SELL Per Day

## Overview

The EA now enforces a strict daily trade limit:
- **Maximum 1 BUY trade per day**
- **Maximum 1 SELL trade per day**
- **Counters reset automatically at midnight (start of new trading day)**

This prevents overtrading and ensures disciplined position management.

## How It Works

### 1. Daily Counter Tracking
Three new variables track daily trades:
```mql5
int buyTradesToday = 0;      // BUY trades opened today
int sellTradesToday = 0;     // SELL trades opened today
int lastTradeDay = 0;        // Last day tracked (for daily reset)
```

### 2. Automatic Daily Reset
The `CheckForOpenOrdersandPositions()` function now:
- Checks if a new trading day has started
- Resets `buyTradesToday` and `sellTradesToday` to 0 at midnight
- Logs the reset event for transparency

```mql5
MqlDateTime dt;
TimeToStruct(TimeCurrent(), dt);
int currentDay = dt.day;

if(currentDay != lastTradeDay)
{
   // New day - reset counters
   buyTradesToday = 0;
   sellTradesToday = 0;
   lastTradeDay = currentDay;
   
   Print("📅 NEW TRADING DAY: ", TimeToString(TimeCurrent(), TIME_DATE), " - Trade counters reset");
}
```

### 3. Pre-Trade Validation
Before opening any trade, the EA checks:
```mql5
if(!CanTradeToday(prediction))
{
   Print("⚠️ TRADE BLOCKED: Daily limit reached");
   return;
}
```

The `CanTradeToday()` function:
- Returns `false` if BUY limit reached (when trying to open BUY)
- Returns `false` if SELL limit reached (when trying to open SELL)
- Logs the blocking reason

### 4. Post-Trade Counter Increment
After successful order execution:
```mql5
if(OrderSend(request, result))
{
   Print("SUCCESS: Order placed. Ticket: ", result.order);
   
   // Increment daily trade counter
   IncrementDailyTradeCounter(prediction);
}
```

## Trading Scenarios

### Scenario 1: Normal Trading Day
```
00:00 - Counters reset (BUY: 0/1, SELL: 0/1)
08:00 - BUY signal → Trade opened (BUY: 1/1, SELL: 0/1)
10:00 - BUY signal → BLOCKED (already 1 BUY today)
14:00 - SELL signal → Trade opened (BUY: 1/1, SELL: 1/1)
16:00 - SELL signal → BLOCKED (already 1 SELL today)
18:00 - BUY signal → BLOCKED (already 1 BUY today)
```

### Scenario 2: One Direction Only
```
00:00 - Counters reset (BUY: 0/1, SELL: 0/1)
09:00 - BUY signal → Trade opened (BUY: 1/1, SELL: 0/1)
11:00 - BUY signal → BLOCKED (already 1 BUY today)
13:00 - BUY signal → BLOCKED (already 1 BUY today)
15:00 - HOLD signal → No trade
17:00 - BUY signal → BLOCKED (already 1 BUY today)
```

### Scenario 3: Opposite Direction After Loss
```
00:00 - Counters reset (BUY: 0/1, SELL: 0/1)
08:00 - BUY signal → Trade opened (BUY: 1/1, SELL: 0/1)
10:00 - BUY position hits SL (loss)
10:01 - BUY trades now blocked by loss counter
12:00 - SELL signal → Trade opened (BUY: 1/1, SELL: 1/1)
14:00 - SELL signal → BLOCKED (already 1 SELL today)
```

## Chart Display

The chart now shows daily trade limits in real-time:
```
POSITION MANAGEMENT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Positions: 1 / 2 (max)
📅 Today's Trades:
   🟢 BUY:  1 / 1
   🔴 SELL: 0 / 1
   (Resets daily at midnight)
```

## Log Messages

### Daily Reset
```
📅 NEW TRADING DAY: 2025.11.24 - Trade counters reset
```

### Trade Opened
```
✅ BUY trade opened - Daily count: 1/1
✅ SELL trade opened - Daily count: 1/1
```

### Trade Blocked
```
⚠️ DAILY LIMIT: Already opened 1 BUY trade(s) today - blocking new BUY
⚠️ TRADE BLOCKED: Daily limit reached for BUY trades
   Limit: 1 BUY + 1 SELL per day | Resets at midnight
```

## Integration with Existing Systems

The daily limit system works **in conjunction** with existing protections:

### 1. Loss Counter System
- If a BUY trade closes at loss → BUY trades blocked until reset
- Daily limit still applies even after loss counter reset
- Example: Reset loss counter at 2pm, but already opened 1 BUY at 9am → still blocked

### 2. Duplicate Position Prevention
- Daily limit checked BEFORE duplicate check
- Prevents opening multiple positions of same type
- Example: Can't open 2 BUY positions even if daily limit allows

### 3. Maximum Positions Per Chart
- Daily limit checked BEFORE max positions check
- Chart limit: 2 positions total (any combination)
- Daily limit: 1 BUY + 1 SELL max
- Example: Can have 1 BUY + 1 SELL = 2 positions (both limits satisfied)

### 4. Symbol Whitelist
- Daily limit checked AFTER symbol validation
- Only counts trades on allowed pairs
- Example: EURUSD BUY doesn't count toward GBPUSD daily limit

## Execution Order (Trade Validation)

When a signal is generated, checks happen in this order:

1. ✅ **Symbol Allowed?** → Block if not in trained pairs
2. ✅ **Max Positions?** → Block if >= 2 positions on chart
3. ✅ **Daily Limit?** → Block if already opened 1 trade in this direction today ⭐ NEW
4. ✅ **Loss Counter?** → Block if previous trade in this direction closed at loss
5. ✅ **Duplicate Position?** → Block if position of same type already open
6. ✅ **Good Trading Conditions?** → Block if news filter active, etc.
7. ✅ **Execute Trade** → Open position and increment daily counter

## Benefits

1. **Prevents Overtrading**: Maximum 2 trades per day (1 BUY + 1 SELL)
2. **Disciplined Approach**: Forces selective trade entry
3. **Risk Management**: Limits daily exposure
4. **Automatic Reset**: No manual intervention needed
5. **Clear Visibility**: Chart shows remaining daily trades
6. **Works with Existing Logic**: Integrates seamlessly with loss counters, duplicate prevention, etc.

## Testing Recommendations

1. **Enable Debug Logging**:
   ```mql5
   ShowDebugInfo = true
   ```

2. **Test Daily Reset**:
   - Run EA overnight
   - Check logs at midnight for reset message
   - Verify counters show 0/1 after reset

3. **Test Trade Blocking**:
   - Open 1 BUY trade manually or via signal
   - Wait for another BUY signal
   - Verify it's blocked with daily limit message

4. **Test Both Directions**:
   - Open 1 BUY trade
   - Open 1 SELL trade
   - Verify both are blocked after that

5. **Test with Loss Counter**:
   - Open BUY trade, close at loss
   - Verify BUY blocked by loss counter
   - Verify SELL still allowed (if daily limit not reached)

## Configuration

No new input parameters needed. The system works automatically with:
- Existing `ShowDebugInfo` for logging
- Existing `inpMagic` for position filtering
- Existing `_Symbol` for symbol filtering

## Troubleshooting

### Issue: Counters not resetting
**Solution**: Check server time. Reset happens at midnight server time, not local time.

### Issue: Trade blocked but counter shows 0/1
**Solution**: Check if loss counter is blocking the trade instead. Loss counter takes precedence.

### Issue: Can't open any trades
**Solution**: Check all blocking conditions:
1. Daily limit (this system)
2. Loss counter
3. Max positions per chart
4. Symbol whitelist
5. News filter

### Issue: Counter increments but no position opened
**Solution**: Check OrderSend result. Counter only increments on successful order execution.

## Files Modified

1. **MQL5/core_functions.mqh**:
   - Added daily counter variables
   - Modified `CheckForOpenOrdersandPositions()` for daily reset
   - Added `CanTradeToday()` function
   - Added `IncrementDailyTradeCounter()` function

2. **MQL5/Auron AI.mq5**:
   - Added daily limit check in `ExecuteTrade()`
   - Added counter increment after successful order
   - Added daily limit display in chart comment

## Summary

The EA now enforces a strict **1 BUY + 1 SELL per day** limit that:
- ✅ Resets automatically at midnight
- ✅ Blocks duplicate trades in same direction
- ✅ Works with existing loss counter system
- ✅ Shows real-time status on chart
- ✅ Provides clear logging for debugging
- ✅ Requires no manual intervention
