# Trading Session Filter - Time-Based Trade Control

## Overview

The EA now includes a comprehensive trading session filter that allows you to trade only during specific market sessions. This is based on your observation that **Asian session and post-London session** perform better.

## Features

✅ **4 Pre-configured Sessions**: Asian, London, New York, Custom
✅ **Flexible Time Ranges**: Set start/end hours for each session
✅ **Multiple Session Support**: Enable any combination of sessions
✅ **Midnight Wrap Support**: Sessions can span across midnight (e.g., 22:00-02:00)
✅ **Real-time Display**: Chart shows current session and trading status
✅ **Server Time Based**: Uses broker server time for consistency

## Input Parameters

### Main Control
- **EnableSessionFilter** (bool): Master switch to enable/disable session filtering
  - `false` = Trade 24/7 (no restrictions)
  - `true` = Only trade during enabled sessions

### Asian Session (Tokyo)
- **TradeAsianSession** (bool): Enable trading during Asian session
- **AsianStartHour** (TimeHour enum): Start time - dropdown with 12-hour format
  - Default: `12:00 AM (Midnight)`
- **AsianEndHour** (TimeHour enum): End time - dropdown with 12-hour format
  - Default: `9:00 AM`
- **Default Range**: 12:00 AM - 9:00 AM server time

### London Session
- **TradeLondonSession** (bool): Enable trading during London session
- **LondonStartHour** (TimeHour enum): Start time - dropdown with 12-hour format
  - Default: `8:00 AM`
- **LondonEndHour** (TimeHour enum): End time - dropdown with 12-hour format
  - Default: `5:00 PM`
- **Default Range**: 8:00 AM - 5:00 PM server time

### New York Session
- **TradeNewYorkSession** (bool): Enable trading during New York session
- **NewYorkStartHour** (TimeHour enum): Start time - dropdown with 12-hour format
  - Default: `1:00 PM`
- **NewYorkEndHour** (TimeHour enum): End time - dropdown with 12-hour format
  - Default: `10:00 PM`
- **Default Range**: 1:00 PM - 10:00 PM server time

### Custom Session
- **TradeCustomSession** (bool): Enable trading during custom time range
- **CustomStartHour** (TimeHour enum): Custom start time - dropdown with 12-hour format
  - Default: `12:00 AM (Midnight)`
- **CustomEndHour** (TimeHour enum): Custom end time - dropdown with 12-hour format
  - Default: `11:00 PM`
- **Use Case**: Define your own optimal trading hours

### TimeHour Enum Options
All time inputs use a dropdown with 24 options in 12-hour format:
```
12:00 AM (Midnight), 1:00 AM, 2:00 AM, 3:00 AM, 4:00 AM, 5:00 AM,
6:00 AM, 7:00 AM, 8:00 AM, 9:00 AM, 10:00 AM, 11:00 AM,
12:00 PM (Noon), 1:00 PM, 2:00 PM, 3:00 PM, 4:00 PM, 5:00 PM,
6:00 PM, 7:00 PM, 8:00 PM, 9:00 PM, 10:00 PM, 11:00 PM
```

## Recommended Settings

### For Your Strategy (Asian + Post-London)

Based on your observation of better performance:

```
EnableSessionFilter = true

// Asian Session (00:00-09:00)
TradeAsianSession = true
AsianStartHour = 0
AsianEndHour = 9

// London Session - DISABLED (you prefer post-London)
TradeLondonSession = false

// Post-London / Early NY Session (17:00-22:00)
TradeNewYorkSession = false  // Disable default NY
TradeCustomSession = true
CustomStartHour = 17  // 5:00 PM (post-London)
CustomEndHour = 22    // 10:00 PM

// Result: Trades only 00:00-09:00 and 17:00-22:00
```

### Alternative: Asian + London Overlap

For maximum volatility during London open:

```
EnableSessionFilter = true

TradeAsianSession = true
AsianStartHour = 7
AsianEndHour = 9

TradeLondonSession = true
LondonStartHour = 8
LondonEndHour = 12

// Result: Trades 07:00-12:00 (Asian close + London open)
```

### Alternative: New York Session Only

For US market hours:

```
EnableSessionFilter = true

TradeAsianSession = false
TradeLondonSession = false

TradeNewYorkSession = true
NewYorkStartHour = 13
NewYorkEndHour = 22

// Result: Trades only 13:00-22:00
```

## How It Works

### 1. Session Check Before Trade
Every time a signal is generated, the EA checks:
```
Signal Generated
    ↓
Symbol Allowed? → NO → Block
    ↓ YES
Max Positions? → YES → Block
    ↓ NO
Within Trading Session? → NO → Block ⭐ NEW
    ↓ YES
Daily Limit? → YES → Block
    ↓ NO
Execute Trade
```

### 2. Midnight Wrap Support
Sessions can span across midnight:
```
Example: Asian Session 22:00-02:00
- 21:00 → Outside session ❌
- 22:00 → Inside session ✅
- 23:00 → Inside session ✅
- 00:00 → Inside session ✅
- 01:00 → Inside session ✅
- 02:00 → Outside session ❌
```

### 3. Multiple Session Support
You can enable multiple sessions:
```
Asian: 00:00-09:00 ✅
London: 08:00-17:00 ✅
NY: 13:00-22:00 ❌

Trading allowed:
- 00:00-09:00 (Asian)
- 08:00-17:00 (London, overlaps with Asian 08:00-09:00)
- Total: 00:00-17:00
```

## Chart Display

The chart shows current session status:

### When Inside Trading Hours
```
STATUS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ 🟢 READY TO BUY
🎯 Trading: ENABLED
📈 Min Confidence: 55%

🕐 Session: Asian ✅
   Enabled: Asian London NY 

📅 Today's Trades:
   🟢 BUY:  0 / 1
   🔴 SELL: 0 / 1
```

### When Outside Trading Hours
```
STATUS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ 🟢 READY TO BUY
🎯 Trading: ENABLED
📈 Min Confidence: 55%

🕐 Session: Outside Trading Hours ⛔
   Enabled: Asian Custom 

📅 Today's Trades:
   🟢 BUY:  1 / 1
   🔴 SELL: 0 / 1
```

### When Filter Disabled
```
🕐 Session Filter: Disabled (24/7)
```

## Log Messages

### Trade Allowed
```
✅ ASIAN SESSION: Trading allowed (08:00 is within 0:00-9:00)
```

### Trade Blocked
```
⚠️ SESSION FILTER: Trading blocked (10:00 is outside all enabled sessions)
⚠️ TRADE BLOCKED: Outside allowed trading sessions
   Current time: 10:35
   Active sessions: Outside Trading Hours
```

## Server Time vs Local Time

**IMPORTANT**: All times are in **broker server time**, not your local time.

### Finding Your Broker's Server Time
1. Check MT5 "Market Watch" window
2. Look at the time displayed
3. Compare with your local time
4. Calculate the offset

### Example: Broker is GMT+2, You are GMT-5
```
Your Local Time: 03:00 (3 AM)
Broker Server Time: 10:00 (10 AM)
Offset: +7 hours

If you want to trade 8 AM - 5 PM your local time:
- Your 8 AM = Broker 3 PM (15:00)
- Your 5 PM = Broker 12 AM (00:00 next day)

Settings:
CustomStartHour = 15
CustomEndHour = 0  // Wraps to next day
```

## Session Timing Reference (GMT/UTC)

### Major Forex Sessions
```
Sydney:    22:00 - 07:00 GMT
Tokyo:     00:00 - 09:00 GMT (Asian)
London:    08:00 - 17:00 GMT
New York:  13:00 - 22:00 GMT
```

### Session Overlaps (High Volatility)
```
Tokyo + London:    08:00 - 09:00 GMT
London + NY:       13:00 - 17:00 GMT
```

### Low Volatility Periods
```
After NY Close:    22:00 - 00:00 GMT
Before Tokyo:      07:00 - 08:00 GMT (Sunday gap)
```

## Use Cases

### 1. Asian Session Scalper
```
EnableSessionFilter = true
TradeAsianSession = true
AsianStartHour = 0
AsianEndHour = 9
// All others = false
```

### 2. London Breakout Trader
```
EnableSessionFilter = true
TradeLondonSession = true
LondonStartHour = 8
LondonEndHour = 12
// All others = false
```

### 3. NY Session Momentum
```
EnableSessionFilter = true
TradeNewYorkSession = true
NewYorkStartHour = 14  // After lunch
NewYorkEndHour = 20    // Before close
// All others = false
```

### 4. Avoid Low Volatility (Your Strategy)
```
EnableSessionFilter = true

// Trade Asian
TradeAsianSession = true
AsianStartHour = 0
AsianEndHour = 9

// Skip London (you observed poor performance)
TradeLondonSession = false

// Trade post-London
TradeCustomSession = true
CustomStartHour = 17
CustomEndHour = 22

// Result: Avoids 09:00-17:00 (low performance period)
```

### 5. Weekend Gap Avoidance
```
EnableSessionFilter = true

// Avoid Sunday open (22:00 Sunday)
TradeAsianSession = true
AsianStartHour = 1   // Start 1 hour after open
AsianEndHour = 9

// Avoid Friday close (22:00 Friday)
TradeNewYorkSession = true
NewYorkStartHour = 13
NewYorkEndHour = 21  // End 1 hour before close
```

## Integration with Other Filters

The session filter works **in conjunction** with:

1. **News Filter**: Both must allow trading
   - Session OK + News OK = Trade ✅
   - Session OK + News Block = No Trade ❌
   - Session Block + News OK = No Trade ❌

2. **Daily Limit**: Session checked before daily limit
   - Outside session = Blocked (doesn't count toward daily limit)
   - Inside session + daily limit reached = Blocked

3. **Loss Counter**: Session checked before loss counter
   - Outside session = Blocked (loss counter still active)
   - Inside session + loss counter active = Blocked

## Performance Optimization

### Your Observation: Asian + Post-London Best

**Hypothesis**: These periods have:
- ✅ Lower spread (Asian session)
- ✅ Clearer trends (post-London consolidation)
- ✅ Less whipsaw (avoiding London volatility)
- ✅ Better signal quality

**Recommended Test**:
```
Week 1: Trade all sessions (baseline)
Week 2: Trade Asian only (00:00-09:00)
Week 3: Trade post-London only (17:00-22:00)
Week 4: Trade Asian + post-London (your strategy)

Compare:
- Win rate
- Average profit per trade
- Drawdown
- Number of trades
```

## Troubleshooting

### Issue: No trades opening
**Check**:
1. Is `EnableSessionFilter = true`?
2. Is current hour within any enabled session?
3. Check broker server time vs your expectations
4. Enable `ShowDebugInfo = true` to see session checks

### Issue: Trading at wrong times
**Solution**: Verify broker server time offset
```
Print("Server Time: ", TimeToString(TimeCurrent(), TIME_MINUTES));
Print("Your Local Time: [check your clock]");
// Calculate offset and adjust session hours
```

### Issue: Session wraps midnight incorrectly
**Example**: Want 22:00-02:00
```
CustomStartHour = 22
CustomEndHour = 2
// EA automatically handles midnight wrap
```

### Issue: Want to trade 24/7 temporarily
**Solution**: 
```
EnableSessionFilter = false
// Disables all session checks
```

## Testing Recommendations

1. **Enable Debug Logging**:
   ```
   ShowDebugInfo = true
   ```

2. **Test Session Boundaries**:
   - Set session to current hour ± 1
   - Verify trades allowed inside, blocked outside

3. **Test Midnight Wrap**:
   - Set session like 23:00-01:00
   - Wait for midnight
   - Verify trading continues

4. **Test Multiple Sessions**:
   - Enable 2+ sessions
   - Verify trades allowed in any enabled session

5. **Test with Other Filters**:
   - Enable news filter + session filter
   - Verify both must allow trading

## Files Modified

1. **MQL5/Auron AI.mq5**:
   - Added session filter input parameters
   - Added session check in `ExecuteTrade()`
   - Added session display in chart comment

2. **MQL5/core_functions.mqh**:
   - Added `IsWithinTradingSession()` function
   - Added `GetCurrentSessionName()` function

## Summary

The session filter gives you precise control over **when** the EA trades, allowing you to:
- ✅ Trade only during high-performance sessions (Asian + post-London)
- ✅ Avoid low-volatility or high-spread periods
- ✅ Customize for your broker's server time
- ✅ Test different session strategies
- ✅ Combine multiple sessions for optimal coverage
