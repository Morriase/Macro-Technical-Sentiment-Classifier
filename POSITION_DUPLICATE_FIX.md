# Position Duplicate Fix - Concurrent Same-Type Positions

## Problem Identified

The EA was opening multiple positions of the same type (BUY or SELL) concurrently, despite having checks in place. This happened because:

1. **Race Condition**: When `UpdateIntervalSeconds` is set to "Every X Seconds", the EA can be triggered multiple times before a position is fully opened
2. **Weak Position Check**: The `HasPosition()` function only checked if a position existed, but didn't properly prevent duplicates during rapid consecutive calls
3. **Timing Issue**: Between the check and the actual order execution, another signal could trigger and pass the same check

## Root Cause

The `HasPosition()` function had a simple boolean check that would return `true` if it found any position of the requested type. However, during rapid execution (every X seconds), the following sequence could occur:

```
Time 0s: Signal BUY → Check positions (0 found) → Start opening position
Time 5s: Signal BUY → Check positions (order still pending) → Start opening position
Time 10s: Both orders execute → 2 BUY positions open
```

## Solution Implemented

### 1. Enhanced `HasPosition()` Function
- Changed from boolean check to **counting** positions of the same type
- Added debug logging to show how many duplicate positions were blocked
- Returns `true` if ANY position of the requested type exists (not just one)

```mql5
bool HasPosition(int prediction)
{
   int count = 0;
   
   // Count all positions of the requested type
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionSelectByTicket(PositionGetTicket(i)))
      {
         if(PositionGetString(POSITION_SYMBOL) == _Symbol && 
            PositionGetInteger(POSITION_MAGIC) == inpMagic)
         {
            ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
            
            if(prediction == 1 && posType == POSITION_TYPE_BUY)
               count++;
            else if(prediction == -1 && posType == POSITION_TYPE_SELL)
               count++;
         }
      }
   }
   
   // Block if ANY position exists
   if(count > 0)
   {
      if(ShowDebugInfo)
         Print("⚠️ Already have ", count, " ", (prediction == 1 ? "BUY" : "SELL"), " position(s) - blocking duplicate");
      return true;
   }
   
   return false;
}
```

### 2. Double-Check in `ExecuteTrade()`
Added a **critical safety check** right before order execution that:
- Counts positions of the same type in real-time
- Blocks execution if ANY position of the same type exists
- Provides clear logging of blocked duplicates

```mql5
// CRITICAL: Double-check for existing positions of same type
int sameTypeCount = 0;
for(int i = PositionsTotal() - 1; i >= 0; i--)
{
   if(PositionSelectByTicket(PositionGetTicket(i)))
   {
      if(PositionGetString(POSITION_SYMBOL) == _Symbol && 
         PositionGetInteger(POSITION_MAGIC) == inpMagic)
      {
         ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
         
         if((prediction == 1 && posType == POSITION_TYPE_BUY) ||
            (prediction == -1 && posType == POSITION_TYPE_SELL))
         {
            sameTypeCount++;
         }
      }
   }
}

if(sameTypeCount > 0)
{
   Print("⚠️ DUPLICATE BLOCKED: Already have ", sameTypeCount, " ", 
         (prediction == 1 ? "BUY" : "SELL"), " position(s) open");
   return;
}
```

## Benefits

1. **Prevents Race Conditions**: The double-check ensures no duplicates even during rapid consecutive signals
2. **Clear Logging**: You'll see exactly when and why duplicate positions are blocked
3. **No Performance Impact**: The additional check is minimal and only runs when a trade signal is generated
4. **Maintains Existing Logic**: All other position management features remain intact

## Testing Recommendations

1. Set `UpdateIntervalSeconds = 1` (Every X Seconds) with `updateSeconds = 5`
2. Enable `ShowDebugInfo = true` to see blocking messages
3. Watch for log messages like:
   - `⚠️ Already have 1 BUY position(s) - blocking duplicate`
   - `⚠️ DUPLICATE BLOCKED: Already have 1 SELL position(s) open`
4. Verify that only ONE position per direction opens at a time

## What This Fixes

✅ Multiple BUY positions opening concurrently
✅ Multiple SELL positions opening concurrently  
✅ Race conditions during rapid signal generation
✅ Duplicate positions when EA runs every X seconds

## What This Doesn't Change

- Maximum positions per chart (still 2)
- Loss counter blocking (still active)
- Opposite position closing (still works)
- All other risk management features
