# EA Position Management Updates

## Overview
The EA has been updated with enhanced position management rules to prevent overtrading and ensure it only trades on pairs the models were trained on.

## New Features

### 1. **Allowed Trading Pairs Only**
The EA will **only trade** on the following currency pairs (the ones the models were trained on):
- **EURUSD** (EUR_USD)
- **GBPUSD** (GBP_USD)
- **AUDUSD** (AUD_USD)
- **USDJPY** (USD_JPY)

If you attach the EA to any other pair (e.g., EURGBP, NZDUSD, etc.), it will:
- Display a warning on startup
- Block all trading signals
- Show "Symbol: NOT ALLOWED" in the chart display

The symbol check handles common broker suffixes (.ecn, .raw, .pro, etc.) automatically.

### 2. **Maximum 2 Positions Per Chart**
The EA will maintain a maximum of **2 open positions** per chart at any time.

This prevents:
- Overexposure on a single pair
- Excessive risk concentration
- Account drawdown from multiple losing trades on the same symbol

### 3. **Loss Prevention System**
**Key Rule:** If a position closes at a loss, the EA will **NOT open another position in the same direction** until:
- A winning trade occurs in that direction (resets the counter), OR
- You manually reset the loss counters

**How it works:**
- **BUY position closes at loss** → BUY trades BLOCKED, SELL trades still allowed
- **SELL position closes at loss** → SELL trades BLOCKED, BUY trades still allowed
- **Winning trade** → Automatically resets the loss counter for that direction

**Example:**
1. EA opens BUY position → closes at -$50 loss
2. BUY trades now BLOCKED (buyLossCount = 1)
3. EA can still open SELL positions
4. If SELL position wins → SELL counter stays at 0
5. If EA gets another BUY signal → BLOCKED (won't open)
6. To re-enable BUY: Either wait for a winning BUY trade, or manually reset

### 4. **Manual Reset Option**
New input parameter: **ResetLossCounters**

To manually reset the loss counters:
1. Open EA settings
2. Go to "Position Management" section
3. Toggle **ResetLossCounters** to `true`
4. Click OK
5. Both BUY and SELL loss counters will be reset to 0
6. All directions will be re-enabled

**Note:** After resetting, you can toggle it back to `false` for next time.

## Chart Display Updates

The EA now shows real-time position management status:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POSITION MANAGEMENT:

📊 Positions: 1 / 2 (max)
✅ Symbol: ALLOWED
🟢 BUY: ENABLED
🔴 SELL: BLOCKED (loss count: 1)
   Last loss: 2025.11.21 14:35
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Status Indicators:
- **🟢 ENABLED** - Direction is allowed to trade
- **🔴 BLOCKED** - Direction blocked due to previous loss
- **✅ Symbol: ALLOWED** - Current pair is in trained pairs list
- **⛔ Symbol: NOT ALLOWED** - Current pair is NOT in trained pairs list

## Benefits

1. **Prevents Revenge Trading**: No more opening multiple losing trades in the same direction
2. **Risk Control**: Maximum 2 positions per chart limits exposure
3. **Model Accuracy**: Only trades on pairs the models were trained on (prevents poor predictions on untrained pairs)
4. **Transparency**: Clear visual feedback on what's blocked and why
5. **Flexibility**: Manual reset option when you want to override the system

## Technical Implementation

### New Global Variables:
```mql5
datetime lastBuyLossTime = 0;
datetime lastSellLossTime = 0;
int buyLossCount = 0;
int sellLossCount = 0;
string allowedPairs[] = {"EURUSD", "GBPUSD", "AUDUSD", "USDJPY", ...};
#define MAX_POSITIONS_PER_CHART 2
```

### New Functions:
- `IsSymbolAllowed()` - Checks if current symbol is in trained pairs list
- `CountPositionsOnChart()` - Counts open positions for current symbol
- `OnTradeTransaction()` - Tracks position outcomes and updates loss counters

### Trade Execution Flow:
```
1. Signal received from server
2. Check: Is symbol allowed? → NO: Block trade
3. Check: Max positions reached? → YES: Block trade
4. Check: Direction has loss count > 0? → YES: Block trade
5. Check: Already have position in this direction? → YES: Block trade
6. Check: Good trading conditions? → NO: Block trade
7. All checks passed → Execute trade
```

## Usage Recommendations

1. **Multi-Chart Setup**: Attach EA to EURUSD, GBPUSD, AUDUSD, and USDJPY charts
2. **Monitor Loss Counters**: Check the chart display regularly to see which directions are blocked
3. **Reset Strategically**: Only reset loss counters when you're confident conditions have changed
4. **Respect the Blocks**: The system is designed to protect your account - don't override it too frequently

## Warnings

⚠️ **Do NOT attach this EA to untrained pairs** (e.g., EURGBP, NZDUSD, XAUUSD, etc.)
- The models were not trained on these pairs
- Predictions will be unreliable
- The EA will block trading automatically

⚠️ **Loss counters persist until reset**
- If BUY is blocked, it stays blocked until a winning BUY trade or manual reset
- This is intentional to prevent overtrading after losses

⚠️ **Maximum 2 positions per chart is hard-coded**
- This cannot be changed via inputs (safety feature)
- If you need more positions, use multiple charts with different magic numbers

## Testing

Before going live:
1. Test on demo account first
2. Verify symbol checking works with your broker's symbol names
3. Test loss counter behavior by closing positions manually
4. Verify the reset function works as expected
5. Check that max positions limit is enforced

## Support

If you encounter issues:
- Check the Experts log for detailed messages
- Look for "TRADE BLOCKED" messages explaining why trades were rejected
- Verify your broker's symbol names match the allowed pairs list
- Ensure you're using the correct magic number (default: 123456)
