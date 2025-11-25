# Session Filter - Quick Start Guide

## Your Optimal Settings (Asian + Post-London)

Based on your observation of better performance:

```
EnableSessionFilter = true

// Asian Session (12:00 AM - 9:00 AM)
TradeAsianSession = true
AsianStartHour = 12:00 AM (Midnight)
AsianEndHour = 9:00 AM

// London Session - SKIP
TradeLondonSession = false

// New York Session - SKIP
TradeNewYorkSession = false

// Post-London Session (5:00 PM - 10:00 PM)
TradeCustomSession = true
CustomStartHour = 5:00 PM
CustomEndHour = 10:00 PM
```

**Result**: EA trades only during 12:00 AM - 9:00 AM and 5:00 PM - 10:00 PM server time

**Note**: All times are selected from dropdown menus in 12-hour format (AM/PM)

## Alternative Presets

### Asian Only
```
EnableSessionFilter = true
TradeAsianSession = true
AsianStartHour = 12:00 AM
AsianEndHour = 9:00 AM
All others = false
```

### London Only
```
EnableSessionFilter = true
TradeLondonSession = true
LondonStartHour = 8:00 AM
LondonEndHour = 5:00 PM
All others = false
```

### New York Only
```
EnableSessionFilter = true
TradeNewYorkSession = true
NewYorkStartHour = 1:00 PM
NewYorkEndHour = 10:00 PM
All others = false
```

### 24/7 Trading
```
EnableSessionFilter = false
```

## Dropdown Time Options

All time inputs show a dropdown with 24 options:
- 12:00 AM (Midnight) through 11:00 AM
- 12:00 PM (Noon) through 11:00 PM

Simply select the desired time from the dropdown - no manual entry needed!

## Chart Display

Shows current session and status:
```
🕐 Session: Asian ✅
   Enabled: Asian Custom
```

## Log Messages

**Allowed**: `✅ ASIAN SESSION: Trading allowed`
**Blocked**: `⚠️ SESSION FILTER: Trading blocked`
