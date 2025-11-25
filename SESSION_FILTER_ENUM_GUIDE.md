# Session Filter - Time Enum Dropdown Guide

## Overview

All time inputs now use **dropdown menus** with 12-hour format (AM/PM) for easy selection.

## TimeHour Enum - All Options

When you click on any time input, you'll see a dropdown with these 24 options:

```
12:00 AM (Midnight)  →  0  (00:00 in 24-hour)
1:00 AM              →  1  (01:00 in 24-hour)
2:00 AM              →  2  (02:00 in 24-hour)
3:00 AM              →  3  (03:00 in 24-hour)
4:00 AM              →  4  (04:00 in 24-hour)
5:00 AM              →  5  (05:00 in 24-hour)
6:00 AM              →  6  (06:00 in 24-hour)
7:00 AM              →  7  (07:00 in 24-hour)
8:00 AM              →  8  (08:00 in 24-hour)
9:00 AM              →  9  (09:00 in 24-hour)
10:00 AM             → 10  (10:00 in 24-hour)
11:00 AM             → 11  (11:00 in 24-hour)
12:00 PM (Noon)      → 12  (12:00 in 24-hour)
1:00 PM              → 13  (13:00 in 24-hour)
2:00 PM              → 14  (14:00 in 24-hour)
3:00 PM              → 15  (15:00 in 24-hour)
4:00 PM              → 16  (16:00 in 24-hour)
5:00 PM              → 17  (17:00 in 24-hour)
6:00 PM              → 18  (18:00 in 24-hour)
7:00 PM              → 19  (19:00 in 24-hour)
8:00 PM              → 20  (20:00 in 24-hour)
9:00 PM              → 21  (21:00 in 24-hour)
10:00 PM             → 22  (22:00 in 24-hour)
11:00 PM             → 23  (23:00 in 24-hour)
```

## How to Use

### Step 1: Open EA Settings
1. Drag EA onto chart
2. Go to "Inputs" tab
3. Scroll to "Trading Session Filter" section

### Step 2: Select Times from Dropdown
Instead of typing numbers, you now:
1. Click on the time input field
2. See a dropdown with 24 options in 12-hour format
3. Select the desired time (e.g., "5:00 PM" instead of typing "17")

### Step 3: Configure Your Sessions
Example for Asian + Post-London:
```
AsianStartHour:  Select "12:00 AM (Midnight)"
AsianEndHour:    Select "9:00 AM"
CustomStartHour: Select "5:00 PM"
CustomEndHour:   Select "10:00 PM"
```

## Common Session Configurations

### Asian Session (Tokyo)
```
Start: 12:00 AM (Midnight)
End:   9:00 AM
```

### London Session
```
Start: 8:00 AM
End:   5:00 PM
```

### New York Session
```
Start: 1:00 PM
End:   10:00 PM
```

### London Open (High Volatility)
```
Start: 8:00 AM
End:   10:00 AM
```

### NY-London Overlap
```
Start: 1:00 PM
End:   5:00 PM
```

### Post-London (Your Preference)
```
Start: 5:00 PM
End:   10:00 PM
```

### Late Night Asian
```
Start: 10:00 PM
End:   2:00 AM (wraps midnight)
```

## Benefits of Enum Dropdown

✅ **No Typos**: Can't enter invalid hours
✅ **Clear Format**: See AM/PM instead of 24-hour
✅ **Quick Selection**: Click and choose
✅ **Visual**: Easy to understand at a glance
✅ **Consistent**: All time inputs work the same way

## Midnight Wrap Examples

### Example 1: Late Night Trading (10 PM - 2 AM)
```
Start: 10:00 PM  (22 in 24-hour)
End:   2:00 AM   (2 in 24-hour)

Trading Hours:
✅ 10:00 PM (22:00)
✅ 11:00 PM (23:00)
✅ 12:00 AM (00:00) ← Midnight
✅ 1:00 AM  (01:00)
❌ 2:00 AM  (02:00) ← End time (not included)
```

### Example 2: Avoid Lunch Hour (Trade 8 AM - 12 PM, 2 PM - 6 PM)
Use two sessions:
```
Session 1 (Morning):
Start: 8:00 AM
End:   12:00 PM (Noon)

Session 2 (Afternoon):
Start: 2:00 PM
End:   6:00 PM
```

## Visual Reference

### Morning Hours (AM)
```
🌙 12:00 AM ─┐
   1:00 AM  │
   2:00 AM  │
   3:00 AM  │ Night/Early Morning
   4:00 AM  │
   5:00 AM  │
🌅 6:00 AM  ─┤
   7:00 AM  │
   8:00 AM  │ Morning
   9:00 AM  │
   10:00 AM │
   11:00 AM ─┘
```

### Afternoon/Evening Hours (PM)
```
☀️ 12:00 PM ─┐
   1:00 PM  │
   2:00 PM  │ Afternoon
   3:00 PM  │
   4:00 PM  │
🌆 5:00 PM  ─┤
   6:00 PM  │
   7:00 PM  │ Evening
   8:00 PM  │
   9:00 PM  │
🌙 10:00 PM ─┤
   11:00 PM ─┘ Night
```

## Tips

1. **Remember Server Time**: All times are broker server time, not your local time
2. **Test First**: Use Strategy Tester to verify session times work as expected
3. **Check Logs**: Enable `ShowDebugInfo = true` to see session validation messages
4. **Start Simple**: Begin with one session, then add more as needed
5. **Document Your Settings**: Save your optimal session configuration

## Comparison: Old vs New

### Old Way (Manual Entry)
```
AsianStartHour = 0    // What time is 0?
AsianEndHour = 9      // Is this AM or PM?
CustomStartHour = 17  // Need to convert to 24-hour
```

### New Way (Dropdown)
```
AsianStartHour = 12:00 AM (Midnight)  // Clear!
AsianEndHour = 9:00 AM                // Obvious!
CustomStartHour = 5:00 PM             // No conversion needed!
```

Much cleaner and easier to understand! 🎉
