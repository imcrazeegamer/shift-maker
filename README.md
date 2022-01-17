# Shift Maker 9001
>And Yes... it's over 9000 because thats how we do things
## Numpy Madness, To many n-ds

### Functional Requirements
- User input a list of people
- User defines what shifts each person can't do and weights for the rest of the shifts (prefer to do or not)
- Backend will Output a Schedule (GUI Table) with the shifts for a week
- Backend will take into account: `Hours of Work Per Person`, `Hours of Rest Per Person`, `Min Amount of sleep`, `last week vs this week: weekend workers`
- Frontend To Work good on Android and ok on PC


### Rules
- Negative weights for:
    - `User inputed negitive weights`
    - `Consecutive Shifts Length > 12h`
    - `<8h of sleep in a day`
    - ~`Multiple shifts in a day` 
    - ~`Weekly shift distribution`
