from datetime import datetime, timedelta, date


def get_weekly_expiry(date: datetime.date) -> datetime.date:
    days_ahead = (3 - date.weekday()) % 7
    return date + timedelta(days=days_ahead)


def get_monthly_expiry(date: datetime.date) -> datetime.date:
    if date.month == 12:
        next_month_date = datetime(date.year + 1, 1, 1).date()
    else:
        next_month_date = datetime(date.year, date.month + 1, 1).date()

    last_day = next_month_date - timedelta(days=1)
    monthly_expiry = last_day - timedelta(days=((last_day.weekday() - 3) % 7))
    if monthly_expiry < date:
        return get_monthly_expiry(date + timedelta(days=1))
    else:
        return monthly_expiry


def find_nearest_expiry(given_date):

    given_date = datetime.strptime(given_date, "%d-%m-%Y").date()
    weekly_expiry = get_weekly_expiry(given_date)
    monthly_expiry = get_monthly_expiry(given_date)

    return weekly_expiry, monthly_expiry


given_date = "26-09-2025"

# x,y=find_nearest_expiry(given_date)
x, y = find_nearest_expiry(given_date)
print(x)
print(y)
