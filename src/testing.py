#Script for testing stuff
from datetime import datetime

year = "2018"
week = "W0"
first_date = datetime.strptime(year + " " + week + " w1", "%Y W%W w%w").date()
print(first_date)
next_week = "W1"
next_date = datetime.strptime(year + " " + next_week + " w1", "%Y W%W w%w").date()
print(next_date)