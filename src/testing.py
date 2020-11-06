#Script for testing stuff
import matplotlib.pyplot as plt
import math


def plot_sine_curve():
    years = []
    for j in range(2):
        weeks = range(1, 53)
        years.extend(weeks)
    sine_weeks = [math.sin(math.pi*w/52) for w in years]
    print(years)
    print(sine_weeks)
    plt.plot(range(len(sine_weeks)), sine_weeks)
    plt.show()
    plt.close()

if __name__ == '__main__':
    plot_sine_curve()