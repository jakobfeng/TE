true_values = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 8, 6, 4, 2, 4, 6, 8, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
for_1 = []
for i in true_values:
    if i == 10:
        for_1.append(i+0.1)
    else:
        for_1.append(9.1)
for_2 = [7] * len(true_values)

for_3 = [i + 1 for i in true_values]
for_4 = [i - 2 for i in true_values]


def get_mape(true, forecast):
    mapes = []
    for i in range(len(true)):
        observed = true[i]
        forecasted = forecast[i]
        mapes.append(100 * abs((observed - forecasted) / observed))
    return round(sum(mapes) / len(mapes), 2)


def compare(observed, for_first, for_sec):
    mape_1 = get_mape(observed, for_first)
    mape_2 = get_mape(observed, for_sec)
    print("Mape on high forecast: {}".format(mape_1))
    print("Mape on low forecast: {}".format(mape_2))


def plot(observed, for_first, for_sec):
    compare(observed, for_first, for_sec)
    import matplotlib.pyplot as plt
    plt.plot(observed, label="True")
    plt.plot(for_first, label="For. 1")
    plt.plot(for_sec, label="For. 2")
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    plot(true_values, for_1, for_4)
