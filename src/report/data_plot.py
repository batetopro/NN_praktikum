import matplotlib.pyplot as plt


def decode(l):
    for k, c in enumerate(l):
        if c == "1":
            return 9 - k


def plot(input, label):
    x, y, color = [], [], []
    colors = [
        '#FF0000',
        '#C71585',
        '#808000',
        '#FFD700',
        '#FF00FF',
        '#800080',
        '#483D8B',
        '#8B0000',
        "#228B22",
        "#008080"
    ]
    # colors = ['red', 'green', 'blue', 'purple']

    for p, l in zip(input, label):
        color.append(colors[decode(l)])
        x.append(p[0])
        y.append(p[1])

    plt.scatter(x, y, color=color)
    plt.show()