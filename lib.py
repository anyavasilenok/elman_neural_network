import numpy


def main_function(seq: list, p: int, m: int, error: float, n: int, alpha: float, predict: int):
    x, y = create_x_y(seq, p)
    x = extend_matrix(x, m)
    w1, w2 = create_weights(p, m)
    w1, w2 = learning(error, n, x, w1, w2, y, alpha)
    out = to_predict(w1, w2, predict, m, x, y)
    return out


def create_x_y(seq: list, p: int):
    x = []
    y = []
    i = 0
    while i + p < len(seq):
        a = []
        for j in range(p):
            a.append(seq[j + i])
        x.append(a)
        y.append(seq[i + p])
        i += 1

    x = numpy.array(x)
    y = numpy.array(y)
    return x, y


def extend_matrix(x, m):
    matrix = numpy.zeros((len(x), m))
    x = numpy.append(x, matrix, axis=1)
    return x


def create_weights(p, m):
    w1 = numpy.random.rand(p+m, m)
    w2 = numpy.random.rand(m, 1)
    return w1, w2


def activation_function(x):
    for i in range(len(x[0])):
        # x[0][i] = numpy.log(x[0][i] + (x[0][i]**2 + 1)**(0.5))
        x[0][i] = x[0][i]
    return x


def function_der(x):
    for i in range(len(x[0])):
        # x[0][i] = 1/((x[0][i] ** 2 + 1)**(0.5))
        x[0][i] = 1
    return x


def learning(error, n, x, w1, w2, y, alpha):
    this_error = 1
    k = 1
    while error <= this_error and k <= n:
        this_error = 0
        for i in range(len(x)):
            z = numpy.zeros((1, len(x[i])))
            for j in range(len(x[i])):
                z[0][j] = x[i][j]
            h = activation_function(z @ w1)
            out = activation_function(h @ w2)
            delta = out - y[i]
            w1 -= alpha * delta * z.T @ w2.T * function_der(z @ w1)
            w2 -= alpha * delta * h.T * function_der(h @ w2)
            this_error = this_error + (delta**2)[0]/2
        print(f"{k}: {this_error}")
        k += 1

    return w1, w2


def to_predict(w1, w2, predict, m, x, y):
    context = y[-1].reshape(1)
    X = x[-1, :-m]
    out = []
    for i in range(predict):
        X = X[1:]
        train = numpy.concatenate((X, context))
        X = numpy.concatenate((X, context))
        train = numpy.append(train, numpy.array([0] * m))
        h = train @ w1
        output = h @ w2
        context = output
        out.append(output[0])
    return out




