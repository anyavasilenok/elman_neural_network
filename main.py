from lib import main_function


if __name__ == "__main__":
    seq = [1, 2, 5, 15, 52, 203, 877, 4140, 21147]
    p = 5
    m = 2
    error = 0.0000001
    n = 500000
    alpha = 0.000000005
    predict = 5
    print(main_function(seq, p, m, error, n, alpha, predict))


