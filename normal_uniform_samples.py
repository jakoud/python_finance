from numpy.random import uniform, normal
from math import sqrt, log, pi, cos
from scipy.stats import shapiro


def bm_transform(sample1, sample2):
    new_sample1, new_sample2 = [], []
    length = len(sample1)
    for i in range(length):
        new1 = sqrt(-2*log(sample1[i]))*cos(2*pi*sample2[i])
        new_sample1.append(new1)
    return new_sample1


def test_samples(sample1, sample2):
    error = 0
    for i in range(len(sample1)):
        error += sqrt((sample1[i]-sample2[i])**2)
    error = str(error/len(sample1))
    print(f'cumulative mean square difference between samples is {error}')


def main():
    set_size = 100
    sample1, sample2 = uniform(size=set_size), uniform(size=set_size)
    normal_sample = normal(size=set_size)
    transformed_sample = bm_transform(sample1, sample2)
    test_samples(sorted(transformed_sample), sorted(normal_sample))
    print(shapiro(transformed_sample))


if __name__ == "__main__":
    main()
