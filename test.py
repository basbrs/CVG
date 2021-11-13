import itertools

param = "a", "b", "c", 3
left = [1, 2, 3, 4]

params = zip(range(len(left)), itertools.repeat(param, len(left)))

for p in params:
    line, (left, right, offset, cost_fn) = p
    print(line, left, right, offset, cost_fn)