class X:
    def __init__(self, k) -> None:
        self.k = k

array = [X(1), X(2), X(3), X(4)]
print(array)

for idx in range(len(array)):
    print(array[idx].k)