dic = {
    (5, 4): [6, 1],
    (8, 8): [3, 7],
    (12, 16): [5, 5],    
    (7, 7): [4, 9],
    (7, 5): [1, 20],
}

dic2 = [(5, 4), (8, 8), (5, 8),(7, 7), (7, 5)]

# print(min(dic2))
def TopKey(x):
    """
        :return: return the min key and its value.
        """

    s = min(x, key=x.get)
    return s, x[s]


print(TopKey(dic))