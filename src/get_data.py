def get_data(start=None, stop=None, specific=None):
    f = open('../data/data.txt')

    data = f.read()

    data = data.split('\n')

    data = [x.split(',', maxsplit=2) for x in data]

    return data[start:stop] if (specific == None) else [data[specific - 1]]
