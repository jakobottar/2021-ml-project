# import numpy

def str2float(d:str):
    try: val = float(d)
    except ValueError: val = d
    return val

def LoadDataset(filename: str):
    data = []
    with open(filename, "r") as f:
        for line in f:
            terms = list(map(str2float, line.strip().split(",")))
            data.append(terms)

    return data[1:], data[0] # returns data and header