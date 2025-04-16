def values(value, table):
    match value:
        case 2:
            table["input"] = 2
            table["k1"] = 1
            table["s1"] = 1
            table["k2"] = 1
            table["s2"] = 1
        case 3:
            table["input"] = 2
            table["k1"] = 1
            table["s1"] = 1
            table["k2"] = 2
            table["s2"] = 1
        case 4:
            table["input"] = 2
            table["k1"] = 2
            table["s1"] = 1
            table["k2"] = 2
            table["s2"] = 1
        case 5:
            table["input"] = 2
            table["k1"] = 2
            table["s1"] = 2
            table["k2"] = 2
            table["s2"] = 1
        case 6:
            table["input"] = 2
            table["k1"] = 2
            table["s1"] = 1
            table["k2"] = 2
            table["s2"] = 2
        case 7:
            table["input"] = 2
            table["k1"] = 2
            table["s1"] = 2
            table["k2"] = 3
            table["s2"] = 2
            table["p2"] = 1
        case 8:
            table["input"] = 3
            table["k1"] = 2
            table["s1"] = 2
            table["k2"] = 3
            table["s2"] = 1
        case 9:
            table["input"] = 2
            table["k1"] = 2
            table["s1"] = 2
            table["k2"] = 3
            table["s2"] = 2
        case 10:
            table["input"] = 3
            table["k1"] = 2
            table["s1"] = 2
            table["k2"] = 2
            table["s2"] = 2
            table["p2"] = 1
        case 11:
            table["input"] = 3
            table["k1"] = 2
            table["s1"] = 2
            table["k2"] = 3
            table["s2"] = 2
            table["p2"] = 1
        case 12:
            table["input"] = 3
            table["k1"] = 2
            table["s1"] = 2
            table["k2"] = 2
            table["s2"] = 2
        case 13:
            table["input"] = 3
            table["k1"] = 2
            table["s1"] = 2
            table["k2"] = 3
            table["s2"] = 2
        case 14:
            table["input"] = 3
            table["k1"] = 2
            table["s1"] = 2
            table["k2"] = 4
            table["s2"] = 2
        case 15:
            table["input"] = 3
            table["k1"] = 2
            table["s1"] = 2
            table["k2"] = 4
            table["s2"] = 3
            table["p2"] = 2

    return table


def ksp(height, width):
    ksp_h = {
        "input" : 0,
        "k1" : 0,
        "k2" : 0,
        "s1" : 0,
        "s2" : 0,
        "p1" : 0,
        "p2" : 0,
        "height" : height,
    }
    ksp_w = {
        "input" : 0,
        "k1" : 0,
        "k2" : 0,
        "s1" : 0,
        "s2" : 0,
        "p1" : 0,
        "p2" : 0,
        "width" : width,
    }

    return values(height, ksp_h), values(width, ksp_w)