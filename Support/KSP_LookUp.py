def ksp(height, width):
    ksp_h = {
        "k1" : 0,
        "k2" : 0,
        "s1" : 0,
        "s2" : 0,
        "p1" : 0,
        "p2" : 0,
        "igtd_h": height,
    }
    ksp_w = {
        "k1" : 0,
        "k2" : 0,
        "s1" : 0,
        "s2" : 0,
        "p1" : 0,
        "p2" : 0,
        "igtd_w": width,
    }
    match height:
        case 2:
            ksp_h["k1"] = 1
            ksp_h["s1"] = 1
            ksp_h["k2"] = 1
            ksp_h["s2"] = 1
        case 3:
            ksp_h["k1"] = 1
            ksp_h["s1"] = 1
            ksp_h["k2"] = 2
            ksp_h["s2"] = 1
        case 4:
            ksp_h["k1"] = 2
            ksp_h["s1"] = 1
            ksp_h["k2"] = 2
            ksp_h["s2"] = 1
        case _: # this means the height is 5 or bigger hence the first conv layer will be 2, 2
            ksp_h["k1"] = 2
            ksp_h["s1"] = 2

    match height:
        case 5:
            ksp_h["k2"] = 2
            ksp_h["s2"] = 1
        case 6:
            ksp_h["k2"] = 3
            ksp_h["s2"] = 1
        case 7:
            ksp_h["k2"] = 4
            ksp_h["s2"] = 1
        case 8:
            ksp_h["k2"] = 4
            ksp_h["s2"] = 2
            ksp_h["p2"] = 1
        case 9:
            ksp_h["k2"] = 3
            ksp_h["s2"] = 2
        case 10:
            ksp_h["k2"] = 4
            ksp_h["s2"] = 2

    match width:
        case 2:
            ksp_w["k1"] = 1
            ksp_w["s1"] = 1
            ksp_w["k2"] = 1
            ksp_w["s2"] = 1
        case 3:
            ksp_w["k1"] = 1
            ksp_w["s1"] = 1
            ksp_w["k2"] = 2
            ksp_w["s2"] = 1
        case 4:
            ksp_w["k1"] = 2
            ksp_w["s1"] = 1
            ksp_w["k2"] = 2
            ksp_w["s2"] = 1
        case _:  # this means the height is 5 or bigger hence the first conv layer will be 2, 2
            ksp_w["k1"] = 2
            ksp_w["s1"] = 2

    match width:
        case 5:
            ksp_w["k2"] = 2
            ksp_w["s2"] = 1
        case 6:
            ksp_w["k2"] = 3
            ksp_w["s2"] = 1
        case 7:
            ksp_w["k2"] = 4
            ksp_w["s2"] = 1
        case 8:
            ksp_w["k2"] = 4
            ksp_w["s2"] = 2
            ksp_w["p2"] = 1
        case 9:
            ksp_w["k2"] = 3
            ksp_w["s2"] = 2
        case 10:
            ksp_w["k2"] = 4
            ksp_w["s2"] = 2

    return ksp_h, ksp_w