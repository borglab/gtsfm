



def pointTransfer(Tri, point1, point2):
    """

    See Equation (1) of Torr & Zisserman, Image & Vision Computing, 1997.

    Args:
        Tri
        point1
        point2

    Returns:
        point3
    """
    point3 = np.zeros((3,1))
    i = 1
    j = 2
    
    for l in range(3):
        v1 = 0
        v2 = 0
        for k in range(3):
            v1 = v1 + point1[k] * Tri[k,j,l]
            v2 = v2 + point1[k] * Tri[k,i,l]
        
        point3[l] = point2[i] * v1 - point2[j] * v2

    return point3