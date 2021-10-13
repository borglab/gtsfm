
def getErrorTrifocal(vpts1, vpts2, vpts3, matchedTriplets, Tri) -> float:
    """Used to obtain error between 2 coordinates"""
    
    count = 0
    m = struct
    for i in range(len(matchedTriplets,1)):

        m(1).a1 = [vpts1(matchedTriplets(i,1)).Location'; 1]
        m(1).a2 = [vpts2(matchedTriplets(i,2)).Location'; 1]
        m(1).a3 = [vpts3(matchedTriplets(i,3)).Location'; 1]
        p3 = pointTransfer(Tri, m(1).a1, m(1).a2);
        p3 = p3/p3(3);
        error = sqrt(sum((p3 - m(1).a3).*(p3 - m(1).a3)))

        if error<5:
            count += 1

    error_ = count/size(matchedTriplets,1)
    return error_
