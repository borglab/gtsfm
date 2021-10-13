

import numpy as np

def RANSACTrifocal(vpts1, vpts2, vpts3, matchedTriplets ):
   """Computing the estimated Trifocal tensor using matches1 and matches2.

    Args:
        vpts1
        vpts2
        vpts3
        matchedTriplets

    Returns:
        Tfinal
        matcheper
        m
        m1
   """

    matcheper = 0
    iter = 0
    Tfinal = 0
    
    m = struct;
    m1 = struct;    
    while matcheper < 0.85 and iter < 6000:
         
        iter += 1
        %iter
        n = matchTriplets.shape[0]
        idx = np.random.choice(n, size=(6,1))
        
        m(1).a1 = [vpts1(matchedTriplets(idx(1),1)).Location'; 1]
        m(1).a2 = [vpts2(matchedTriplets(idx(1),2)).Location'; 1]
        m(1).a3 = [vpts3(matchedTriplets(idx(1),3)).Location'; 1]

        m(2).a1 = [vpts1(matchedTriplets(idx(2),1)).Location'; 1]
        m(2).a2 = [vpts2(matchedTriplets(idx(2),2)).Location'; 1]
        m(2).a3 = [vpts3(matchedTriplets(idx(2),3)).Location'; 1]

        m(3).a1 = [vpts1(matchedTriplets(idx(3),1)).Location'; 1]
        m(3).a2 = [vpts2(matchedTriplets(idx(3),2)).Location'; 1]
        m(3).a3 = [vpts3(matchedTriplets(idx(3),3)).Location'; 1]

        m(4).a1 = [vpts1(matchedTriplets(idx(4),1)).Location'; 1]
        m(4).a2 = [vpts2(matchedTriplets(idx(4),2)).Location'; 1]
        m(4).a3 = [vpts3(matchedTriplets(idx(4),3)).Location'; 1]

        m(5).a1 = [vpts1(matchedTriplets(idx(5),1)).Location'; 1]
        m(5).a2 = [vpts2(matchedTriplets(idx(5),2)).Location'; 1]
        m(5).a3 = [vpts3(matchedTriplets(idx(5),3)).Location'; 1]

        m(6).a1 = [vpts1(matchedTriplets(idx(6),1)).Location'; 1]
        m(6).a2 = [vpts2(matchedTriplets(idx(6),2)).Location'; 1]
        m(6).a3 = [vpts3(matchedTriplets(idx(6),3)).Location'; 1]
        
        lamb1 = ([m(1).a1, m(2).a1, m(3).a1])
        lambda1 = lamb1\m(4).a1
        lamb2 = ([m(1).a2, m(2).a2, m(3).a2])
        lambda2 = lamb2\m(4).a2
        lamb3 = ([m(1).a3, m(2).a3, m(3).a3])
        lambda3 = lamb3\m(4).a3
        
        B1 = ([lambda1(1) * m(1).a1, lambda1(2) * m(2).a1, lambda1(3) * m(3).a1])
        B2 = ([lambda2(1) * m(1).a2, lambda2(2) * m(2).a2, lambda2(3) * m(3).a2])
        B3 = ([lambda3(1) * m(1).a3, lambda3(2) * m(2).a3, lambda3(3) * m(3).a3])
        
        w1, w2, w3, w4, w5, w6 = 1, 1, 1, 1, 1, 1
        
	    # These conditions ensure that a poor choice of points are not selected
		# to compute the Trifocal Tensor. The check is based on the condition
		# number of the matrices involved in computing the trifocal tensor	
        if rcond(lamb1) < 0.0000001 or np.isnan(rcond(lamb1)):
            w1 = 0

        if rcond(lamb2) < 0.0000001 or np.isnan(rcond(lamb2)):
            w2 = 0

        if rcond(lamb3) < 0.0000001 or np.isnan(rcond(lamb3)):
            w3 = 0

        if rcond(B1) < 0.0000001 or np.isnan(rcond(B1)):
            w4 = 0

        if rcond(B2) < 0.0000001 or np.isnan(rcond(B2)):
            w5 = 0

        if rcond(B3) < 0.0000001 or np.isnan(rcond(B3)):
            w6 = 0
       
	    # Only if the above conditions are not violated, proceed ahead.
		# Otherwise ignore this set of points	
        if all([w1, w2, w3, w4, w5, w6]):
         
            Tri = getTrifocal(m[1], m[2], m[3], m[4], m[5], m[6])

            matcheper1 = getErrorTrifocal(vpts1, vpts2, vpts3, matchedTriplets, Tri)
            iter += 1
            if(matcheper1 > matcheper):
                Tfinal = Tri;
                matcheper = matcheper1

            if matcheper < 0.85 and matcheper > 0.70:
                m1 = m

            # matcheper

        print(f"Iteration: {iter} Best percentage: {matcheper}")


    return Tfinal, matcheper, m, m1


