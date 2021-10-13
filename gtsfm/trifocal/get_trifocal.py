
"""
Reference:
https://github.com/san25dec/Trifocal-Tensor-Estimation/blob/master/GeometryReport.pdf
"""

import numpy as np

def getTrifocal( m1, m2, m3, m4, m5, m6 ):
    """
    m1, m2, m3, m4, m5, m6 are the 6 matches(structs) input
    mi.a1 mi.a2 mi.a3 are the 3 image points

    Args:
        m1
        m2
        m3
        m4
        m5
        m6

    Returns:
        Tri
    """
    lambda1 = np.linalg.inv([m1.a1, m2.a1, m3.a1])*m4.a1
    B1 = np.linalg.inv([lambda1(1)*m1.a1, lambda1(2)*m2.a1, lambda1(3)*m3.a1])

    print('***** 1 ****', lambda1)
    m1.a1
    m2.a1
    m3.a1
    m4.a1
    B1

    lambda2 = inv([m1.a2, m2.a2, m3.a2]) * m4.a2
    B2 = np.linalg.inv([lambda2(1)*m1.a2, lambda2(2)*m2.a2, lambda2(3)*m3.a2]);

    print('***** 2 ****')
    lambda2
    m1.a2
    m2.a2
    m3.a2
    m4.a2
    B2

    lambda3 = np.linalg.inv([m1.a3, m2.a3, m3.a3])*m4.a3;
    B3 = np.linalg.inv([lambda3(1)*m1.a3, lambda3(2)*m2.a3, lambda3(3)*m3.a3])

    print('***** 3 ****')
    lambda3
    m1.a3
    m2.a3
    m3.a3
    m4.a2
    B3

    X5 = struct
    X5.a1 = B1*m5.a1
    X5.a2 = B2*m5.a2
    X5.a3 = B3*m5.a3
    
    X6 = struct
    X6.a1 = B1*m6.a1
    X6.a2 = B2*m6.a2
    X6.a3 = B3*m6.a3
    
    Matrix = np.zeros((3, 5))
    x5 = X5.a1(1)
    y5 = X5.a1(2)
    w5 = X5.a1(3)
    
    x6 = X6.a1(1)
    y6 = X6.a1(2)
    w6 = X6.a1(3)
    
    Matrix[0,:] = np.array([-x5*y6 + x5*w6, x6*y5 - y5*w6, -x6*w5 + y6*w5, -x5*w6 + y5*w6, x5*y6 - y6*w5 ])
                
    x5 = X5.a2(1)
    y5 = X5.a2(2)
    w5 = X5.a2(3)
    
    x6 = X6.a2(1)
    y6 = X6.a2(2)
    w6 = X6.a2(3)
    
    Matrix[1,:] = np.array([-x5*y6 + x5*w6, x6*y5 - y5*w6, -x6*w5 + y6*w5, -x5*w6 + y5*w6, x5*y6 - y6*w5 ])
    x5 = X5.a3(1)
    y5 = X5.a3(2)
    w5 = X5.a3(3)
    
    x6 = X6.a3(1)
    y6 = X6.a3(2)
    w6 = X6.a3(3)
    
    Matrix[2,:] = np.array([-x5*y6 + x5*w6, x6*y5 - y5*w6, -x6*w5 + y6*w5, -x5*w6 + y5*w6, x5*y6 - y6*w5 ])
                    
    _, _, Vh = np.linalg.svd(Matrix)
    V = Vh.T
    
    T1 = V[:, -2]
    T2 = V[:, -1]
    
    a1 = T1[0]
    a2 = T1[1]
    a3 = T1[2]
    a4 = T1[3]
    a5 = T1[4]
    
    b1 = T2[0]
    b2 = T2[1]
    b3 = T2[2]
    b4 = T2[3]
    b5 = T2[4]
    
    polynomial = Fun(a1,b1,a2,b2,a5,b5)-Fun(a2,b2,a3,b3,a5,b5) ...
                -Fun(a2,b2,a4,b4,a5,b5)-Fun(a1,b1,a3,b3,a4,b4) ...
                +Fun(a2,b2,a3,b3,a4,b4)+Fun(a3,b3,a4,b4,a5,b5);
    
    alphas = np.roots(polynomial)
    alphas = alphas(imag(alphas) == 0)
    
    T = T1 + alphas(1)*T2
   
    XbyW = (T(4)-T(5))/(T(2)-T(3))
    YbyW = T(4)/(T(1)-T(3))
    ZbyW = T(5)/(T(1)-T(2))
    
    Matrix1 = np.zeros((4,4))
    
    x5 = X5.a1(1)
    y5 = X5.a1(2)
    w5 = X5.a1(3)
    
    x6 = X6.a1(1)
    y6 = X6.a1(2)
    w6 = X6.a1(3)
    
    Matrix1[0, :] = [w5, 0, -x5, w5-x5]
    Matrix1[1, :] = [0, w5, -y5, w5-y5]
    Matrix1[2, :] = [w6*XbyW, 0, -x6*ZbyW, w6-x6]
    Matrix1[3, :] = [0, w6*YbyW, -y6*ZbyW, w6-y6]
    
    printf('***** Matrix1 *****', Matrix1)
    Matrix2 = np.zeros((4,4))
    
    x5 = X5.a2(1)
    y5 = X5.a2(2)
    w5 = X5.a2(3)
   
    x6 = X6.a2(1)
    y6 = X6.a2(2)
    w6 = X6.a2(3)
    
    Matrix2[0, :] = [w5, 0, -x5, w5-x5];
    Matrix2[1, :] = [0, w5, -y5, w5-y5];
    Matrix2[2, :] = [w6*XbyW, 0, -x6*ZbyW, w6-x6];
    Matrix2[3, :] = [0, w6*YbyW, -y6*ZbyW, w6-y6];
    
    print('***** Matrix2 *****', Matrix2)

    Matrix3 = np.zeros((4,4))
    
    x5 = X5.a3(1)
    y5 = X5.a3(2)
    w5 = X5.a3(3)
    
    x6 = X6.a3(1)
    y6 = X6.a3(2)
    w6 = X6.a3(3)
    
    Matrix3[0, :] = [w5, 0, -x5, w5-x5]
    Matrix3[1, :] = [0, w5, -y5, w5-y5]
    Matrix3[2, :] = [w6*XbyW, 0, -x6*ZbyW, w6-x6]
    Matrix3[3, :] = [0, w6*YbyW, -y6*ZbyW, w6-y6]

    print('***** Matrix3 *****', Matrix3)
    
    _, _, Vh = np.linalg.svd(Matrix1)
    V = Vh.T
    abgd1 = V[:,-1]
    
    _, _, Vh = np.linalg.svd(Matrix2)
    V = Vh.T
    abgd2 = V[:,-1]
    
    _, _, Vh = np.linalg.svd(Matrix3)
    V = Vh.T
    abgd3 = V[:,-1]
    
    P1 = np.zeros((3, 4))
    
    P1[1, :] = [abgd1(1), 0, 0, abgd1[4]]
    P1[2, :] = [0, abgd1(2), 0, abgd1[4]]
    P1[3, :] = [0, 0, abgd1(3), abgd1[4]]
    
    P2 = np.zeros((3, 4))
    
    P2[1, :] = [abgd2(1), 0, 0, abgd2[4]]
    P2[2, :] = [0, abgd2(2), 0, abgd2[4]]
    P2[3, :] = [0, 0, abgd2(3), abgd2[4]]
    
    P3 = np.zeros((3, 4))
    
    P3[1, :] = [abgd3(1), 0, 0, abgd3[4]]
    P3[2, :] = [0, abgd3(2), 0, abgd3[4]]
    P3[3, :] = [0, 0, abgd3(3), abgd3[4]]
    
    P1 = B1 \ P1
    P2 = B2 \ P2
    P3 = B3 \ P3
    
    H = np.zeros((4, 4))
    H[1:3, :] = P1
    H[4, :] = cross4(H[1,:].T, H[2,:].T, H[3,:].T)
    
    H = np.linalg.inv(H)
    
    P2 = P2 @ H
    P3 = P3 @ H
    P1 = P1 @ H
    
    Tri = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                Tri[i,j,k] = P2[j,i] * P3[k,3] - P2[j,3] * P3[k,i]

    return Tri




def cross4( a, b, c ):
    """
    Args:
        a
        b
        c

    Returns:
        d
    """
    l1 = det([a[2:4]'; b(2:4)'; c(2:4)'])
    l2 = -det([[a(1),a(3:4)'] ; [b(1),b[3:4]']; [c(1),c[3:4]']])
    l3 = det([[a[1:2]',a(4)] ; [b[1:2]',b(4)]; [c[1:2]',c(4)]])
    l4 = -det([a[1:3]' ; b(1:3)'; c[1:3]'])
    d = [l1; l2; l3; l4]

    return d
    





