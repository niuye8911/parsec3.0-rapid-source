Maximize
1.31 itr  + 57.06 hash  + 1.318 probe  + [  + 0.174 hash * itr  + 1.47e-18 probe * itr  + 2.151e-16 probe * hash ]
Subject To
c1: iterationNum_0 - iterationNum = 0
c2: hashNum_0 - hashNum = 0
c3: probeNum_0 - probeNum = 0

c4: iterationNum_0 = 1 -> itr >= 0
c5: hashNum_0 = 1 -> hash >= 0
c6: probeNum_0 = 1 -> probe >= 0
c7:iterationNum0 = 1 -> itr >= 21
c8:iterationNum0 = 1 -> itr <= 25
c9:iterationNum0 = 1 -> hash >= 2
c10:iterationNum0 = 1 -> hash <= 8
c11:iterationNum1 = 1 -> itr >= 11
c12:iterationNum1 = 1 -> itr <= 20
c13:iterationNum1 = 1 -> hash >= 4
c14:iterationNum1 = 1 -> hash <= 8
c15:iterationNum2 = 1 -> itr >= 6
c16:iterationNum2 = 1 -> itr <= 10
c17:iterationNum2 = 1 -> hash >= 6
c18:iterationNum2 = 1 -> hash <= 8
c19:iterationNum3 = 1 -> itr >= 1
c20:iterationNum3 = 1 -> itr <= 5
c21:iterationNum3 = 1 -> hash >= 8
c22:iterationNum3 = 1 -> hash <= 8
c23:iterationNum = 1 -> iterationNum0 + iterationNum1 + iterationNum2 + iterationNum3 = 1
c24:hashNum0 = 1 -> hash >= 7
c25:hashNum0 = 1 -> hash <= 8
c26:hashNum0 = 1 -> probe >= 2
c27:hashNum0 = 1 -> probe <= 20
c28:hashNum1 = 1 -> hash >= 5
c29:hashNum1 = 1 -> hash <= 6
c30:hashNum1 = 1 -> probe >= 12
c31:hashNum1 = 1 -> probe <= 20
c32:hashNum2 = 1 -> hash >= 3
c33:hashNum2 = 1 -> hash <= 4
c34:hashNum2 = 1 -> probe >= 16
c35:hashNum2 = 1 -> probe <= 20
c36:hashNum3 = 1 -> hash >= 1
c37:hashNum3 = 1 -> hash <= 2
c38:hashNum3 = 1 -> probe >= 20
c39:hashNum3 = 1 -> probe <= 20
c40:hashNum = 1 -> hashNum0 + hashNum1 + hashNum2 + hashNum3 = 1



c41: 0.2204 itr  + 194 indicator 
 + 0.349 probe  + [  + 8.23e-14 hash * itr  + 1.87e-15 probe * itr  + 0.223 probe * hash ] <= 1915.000000
c42:particle = 1
c43:layer = 1

Bounds
2 <= itr <= 25
2 <= hash <= 8
2 <= probe <= 20
indicator = 1

Generals
itr
hash
probe

Binaries
iterationNum
iterationNum_0
hashNum
hashNum_0
probeNum
probeNum_0
iterationNum0
iterationNum1
iterationNum2
iterationNum3
hashNum0
hashNum1
hashNum2
hashNum3

End