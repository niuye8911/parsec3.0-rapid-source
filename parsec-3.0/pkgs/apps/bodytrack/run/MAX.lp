Maximize
 + 0.0303734 layer_seg0_V + 0.320441 layer_seg0 + 0.00444365 layer_seg1_V + 0.372303 layer_seg1 + 0.00351873 layer_seg2_V + 0.375082 layer_seg2 + 0.000467529 layer_seg3_V + 0.387285 layer_seg3 + 9.19495e-06 particle_seg0_V + 0.595665 particle_seg0 + 6.383e-06 particle_seg1_V + 0.597178 particle_seg1 + 2.31824e-09 particle_seg2_V + 0.602836 particle_seg2 + 7.14994e-09 particle_seg3_V + 0.601174 particle_seg3 + 3.63749e-09 particle_seg4_V + 0.599498 particle_seg4 + 9.74693e-06 particle_seg5_V + 0.576895 particle_seg5 + 7.24332e-07 particle_seg6_V + 0.601254 particle_seg6 + 2.17355e-07 particle_seg7_V + 0.603128 particle_seg7 + [  + 0 layer ^ 2 + 0 particle ^ 2 + 0 particle * layer ]
Subject To
c1: layerNum_0 - layerNum = 0
c2: particleNum_0 - particleNum = 0

c3: layerNum_0 = 1 -> layer >= 0
c4: particleNum_0 = 1 -> particle >= 0
c5:layer_seg0 = 1 -> layer <= 2
c6:layer_seg0 = 1 -> layer >= 1
c7:layer_seg0 = 1 -> layer_seg0_V - layer = 0
c8:layer_seg0 = 0 -> layer_seg0_V = 0
c9:layer_seg1 = 1 -> layer <= 3
c10:layer_seg1 = 1 -> layer >= 2
c11:layer_seg1 = 1 -> layer_seg1_V - layer = 0
c12:layer_seg1 = 0 -> layer_seg1_V = 0
c13:layer_seg2 = 1 -> layer <= 4
c14:layer_seg2 = 1 -> layer >= 3
c15:layer_seg2 = 1 -> layer_seg2_V - layer = 0
c16:layer_seg2 = 0 -> layer_seg2_V = 0
c17:layer_seg3 = 1 -> layer <= 5
c18:layer_seg3 = 1 -> layer >= 4
c19:layer_seg3 = 1 -> layer_seg3_V - layer = 0
c20:layer_seg3 = 0 -> layer_seg3_V = 0
c21:layer_seg0 + layer_seg1 + layer_seg2 + layer_seg3  = 1
c22:particle_seg0 = 1 -> particle <= 533
c23:particle_seg0 = 1 -> particle >= 100
c24:particle_seg0 = 1 -> particle_seg0_V - particle = 0
c25:particle_seg0 = 0 -> particle_seg0_V = 0
c26:particle_seg1 = 1 -> particle <= 966
c27:particle_seg1 = 1 -> particle >= 533
c28:particle_seg1 = 1 -> particle_seg1_V - particle = 0
c29:particle_seg1 = 0 -> particle_seg1_V = 0
c30:particle_seg2 = 1 -> particle <= 1399
c31:particle_seg2 = 1 -> particle >= 966
c32:particle_seg2 = 1 -> particle_seg2_V - particle = 0
c33:particle_seg2 = 0 -> particle_seg2_V = 0
c34:particle_seg3 = 1 -> particle <= 1833
c35:particle_seg3 = 1 -> particle >= 1399
c36:particle_seg3 = 1 -> particle_seg3_V - particle = 0
c37:particle_seg3 = 0 -> particle_seg3_V = 0
c38:particle_seg4 = 1 -> particle <= 2266
c39:particle_seg4 = 1 -> particle >= 1833
c40:particle_seg4 = 1 -> particle_seg4_V - particle = 0
c41:particle_seg4 = 0 -> particle_seg4_V = 0
c42:particle_seg5 = 1 -> particle <= 2700
c43:particle_seg5 = 1 -> particle >= 2266
c44:particle_seg5 = 1 -> particle_seg5_V - particle = 0
c45:particle_seg5 = 0 -> particle_seg5_V = 0
c46:particle_seg6 = 1 -> particle <= 3133
c47:particle_seg6 = 1 -> particle >= 2700
c48:particle_seg6 = 1 -> particle_seg6_V - particle = 0
c49:particle_seg6 = 0 -> particle_seg6_V = 0
c50:particle_seg7 = 1 -> particle <= 4000
c51:particle_seg7 = 1 -> particle >= 3133
c52:particle_seg7 = 1 -> particle_seg7_V - particle = 0
c53:particle_seg7 = 0 -> particle_seg7_V = 0
c54:particle_seg0 + particle_seg1 + particle_seg2 + particle_seg3 + particle_seg4 + particle_seg5 + particle_seg6 + particle_seg7  = 1


c55:  + 5.92376 layer_seg0_V + 0.349425 layer_seg0 + 0 layer_seg1_V + 10.9162 layer_seg1 + 0 layer_seg2_V + 6.90773 layer_seg2 + 0 layer_seg3_V + 0 layer_seg3 + 0.00886135 particle_seg0_V + 0 particle_seg0 + 0.0115931 particle_seg1_V + 0 particle_seg1 + 0.011972 particle_seg2_V + 0 particle_seg2 + 0.00573408 particle_seg3_V + 8.82002 particle_seg3 + 0.0065765 particle_seg4_V + 7.27587 particle_seg4 + 0 particle_seg5_V + 20.0894 particle_seg5 + 0 particle_seg6_V + 17.7362 particle_seg6 + 0 particle_seg7_V + 9.84177 particle_seg7 + [  + 2.5771 layer ^ 2 + 5.64359e-06 particle ^ 2 + 0.00819034 particle * layer ] <= 213.000000
c56:particleNum = 1
c57:layerNum = 1

Bounds
1 <= layer <= 5
100 <= particle <= 4000
indicator = 1

Generals
layer
particle

Binaries
layerNum
layerNum_0
layer_seg0
layer_seg1
layer_seg2
layer_seg3
particleNum
particleNum_0
particle_seg0
particle_seg1
particle_seg2
particle_seg3
particle_seg4
particle_seg5
particle_seg6
particle_seg7

End
