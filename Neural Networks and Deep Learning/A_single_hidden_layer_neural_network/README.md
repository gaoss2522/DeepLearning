LR

输入数据： X Y

z = WX+b
a = sigmoid(z)

Loss = -(yloga)-(1-y)(1-log(a))

Cost = 1/m * sum(Loss)

Min Cost

Loss 函数是凸函数，只会存个一个最优值， 不存在很多极小值的情况

对 Loss function 求导，得到的就是梯度下降最快的方向

更新dw db ，这个结果肯定比之前的Loss小，不断的更新，设置迭代次数
损失函数只会越来越小，不会跳跃，在两次误差变化很小的情况下可以终止

dL = dL da dz
dW = da dz DW

dL = DL da dz
Db = da dz db

经过推导可知：

DL/DZ = (a-y)
DL/DW = sum (a-y)X
DL/Db = sum (a-y)


-------------------------------------
planar

输入数据： X Y

z1 = W1X+b1
a1 = tanh(z1)
z2 = W2a1+b2
a2 = sigmoid(z2)

Loss = -(yloga2)-(1-y)(1-log(a2))

Cost = 1/m * sum(Loss)

dL   da2   dz2   da1   dz1
da2  dz2   da1   dz1   dw1
  (a2-y) *  W2 * (1-pow(a1,2)) * X


------------------------------------------------------
https://juejin.im/entry/58a1576e2f301e006952ded1

sigmoid, tanh, relu 函数的区别，对比

----------------------------------------------------------




