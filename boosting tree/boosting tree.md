# boosting tree(提升树)

提升方法实际采用加法模型(即基函数的线性组合)与前向分步算法.**以决策树为基函数的提升方法称为提升树(boosting tree)**.

提升树模型可以表示为决策树的加法模型:
$$
f_M(x) = \sum_{m=1}^MT(x;\Theta_m)
$$
其中,$T(x;\Theta_m)$表示决策树;$\Theta_m$为决策树的参数;M为树的个数.

## 提升树算法

**回归问题的提升树算法流程:**

1. 初始化$f_0(x)=0$

2. 对m=1,2,...,M

3. 计算残差
   $$
   r_{mi} = y_i - f_{m-1}(x_i) , i=1,2,...,N
   $$

4. 拟合残差$r_{mi}$学习一个回归树,得到$T(x;\Theta_m)$

5. 更新$f_m(x) = f_{m-1}(x)+T(x;\Theta_m)$

6. 得到回归问题提升树
   $$
   f_M(x) = \sum_{m=1}^M T(x;\Theta_m)
   $$


### 梯度提升

提升树利用加法模型与前向分布算法实现学习的优化过程.当损失函数是**平方损失和指数损失函数**时,每一步优化是很简单的.

**梯度提升是利用损失函数的负梯度在当前模型的值作为回归问题提升树算法中的残差的近似值,拟合一个回归树**.

残差近似值表示为:
$$
- \left[ \frac{\partial L(y,f(x_i))}{\partial f(x_i)} \right]_{f(x)=f_{m-1}(x)}
$$
**梯度提升算法流程**:

1. 初始化
   $$
   f_0(x) = \arg \min_c \sum_{i=1}^N L(y_i,c)
   $$

2. 对m=1,2,...,M

3. 对i=1,2,...,N,计算
   $$
   r_{mi} = - \left[ \frac{\partial L(y,f(x_i))}{\partial f(x_i)} \right]_{f(x)=f_{m-1}(x)}
   $$

4. 对$r_{mi}$拟合一个回归树,得到第m棵树的叶节点区域$R_{mj}$,j=1,2,...,J

5. 对j=1,2,...,J,计算
   $$
   c_{mj} = \arg \min_c \sum_{x_i \in R_{mj}} L(y_i, f_{m-1}(x_i)+c)
   $$

6. 更新$f_m(x)=f_{m-1}(x)+\sum_{j=1}^Jc_{mj}I(x\in R_{mj})$

7. 得到回归树
   $$
   \hat{f}(x) = f_M(x) = \sum_{m=1}^M \sum_{j=1}^J c_{mj} I(x \in R_{mj})
   $$


