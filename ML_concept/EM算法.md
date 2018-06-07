#EM算法

EM算法也叫期望极大算法是一种**迭代算法**。用于**含有隐变量(hidden variable)**的概率模型参数的**极大似然估计**，或**极大后验概率估计**。EM算法的每次迭代由两步组成：E-step，求期望;M-step,求极大。

## 问题建模

###问题概述

概率模型既含有观测变量(训练数据集)，又含有隐变量或潜在变量(latent variable)。如果概率模型的变量都是观测变量，那么给定数据，可以直接用极大似然估计法估计模型参数。但是，当模型含有隐变量时，就不能简单的使用这些估计方法。**EM算法就是含有隐变量的概率模型参数的极大似然估计法或者叫极大后验概率估计法**。

### 建模

#### 举例(三个硬币模型)

假设有3枚硬币，分别记作A，B，C。这些硬币正面出现的概率分别为$\pi,p,q$。进行如下掷硬币试验：先掷硬币A，根据其结果选出硬币B或硬币C，正面选硬币B，反面选硬币C；然后掷选出的硬币，掷硬币的结果，出现正面记作1，出现反面记作0；独立地重复n次试验(这里，n=10)，观测结果为：1,1,0,1,0,0,1,0,1,1。

假设只能观测到掷硬币的结果，不能观测掷硬币的过程，问如何估计三硬币正面出现的概率，即三硬币模型的参数。

**解：****三硬币模型**可以写作：
$$
\begin{align*}
P(y|\theta) &= \sum_z P(y,z|\theta) = \sum_z P(z|\theta)P(y|z,\theta) \\
&= \pi p^y(1-p)^{1-y}+(1-\pi)q^y(1-q)^{1-y}
\end{align*}
$$
这里，随机变量y是观测变量，表示一次试验观测的结果是1或0；随机变量z是隐变量，表示未观测到的掷硬币A的结果；$\theta=(\pi,p,q)$是模型参数。这一模型是以上数据的生成模型，注意，随机变量y的数据可以观测，随机变量z的数据不可观测。

将观测数据表示为$Y = (y_1,y_2,\dots,y_n)^T$，未观测数据表示为$Z=(z_1,z_2,\dots,z_n)^T$，则**观测数据的似然函数**为：
$$
P(Y|\theta) = \sum_Z P(Y,Z|\theta)= \sum_Z P(Z|\theta)P(Y|Z,\theta)
$$
即
$$
P(Y|\theta) = \prod_{j=1}^{n}\left[ \pi p^{y_j}(1-p)^{1-y_j}+(1-\pi)q^{y_j}(1-q)^{1-y_j}\right]
$$
**求模型参数$\theta = (\pi,p,q)$的极大似然估计**，即
$$
\hat{\theta} = \arg \max_\theta \log P(Y|\theta)
$$
**目标函数不是凸函数**，无法通过凸优化来进行求解。这里**只有通过迭代的方法求解**。**EM算法就是一种求解此问题的迭代方法**。

使用EM算法求解上述目标函数的过程如下：

**初始化：**

EM算法首先选取参数的估计值，记作$\theta^0 = (\pi^0,p^0,q^0)$，然后通过下面的步骤迭代计算参数的估计值，直至收敛为止。第i次迭代参数的估计值为$\theta^i = (\pi^i,p^i,q^i)$。EM算法的第i+1次迭代如下：

**E-step：**

计算在模型参数$\pi_i,p_i,q_i$下观测数据$y_j$来自**掷硬币B的概率**:
$$
\mu_j^{i+1} = \frac{\pi^i(p^i)^{y_j}(1-p^i)^{1-y_j}}{\pi^i(p^i)^{y_j}(1-p^i)^{1-y_j} +(1-\pi^i)(q^i)^{y_j}(1-q^i)^{1-y_j}}
$$
**M-step:**

计算模型参数的新估计值
$$
\begin{align*}
\pi^{i+1} &= \frac{1}{n}\sum_{j=1}^{n}\mu_j^{i+1} \\
p^{i+1} &= \frac{\sum_{j=1}^{n}\mu_j^{i+1}y_j}{\sum_{j=1}^{n}\mu_j^{i+1}} \\
q^{i+1} &= \frac{\sum_{j=1}^{n}(1-\mu_j^{i+1})y_j}{\sum_{j=1}^{n}(1-\mu_j^{i+1})}
\end{align*}
$$
以上n表示观测到的样本数。假设模型参数的初值取为：$\pi^0=0.5,p^0=0.5,q^0=0.5$，通过不断地迭代，得到最终的模型参数$\theta$的极大似然估计：
$$
\hat{\pi}=0.5,\hat{p}=0.6,\hat{q}=0.6
$$
需要注意的是**此处的结果会因为初始值的不同而不同的**。算法接的标准是找到了最大的期望值也就是$\mu_j^{i+1}$为最大值。

**总结：**

一般地，用Y表示观测随机变量的数据，Z表示隐随机变量的数据。Y和Z连在一起称为完全数据(complete-data)，观测数据Y又称为不完全数据(incomplete-data)。假设给定观测数据Y，其概率分布是$P(Y|\theta)$，其中$\theta$为需要估计的模型参数，那么不完全数据Y的似然函数是$P(Y|\theta)$，对数似然函数$l(\theta) = \log P(Y|\theta) $；假设Y和Z的联合概率分布是$P(Y,Z|\theta)$，那么完全数据的对数似然函数是$l(\theta) = \log P(Y,Z|\theta)$。

EM算法通过迭代求对数似然函数的极大似然估计。每次迭代包含两步：

- E-step 求期望
- M-step 求极大化

#### EM算法

输入：观测变量数据Y，隐变量数据Z，联合分布$P(Y,Z|\theta)$，条件分布$P(Z|Y,\theta)$;

输出：模型参数$\theta$

1. 选择参数的初值$\theta^0$,开始迭代

2. E-step：记$\theta^i$为第i次迭代参数$\theta$的估计值，在第i+1次迭代的E步，计算：
   $$
   \begin{align*}
   Q(\theta,\theta^i) &= E_Z[\log P(Y,Z|\theta)|Y,\theta^i] \\
   &= \sum_Z \log P(Y,Z|\theta)P(Z|Y,\theta^i)
   \end{align*}
   $$
   这里，$P(Z|Y,\theta^i)​$是在给定观测数据Y和当前的参数估计$\theta^i​$下隐变量数据Z的条件概率分布;

3. M-step:求使$Q(\theta,\theta^i)$极大化的$\theta$，确定第i+1次迭代的参数的估计值$\theta^{i+1}$
   $$
   \theta^{i+1} = \arg \max_\theta Q(\theta,\theta^i)
   $$

4. 重复第2步和第3步，直到收敛。$Q(\theta,\theta^i)$也称Q函数。

####Q函数

**Q函数**用于表示完全数据的对数似然函数$\log P(Y,Z|\theta)$关于在给定观测数据Y和当前参数$\theta^i$下对为观测数据Z的条件概率分布$P(Z|Y,\theta^i)$的期望。即:
$$
\begin{aligned}
Q(\theta,\theta^i) &= E_Z[\log P(Y,Z|\theta) |Y, \theta^i] \\
&= \sum_ZP(Z|Y,\theta^i) \log P(Y,Z|\theta)
\end{aligned}
$$
**关于EM算法的几点说明:**

1. 参数的初值可以任意选择,但需要注意**EM算法对初值敏感.**

2. **E步求$Q(\theta,\theta^i)$**,Q函数式中Z是未观测数据,Y是观测数据.注意,Q函数中的**第一个参数表示要极大化的参数**;**第二个参数表示参数当前的估计值**.每次迭代其实就是在就Q函数及其极大值.

3. **M步求$Q(\theta,\theta^i)$的极大化**,得到$\theta^{i+1}$,完成一次迭代$\theta^i \rightarrow \theta^{i+1}$

4. 给出**停止迭代的条件**.给出一个阈值$\epsilon$,当以下条件成立时,停止迭代:
   $$
   \|\theta^{i+1} - \theta^i\| < \epsilon \ 或 \ \|Q(\theta^{i+1},\theta^i) - Q(\theta^i,\theta^i)\| < \epsilon
   $$
   

### 问题

前面我们在三硬币模型的例子中使用的似然估计模型为观测数据的似然估计模型，而不是原始的EM算法中使用的完全数据似然估计模型。**怎么证明通过求解不完全数据的似然估计模型来间接求解完全数据的似然估计模型呢？**

##EM算法的导出

面对一个含有隐变量的概率模型,目标是极大化观测数据(不完全数据)Y关于参数$\theta$的对数似然函数,即极大化:
$$
\begin{aligned}
L(\theta) &= \log P(Y|\theta) = log \sum_Z P(Y,Z|\theta) \\
&= \log \left( \sum_ZP(Y|Z,\theta) P(Z|\theta) \right)
\end{aligned}
$$
事实上,EM算法是通过迭代逐步近似极大化$L(\theta)$的,所以每一次迭代结束后,需要满足$L(\theta^i) \leq L(\theta)$.所以考虑二者的差:
$$
\begin{aligned}
L(\theta) - L(\theta^i) &= \log \left( \sum_Z P(Y|Z,\theta)P(Z|\theta) \right) - \log P(Y|\theta^i) \\
&= \log \left( \sum_Z P(Z|Y,\theta^i) \frac{P(Y|Z,\theta)P(Z|\theta)}{P(Z|Y,\theta^i)} \right) - \log P(Y|\theta^i) \\
&\geq \sum_Z P(Z|Y,\theta^i) \log \frac{P(Y|Z,\theta)P(Z|\theta)}{p(Z|Y,\theta^i)} - \log P(Y|\theta^i) \\
&= \sum_Z P(Z|Y,\theta^i) \log \frac{P(Y|Z,\theta)P(Z|\theta)}{P(Z|Y,\theta^i)P(Y|\theta^i)}
\end{aligned}
$$
令,
$$
B(\theta,\theta^i) \hat{=} L(\theta^i) + \sum_Z P(Z|Y,\theta^i) \log \frac{P(Y|Z,\theta)P(Z|\theta)}{P(Z|Y,\theta^i)P(Y|\theta^i)} \tag{1}
$$
则有,
$$
L(\theta) \geq B(\theta,\theta^i)
$$
而且有,
$$
L(\theta^i) = B(\theta^i,\theta^i)
$$
从上面的式子得出,任何使得$B(\theta,\theta^i)$函数增大的$\theta$,也可以使$L(\theta)$增大,为了使$L(\theta)$尽可能大的增大,那么相当于求使B函数取极大值的$\theta^{i+1}$:
$$
\theta^{i+1} = \arg \max_\theta B(\theta, \theta^i) \tag{2}
$$
使用极大似然估计法求解上面(2)式:
$$
\begin{aligned}
\theta^{i+1} &= \arg \max_\theta B(\theta, \theta^i) \\
&= \arg \max_\theta L(\theta^i) + \sum_Z P(Z|Y,\theta^i) \log \frac{P(Y|Z,\theta)P(Z|\theta)}{P(Z|Y,\theta^i)P(Y|\theta^i)} \\
&= \arg \max_\theta \sum_Z P(Z|Y,\theta^i) \log \frac{P(Y|Z,\theta)P(Z|\theta)}{P(Z|Y,\theta^i)P(Y|\theta^i)}  \\
&=  \arg \max_\theta  \sum_Z P(Z|Y,\theta^i) \log {P(Y|Z,\theta)P(Z|\theta)} \\
&= \arg \max_\theta Q(\theta,\theta^i)
\end{aligned}
$$
**结论:** EM算法的一次迭代,即求Q函数及其极大化.**EM算法通过不断求解下界的极大化逼近求解对数似然函数极大化的算法**.

## 收敛证明

请参阅统计学习方法

## EM算法的推广

EM算法还可以解释为**F函数(F function)**的极大-极大算法(maximization-maximization algorithm).

### F函数的极大-极大算法

**F函数**:假设**隐变量数据Z的概率分布为$\widetilde{P}(Z)$,**定义分布$\widetilde{P}$与参数$\theta$的函数$F(\widetilde{P},\theta)$如下:
$$
F(\widetilde{P},\theta) = E_\widetilde{P} [\log P(Y,Z|\theta)] + H(\widetilde{P})
$$
称为F函数,式中$H(\widetilde{P}) = - E_{\widetilde{P}} \log \widetilde{P}(Z)$是分布$\widetilde{P}(Z)$的熵.F函数可表示成如下形式:
$$
F(\widetilde{P},\theta) = \sum_Z \widetilde{P}(Z) \log P(Y,Z|\theta) - \sum_Z \widetilde{P}(Z) \log \widetilde{P}(Z)
$$
1.对于固定的$\theta​$,存在唯一的分布$\widetilde{P}_\theta​$极大化F函数,这时:
$$
\widetilde{P}_\theta(Z) = P(Z|Y,\theta) \tag{3}
$$
并且$\widetilde{P}_\theta$随$\theta$连续变化.

构建拉格朗日式子:
$$
L = F(\widetilde{P},\theta) + \lambda(1-\sum_Z\widetilde{P}(Z))
$$
对上式的$\widetilde{P}$求偏导数:
$$
\frac{\partial L}{\partial \widetilde{P}(Z)} = \log P(Y,Z|\theta) - \log P(Z) - 1 - \lambda
$$
令偏导数为0,得出:
$$
\frac{P(Y,Z|\theta)}{\widetilde{P}_\theta(Z)} = e^{1+\lambda}
$$
在从约束条件$\sum_Z \widetilde{P}_\theta(Z) = 1$得到结论公式(3)

2.由公式(3),可得:
$$
\begin{aligned}
F(\widetilde{P},\theta) &= \sum_Z \widetilde{P}(Z) \log P(Y,Z|\theta) - \sum_Z \widetilde{P}(Z) \log \widetilde{P}(Z) \\
&= \sum_Z P(Z|Y,\theta) \log \frac{P(Y,Z|\theta)}{P(Z|Y,\theta)} \\
&= \sum_Z P(Z|Y,\theta) \log P(Y|\theta) \\
&= \log P(Y|\theta) \sum_Z P(Z|Y,\theta) \\
&= \log P(Y|\theta)
\end{aligned}
$$

### GEM算法

#### GEM算法1

直接求使Q函数极大化的$\theta$.

1. 初始化参数$\theta^0$,开始迭代
2. 第i+1次迭代,第一步:记$\theta^i$为参数$\theta$的估计值,$\widetilde{P}^i$为函数$\widetilde{P}$的估计.求$\widetilde{P}^{i+1}$使$\widetilde{P}$极大化$F(\widetilde{P},\theta^i)$
3. 第二步:求$\theta^{i+1}$使$F(\widetilde{P}^{i+1},\theta)$极大化
4. 重复2和3直到收敛

算法1中,有时求$Q(\theta,\theta^i)$的极大化很难,在算法2和算法3中改变策略,求$\theta^{i+1}$使得$Q(\theta^{i+1},\theta^i) > Q(\theta^i,\theta^i)$.

#### GEM算法2

1. 初始化参数$\theta^0$,开始迭代

2. 第i+1次迭代,第一步:记$\theta^i$为参数$\theta$的估计值,计算
   $$
   \begin{align*}
   Q(\theta,\theta^i) &= E_Z[\log P(Y,Z|\theta)|Y,\theta^i] \\
   &= \sum_Z \log P(Y,Z|\theta)P(Z|Y,\theta^i)
   \end{align*}
   $$

3. 第二步:求$\theta^{i+1}​$使
   $$
   Q(\theta^{i+1},\theta^i) > Q(\theta^i,\theta^i)
   $$

4. 重复2和3,直到收敛

当参数$\theta$的维数为d(d>=2)时,可采用一种特殊的GEM算法,它将EM算法的M步分解为d次条件极大化,**每次只改变参数向量的一个分量**,其余分量不改变.

### GEM算法3(参数维度大于1)

1. 初始化参数$\theta^0=(\theta_1^0,\theta_2^0,...,\theta_d^0)$,开始迭代

2. 第i+1次迭代,第一步:记$\theta^i = (\theta_1^i,\theta_2^i,...,\theta_d^i)$为参数$\theta=(\theta_1,\theta_2,...,\theta_d)$的估计值,计算
   $$
   \begin{align*}
   Q(\theta,\theta^i) &= E_Z[\log P(Y,Z|\theta)|Y,\theta^i] \\
   &= \sum_Z \log P(Y,Z|\theta)P(Z|Y,\theta^i)
   \end{align*}
   $$

3. 第二步:进行d次条件极大化:

   首先,在$\theta_2^i,...,\theta_k^i$保持不变的条件下求使Q函数达到极大的$\theta_1^{i+1}$.

   然后,在$\theta_1=\theta_1^{i+1},\theta_j=\theta_j^i, j=3,4,...,k$的条件下求使Q函数达到极大的$\theta_2^{i+1}$

   如此继续,经过d次条件极大化,得到$\theta^{i+1} = (\theta_1^{i+1},\theta_2^{i+1},...,\theta_d^{i+1})$使得
   $$
   Q(\theta^{i+1},\theta^i) > Q(\theta^i,\theta^i)
   $$

4. 重复2和3,直到收敛.