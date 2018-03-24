# AdaBoost算法

AdaBoost算法就是使用Boosting集成方法来把多个决策树组合起来生成一个新的，性能更好的模型。Boosting是一族可将弱学习器提升为强学习器的算法。这族算法的工作机制类似:先从初始训练集训练出一个基学习器，再根据基学习器的表现对训练样本分布进行调整，使得先前基学习器做错的训练样本在后续受到更多关注，然后基于调整后的样本分布来训练下一个基学习器；如此重复进行，直至基学习器数目达到事先指定的值T，最终将这T个基学习器进行加权结合。

## 算法推演

输入:训练数据集$T=\{(x_1,y_1),(x_2,y_2),\dots,(x_n,y_n)\}$,其中$x_i \in \mathcal{X} \subseteq R^n$，$y_i \in \mathcal{Y}={-1,+1}$；弱学习算法；

输出：最终分类器G(x)

1. 初始化训练数据的权值分布

$$
D_1 = (w_{11},\dots,w_{1i},\dots,w_{1n}),w_{1i}=\frac{1}{n},i=1,2,\dots,n
$$

假设训练数据集具有均匀的权值分布，即每个训练样本在基本分类器的学习中作用相同。

2. 对$m=1,2,\dots,M$，for循环，迭代M次，生成M个基础分类器。

   1. 使用具有权值分布$D_m$的训练数据集学习，得到基本分类器
      $$
      G_m(x):\mathcal{X} \rightarrow \{-1,+1\}
      $$

   2. 计算$G_m(x)$在训练数据集上的分类误差率
      $$
      e_m = \sum_{i=1}^{n}P(G_m(x_i) \neq y_i) = \sum_{i=1}^{n}w_{mi}I(G_m(x_i) \neq y_i) = \sum_{G_m(x_i)\neq y_i}w_{mi}
      $$
      这里，$w_{mi}$表示第m轮中第i个实例的权值，$\sum_{i=1}^{n}w_{mi}=1$。这表明，$G_m(x)$在加权的训练数据集上的分类误差率是被$G_m(x)$误分类样本的权值之和，由此可以看出数据权值分布$D_m$与基本分类器$G_m(x)$的分类误差率的关系。

      ​

   3. 计算$G_m(x)$的系数
      $$
      \alpha_m = \frac{1}{2}\log \frac{1-e_m}{e_m}
      $$
      $\alpha_m$表示$G_m(x)$在最终分类器中的重要性。由式子可知，当$e_m \leq \frac{1}{2}$时，$\alpha_m \geq 0$，并且$\alpha_m$随着$e_m$的减小而增大，所以分类误差率越小的基本分类器在最终分类器中的作用越大。

      ​

   4. 更新训练数据集的权值分布
      $$
      D_{m+1} = (w_{m+1,1},\dots,w_{m+1,i},\dots,w_{m+1,n}) \\
      w_{m+1,i} = \frac{w_{mi}}{Z_m}\exp(-\alpha_my_iG_m(x_i)),i=1,2,\dots,n
      $$
      也可以把权值分布更新规则改成:
      $$
      \begin{equation}
      w_{m+1,i}=\left\{
      \begin{aligned}
      \frac{w_{mi}}{Z_m}e^{-\alpha_m} & & G_m(x_i)=y_i \\
      \frac{w_{mi}}{Z_m}e^{\alpha_m} & & G_m(x_i) \neq y_i
      \end{aligned}
      \right.
      \end{equation}
      $$
      由此可知，被基本分类器$G_m(x)$误分类样本的权值得以扩大，而被正确分类样本的权值却得以缩小。两种情况进行比较，由系数$\alpha_m$的公式可得，误分类样本的权值被放大$e^{2\alpha_m} = \frac{1-e_m}{e_m}$倍。因此，误分类样本在下一轮学习中起更大的作用。

      这里，$Z_m$是规范化因子
      $$
      Z_m = \sum_{i=1}^{n}w_{mi}\exp(-\alpha_my_iG_m(x_i))
      $$
      它使$D_{m+1}$成为一个概率分布。

3. 构建基本分类器的线性组合
   $$
   f(x) = \sum_{m=1}^{M}\alpha_mG_m(x)
   $$
   得到最终分类器
   $$
   G(x) = \rm sign \left( \sum_{m=1}^{M}\alpha_mG_m(x) \right)
   $$
   线性组合$f(x)$实现M个基本分类器的加权表决。系数$\alpha_m$表示了基本分类器$G_m(x)$的重要性，这里，**所有$\alpha_m$之和并不为1**。$f(x)$的符号决定实例x的分类，它的绝对值表示分类的确信度。利用基本分类器的线性组合构建最终分类器是AdaBoost的另一特点。



##训练误差分析

AdaBoost最基本的性质是它能在学习过程中不断减少训练误差，即在训练数据集熵的分类误差率。

**AdaBoost的训练误差界**:AdaBoost算法最终分类器的训练误差界为：
$$
\frac{1}{N}\sum_{i=1}^{N}I(G(x_i)\neq y_i) \leq \frac{1}{N}\sum_i \exp(-y_if(x_i)) = \prod_mZ_m
$$
这里，$G(x),f(x)$和$Z_m$。

**证明:** 当$G(x_i) \neq y_i$时，$y_if(x_i) < 0$，因而$\exp(-y_i f(x_i)) \geq 1$.由此直接推导出前半部分。后半部分的推导要用到$Z_m$的定义和$w_{mi}$的变形：
$$
w_{mi}\exp(-\alpha_my_iG_m(x_i)) = Z_mw_{m+1,i}
$$
现推导如下：
$$
\begin{align*}
\frac{1}{N}\sum_{i}\exp(-y_if(x_i)) &= \frac{1}{N}\sum_{i}\left(-\sum_{m=1}{M}\alpha_my_iG_m(x_i)\right) \\
&= \sum_{i}w_{1i}\prod_{m=1}^{M}\exp(-\alpha_m y_i G_m(x_i)) \\
&= Z_1\sum_{i}w_{2i}\prod_{m=2}^{M}\exp(-\alpha_m y_i G_m(x_i)) \\
&= Z_1Z_2\sum_{i}w_{3i}\prod_{m=3}^{M}\exp(-\alpha_m y_i G_m(x_i)) \\
&= \dots \\
&= Z_1Z_2\dots Z_{M-1} \sum_{i}w_{Mi}\exp (-\alpha_M y_i G_M(x_i)) \\
&= \prod_{m=1}^{M}Z_m
\end{align*}
$$

这一定理说明，可以在每一轮选取适当的$G_m$使得$Z_m$最小，从而使训练误差下降最快。对二类分类问题，有如下结果：
$$
\prod_{m=1}^{M}Z_m = \prod_{m=1}^{M}[2\sqrt{e_m(1-e_m)}] = \prod_{m=1}^{M}\sqrt{(1-4\gamma_m^2)} \leq \exp(-2\sum_{m=1}^{M}\gamma_m^2)
$$
这里，$\gamma_m = \frac{1}{2}-e_m$。

**证明:**由$Z_m$的定义式以及分类误差率$e_m$得：
$$
\begin{align*}
Z_m &= \sum_{i=1}^{N}w_{mi}\exp(-\alpha_m y_i G_m(x_i)) \\
&= \sum_{y_i=G_m(x_i)} w_{mi}e^{-\alpha_m} + \sum_{y_i \neq G_m(x_i)}w_{mi}e^{\alpha_m} \\
&= (1-e_m)e^{-\alpha_m} + e_m e^{\alpha_m} \\
&= 2\sqrt{e_m(1-e_m)} = \sqrt{1-4\gamma_m^2}
\end{align*}
$$
至于不等式
$$
\prod_{m=1}^{M}\sqrt{(1-4\gamma_m^2)} \leq \exp(-2\sum_{m=1}^{M}\gamma_m^2)
$$
则可先由$e^x$和$\sqrt{1-x}$在点$x=0$的泰勒展开式推出不等式$\sqrt{(1-4\gamma_m^2)} \leq \exp(-2\gamma_m^2)$，进而得到以下推论：

如果存在$\gamma > 0$，对所有m有$\gamma_m \geq \gamma$，则：
$$
\frac{1}{N} \sum_{i=1}^{N}I(G(x_i) \neq y_i) \leq \exp(-2M\gamma^2)
$$
这表明在此条件下，AdaBoost的训练误差是以指数速率下降的。

