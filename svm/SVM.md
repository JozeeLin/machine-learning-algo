#SVM-支持向量机

##引言

通过Logistic Regression得知概率密度函数$p(y=1|X;\theta) = h_\theta(X)=g(\theta^TX)$，表示给定X和$\theta$ ，y=1的概率。这里我们设置阈值为0.5，即$h_\theta(X) \geq 0.5$时，把y预测为1，也就是当$\theta^TX \geq 0$时，把y预测为1。同样的，$\theta^TX$越大，$h_\theta(X)$的值也会越大，预测y为1的自信也就越大。 

所以我们得到这样的推断，如果$\theta^TX \gg 0$，y=1，相似的，如果$\theta^TX \ll 0$，y=0。给定一个样本，当y=1的时候，我们可以找到一个$\hat{\theta}$使得$\theta^TX \gg 0$，当y=0的时候，可以找到一个$\hat{\theta}$使得$\theta^TX \ll 0$。稍后会使用函数间隔的概念来形式化上面的问题。

![svm-1](image/svm-1.png)

上图显示(这里假定X表示正样本1，O表示负样本0)，直线表示**分类超平面**$\theta^TX=0$，是分类器的决策边界。正如我们所见，A点距离决策边界很远，所以分类器可以很自信的把它划分为y=1，而C点距离决策边界很近，当决策边界稍微变动就很有可能使C点划分为y=0，B点介于两者之间，所属的情况更为广泛(处于这个位置的样本数量是最多，最集中的)。

所以，我们的目标是找到一个决策边界，是的所有的样本都被正确的分类，而且更有自信的分类。稍后我们会使用几何间隔的概念来形式化上面的问题。

## 假设函数

假设函数
$$
h_{w,b}(X) = g(W^TX+b)
$$
在这里，当$z \geq 0$ ，$g(z) = 1$，$z < 0$，$g(z)=-1$。

## 函数间隔和几何间隔

### 函数间隔

给定一个样本$(X^i,y^i)$ ，我们定义参数为$(W, b)$的函数间隔为:
$$
\hat{\gamma}^i = y^i(W^TX+b)
$$
根据前面的推断，当$y^i=1$时，$\hat{\gamma}^i \gg 0$会很大，同理地，当$y^i = -1$时，$\hat{\gamma}^i \gg 0$。所以，当样本$(X^i,y^i)$是正样本时，我们需要使$W^TX+b$是一个很大的正整数，相反地，如果为负样本，那么我们需要使$W^TX+b$是一个绝对值很大的负整数。所以，当$\hat{\gamma}^i > 0$时，样本被正确分类，同时函数间隔越大，表示该样本被正确分类的信心越大。

注意：函数间隔存在一个不足之处，令$W=2W,b=2b$时，并不会改变假设函数$h_\theta(X)$的结果，但是会使函数间隔缩放2倍。这也说明了，函数间隔可以随意缩放，而不会产生任何有意义的变化。

**更一般的，定义函数间隔**：

给定一个训练样本$S={(X^i,y^i),i=1,2,\dots,m}$，令$\hat{\gamma} = \min_\limits{i=1,\dots,m} \hat{\gamma}^i$。

### 几何间隔

![svm-2](image/svm-2.png)

分类超平面为$W^TX+b = 0$，对应的向量$W$为分类超平面的法向量。A点表示样本$(X^i, y^i=1)$的$X^i$特征，点A到决策边界的距离表示为$\gamma^i$(线段AB的长度)，点B就可以表示为$X^i-\gamma^i.\frac{W}{\|W\|}$ ，其中$\frac{W}{\|W\|}$表示分类超平面的单位法向量。因为B点在分类超平面上，所以：
$$
W^T(X^i-\gamma^i.\frac{W}{\|W\|})+b = 0
$$
解出，$\gamma^i = \frac{W^TX^i+b}{\|W\|}=(\frac{W}{\|W\|})^TX^i+\frac{b}{\|W\|}$。

更一般的，把几何间隔表示为：
$$
\gamma^i = y^i((\frac{W}{\|W\|})^TX^i+\frac{b}{\|W\|})
$$
**注意:**当$\|W\|=1$时，几何间隔等于函数间隔。同时，无论如何缩放W和b，都不会改变几何间隔。因此，我们就可以对几何间隔增加的参数增加任意的缩放约束条件，也不会改变几何函数的求解。

**最后，修正几何间隔为：**

给定一个训练样本$S={(X^i,y^i),i=1,2,\dots,m}$，令$\gamma = \min_\limits{i=1,\dots,m} \gamma^i$。

## 最优间隔分类器

> 根据前面的讨论，我们可以通过最大化几何间隔得到一个最优的决策边界使得测试样本被更好的正确分类。

假设给定训练样本是线性可分的，也就是存在一些分类超平面可以把正负样本正确分开。现在我们的问题是，如何找到一个分类超平面使得几何间隔最大化，**目标函数为:**
$$
\max_{\gamma,W,b} \gamma \\
S.T. \ y^i(W^TX^i+b) \geq \gamma \ ，i=1,\dots,m \\
\|W\|=1
$$
约束条件表示所有的样本的几何间隔都大或等于$\gamma$，同时保证$\|W\|=1$使得几何间隔和函数间隔保持一致。

但是，约束函数$\|W\|=1$不是凸函数，无法使用凸优化来对上面的函数求解。**目标函数更正为:**
$$
\max_{\hat{\gamma},W,b} \frac{\hat{\gamma}}{\|W\|} \\
S.T. \ y^i(W^TX^i+b) \geq \hat{\gamma} \ ，i=1,\dots,m
$$
虽然约束条件都是凸集，但是目标函数却变成了非凸函数。但是由于上面我们可以知道随意缩放W,b也不会改变几何间隔的值，所以这里令$\hat{\gamma} = 1$。

**那么目标函数更正为:**
$$
\max_{W,b} \frac{1}{\|W\|} \\
S.T. \ y^i(W^TX^i+b) \geq 1 \ ，i=1,\dots,m
$$
由于最大化$\frac{1}{\|W\|}$，相当于最小化$\|W\|^2$，所以**目标函数更正为:**
$$
\min_{W,b} \frac{1}{2}\|W\|^2 \\
S.T. \ y^i(W^TX^i+b) \geq 1 \ , i=1,\dots,m
$$
上面的问题属于二次规划(QP)问题带有线性约束，也是凸优化问题。至此，我们就得到了最优间隔分类器。

## 拉格朗日对偶

> 有约束最优化问题。

###原始形式

考虑如下问题：
$$
\min_W f(W) \\
S.T. \ h_i(W)=0 \ , i=1,\dots,l
$$
使用[拉格朗日乘数法](https://zh.wikipedia.org/wiki/%E6%8B%89%E6%A0%BC%E6%9C%97%E6%97%A5%E4%B9%98%E6%95%B0)来求解上述问题：
$$
\mathcal{L} (W,\beta) = f(W)+\sum_{i=1}^{l}\beta_ih_i(W)
$$
这里，$\beta_i$称为拉格朗日乘数。通过设置拉格朗日函数的对$W,\beta$的偏导为0，来求解$W,\beta$:
$$
\frac{\partial{\mathcal{L}}}{\partial{W_i}} = 0 \ ; \frac{\partial{\mathcal{L}}}{\partial{\beta_i}} = 0
$$
我们是上面的拉格朗日函数更加一般化，使之同时包含不等式约束和等式约束。下面问题我们称为**最优化问题的原始形式**：
$$
\min_w f(W)  \\
S.T. g_i(W) \leq 0 \ , i=1,\dots,k \\
h_i(W) = 0，i=1,\dots,l
$$
为了解决上述问题，我们定义了**拉格朗日的一般形式**：
$$
\mathcal{L}(W,\alpha,\beta) = f(W)+\sum_{i=1}^{k}\alpha_i g_i(W)+\sum_{i=1}^{l}\beta_ih_i(W)
$$
$\alpha_i\ , \beta_i$为拉格朗日乘数。令：
$$
\theta_{\mathcal{P}}(W) = \max_{\alpha,\beta:\alpha_i\geq0} \mathcal{L}(W,\alpha,\beta)
$$
下标$\mathcal{P}$表示"原始形式"。如果给定的W不满足约束条件，那么导致$\theta_{\mathcal{P}}(W) = \infty$：
$$
\begin{equation}
\theta_{\mathcal{P}}(W)=\left\{
\begin{aligned}
f(W) & & 若W 满足约束条件 \\
\infty & & 其他
\end{aligned}
\right.
\end{equation}
$$
所以，当W满足约束条件的时候，$\theta_{\mathcal{P}}$等于$f(W)$，所以上述的**最优化问题的原始形式**可以变成以下形式：
$$
\min_{W}\theta_{\mathcal{P}}(W) = \min_{W} \max_{\alpha,\beta;\alpha_i\geq0} \mathcal{L}(W,\alpha,\beta)
$$
令$p^* = \min_{W} \theta_{\mathcal{P}}(W)$，$p^*$也叫做原始问题的值。

### 对偶形式

定义:
$$
\theta_{\mathcal{D}}(\alpha, \beta) = \min_{W} \mathcal{L}(W,\alpha,\beta)
$$
假设$\widetilde{W}$满足约束条件，则:
$$
\min_{W} \mathcal{L}(W,\alpha,\beta) \leq \mathcal{L}(\widetilde{W},\alpha,\beta) \\
\mathcal{L}(\widetilde{W},\alpha,\beta) = f(\widetilde{W})+\sum_{i=1}^{k}\alpha_i g_i(\widetilde{W})+\sum_{i=1}^{l}\beta_ih_i(\widetilde{W}) \leq f_0(\widetilde{W})
$$
所以：
$$
\min_{W} \mathcal{L}(W,\alpha,\beta) \leq p^*
$$
下标$\mathcal{D}$表示“对偶”。根据上面的原始形式，很快得到**最优化问题的对偶形式：**
$$
\max_{\alpha,\beta;\alpha_i \geq 0} \theta_{\mathcal{D}}(\alpha, \beta) = \max_{\alpha,\beta;\alpha_i \geq 0} \min_{W} \mathcal{L}(W,\alpha,\beta)
$$
令$d^* = \max_{\alpha,\beta;\alpha_i \geq =0} \theta_{\mathcal{D}}(W)$，$d^*$也叫对偶问题的值。

### 结论

$$
d^* = \max_{\alpha,\beta;\alpha_i \geq 0} \min_{W} \mathcal{L}(W,\alpha,\beta) \leq \min_{W} \max_{\alpha,\beta;\alpha_i \geq 0} \mathcal{L}(W,\alpha,\beta) = p^*
$$

对于凸优化问题强对偶条件成立时，上面的不等式会变成等式$d^*=p^*$。所以，我们可以**通过求解对偶问题来解决原始问题**。强对偶条件成立，表示一定存在$W^*,\alpha^*,\beta^*$，其中$W^*$是原始问题的解，$\alpha^*,\beta^*$为对偶问题的解。也就是$d^*=p^*=\mathcal{L}(W^*,\alpha^*,\beta^*)$。

**Karush-Kuhn-Tucker(KKT)条件:**
$$
\frac{\partial}{\partial W_i} \mathcal{L}(W^*,\alpha^*,\beta^*) = 0 \ , i=1,\dots,n \tag{1}
$$

$$
\frac{\partial}{\partial \beta_i} \mathcal{L}(W^*,\alpha^*,\beta^*) = 0 \ , i=1,\dots,l \tag{2}
$$

$$
\alpha_i^* g_i(W^*) = 0 \ , i=1,\dots,k \tag{3}
$$

$$
g_i(W^*) \leq 0 \ , i=1,\dots,k \tag{4}
$$

$$
\alpha_i^* \geq 0 \ , i=1,\dots,k \tag{5}
$$

对于公式(3)叫做对偶互补条件，它表示当$\alpha_i^* > 0​$，$g_i(W^*) = 0​$，当$\alpha_i^* = 0​$，$g_i(W^*) < 0​$。

所有满足KKT条件的$W^*,\alpha^*,\beta^*$，都是原始问题和对偶问题的解。

#### 凸优化问题的标准形式

- 凸优化问题

$$
\min f(X) \\
S.T. g_i(X) \leq 0 \ , i=1,\dots,k \\
h_i(X) = 0 \ , i=1,\dots,l
$$

- 则有$f(X)$是凸函数，可行域是凸集
  - 目标函数是凸函数
  - 不等式约束函数必须是凸的
  - 等式约束函数必须是仿射的
- 最优值(目标函数在可行域上的最小值)
  - $p^* = +\infty$ 不可行(可行域是空集)
  - $p^* = -\infty$ unbounded below(存在可行点使得$f(X)\rightarrow -\infty$)
  - $f(X^*)=p^*$ 有解



## 最优间隔分类器(硬间隔)

下面我们给出最优间隔分类器的原始形式：
$$
\min_{\gamma,W,b} \frac{1}{2}\|W\|^2 \\
S.T. y^i(W^TX^i+b) \geq 1 \ , i=1,\dots,m
$$
对应的，我们可以把约束条件表示成如下形式：
$$
g_i(W) = -y^i(W^TX^i+b)+1 \leq 0
$$
根据KKT的互补条件，$\alpha_i > 0$，当且仅当函数间隔等于1($y^i(W^TX^i+b)=1$)。

![svm-3](image/svm-3.png)

实线表示最优间隔分类器的分类超平面。最小几何间隔对应的3个样本点，分别是1个负样本和2个正样本，它们都分别在平行于分类超平面的两条虚线上。因此，只有存在3个满足$\alpha_i > 0$。这三个样本点叫做**支持向量**。事实上，支持向量的个数越小于训练样本数越好。

把最优间隔分类器问题写成**拉格朗日函数形式**：
$$
\mathcal{L}(W,b,\alpha) = \frac{1}{2}\|W\|^2 - \sum_{i=1}^{m}\alpha_i[y^i(W^TX^i+b)-1] \tag{6}
$$
接着，我们找出**拉格朗日函数的对偶形式**:
$$
d^* = \max_{\alpha\geq0}(\min_{W,b}\mathcal{L}(W,b,\alpha))
$$
令$\mathcal{L}(W,b,\alpha)$对W和b的偏导为0，则有：
$$
\nabla_W \mathcal{L}(W,b,\alpha) = W-\sum_{i=1}^{m}\alpha_iy^iX^i = 0
$$
得出：
$$
W = \sum_{i=1}^{m}\alpha_iy^iX^i \tag{7}
$$

$$
\frac{\partial}{\partial b} \mathcal{L}(W,b,\alpha) = \sum_{i=1}^{m}\alpha_iy^i = 0 \tag{8}
$$

把公式(7)代入公式(6):
$$
\mathcal{L}(W,b,\alpha) = \sum_{i=1}^{m}\alpha_i - \frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_i\alpha_jy^iy^j(X^i)^TX^j - b\sum_{i=1}^{m}\alpha_iy^i \tag{9}
$$
把公式(8)代入公式(9):
$$
\mathcal{L}(W,b,\alpha) = \sum_{i=1}^{m}\alpha_i - \frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_i\alpha_jy^iy^j(X^i)^TX^j \tag{10}
$$
至此，对偶形式变为:
$$
\max_{\alpha} W(\alpha) = \sum_{i=1}^{m}\alpha_i - \frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_i\alpha_jy^iy^j((X^i)^TX^i) \\
S.T. \ \alpha_i \geq 0 \ , i=1,\dots,m \\
\sum_{i=1}^{m}\alpha_iy^i = 0
$$


假定通过对偶形式求解出$\alpha^*$，根据KKT有：

- $\alpha_i^* (y^i((W^*)^TX^i+b^*) - 1)= 0$
- $\alpha_i((W^*)^TX^i+b^*) \geq 1$
- $\alpha_i^* \geq 0$

把求出来的$\alpha_i^*$代入公式(7),可得$W^* = \sum_\limits{i=1}^{m}\alpha_i^*y^iX^i$，同时代入公式(8)，可得$\sum_\limits{i=1}^{m}\alpha_i^*y_i = 0$。

根据**支持向量**的定义，若$\alpha_j^* = 0$，对$W^*$无贡献，若$\alpha_j^* > 0$，则$y^j((W^*)^TX^j+b^*) = 1$。

所以$W^*$代入$y^i((W^*)^TX^i+b^*) = 1$，则$b^* = y_j - \sum_\limits{i=1}^{m}\alpha_i^*y^i\left<X_i^TX_j\right>$。(注意:$j$表示支持向量的样本下标)。

更一般的，根据**支持向量**的定义，支持向量所在的虚线，可以分别表示为:

- 当y=1时，$(W^*)^TX+b^*+\widetilde{b}$
- 当y=-1时，$(W^*)^TX+b^*-\widetilde{b}$

此处，$\widetilde{b}$是最小几何间隔，也就是虚线到分类超平面之间的距离。所以，$b^*$还可以是以下的表达形式:
$$
b^* = - \frac{\max_\limits{i=1,y^i=-1}^{m}(W^*)^TX^i+\min_\limits{j=1,y^j=1}^{m}(W^*)^TX^j}{2}
$$
根据公式(7)，可得：
$$
W^TX+b = (\sum_\limits{i=1}^{m}\alpha_iy^iX^i)^TX+b \tag{11} = \sum_\limits{i=1}^{m}\alpha_iy^i\left< X^i,X \right>+b
$$
所以，只要找到了支持向量对应$\alpha$就可以通过计算样本特征与支撑向量间的內积来进行预测。

参考:

- [cs229-note3](http://cs229.stanford.edu/notes/cs229-notes3.pdf)
- [cs229-video](https://open.163.com/movie/2008/1/C/6/M6SGF6VB4_M6SGJVMC6.html)

## 核函数

> 在多项式回归里举过一个例子房屋面积和售价的关系，使用线性回归的方式拟合非线性函数。即令$h_\theta(x) = \theta_0+\theta_1x+\theta_2x^2+\theta_3x^3$。为了应用线性回归求解这个问题，我们使用$x_1,x_2,x_3$来分别表示$x_1=x,x_2=x^2,x_3=x^3$。所以假设函数修正为$h_\theta(X) = \theta^TX=\theta_0+\theta_1x_1+\theta_2x_2+\theta_3x_3$。但是这里我们使用映射的思想来描述以上三次多项式的构建，就是把输入房屋面积$x$，映射成$[x,x^2,x^3]$更高维度的特征向量。使用映射得到的特征向量来构建模型求解问题。构建关系表示为:
> $$
> \phi(x) = [x,x^2,x^3]^T
> $$
>

回顾前面的最优间隔分类器最终的净输入函数$W^TX+b$变成了求样本特征向量之间的内积了即$\left<X,Z\right>$，根据前面的多项式的映射升维思想，内积应该变成$\left<\phi(X),\phi(Z)\right>$，特别的引出了kernel函数的表达式：
$$
K(X,Z) = \phi(X)^T\phi(Z)
$$
举例说明即使kernel函数在处理高维度数据时会大大提高计算效率：

令$X,Z \in R^n$，$K(X,Z) = (X^TZ)^2$:
$$
K(X,Z) = (\sum_{i=1}^{n}x_iz_i)(\sum_{j=1}^{n}x_jz_j) \\
= \sum_{i=1}^{n}\sum_{j=1}^{n}x_ix_jz_iz_j \\
= \sum_{i,j=1}^{n}(x_ix_j)(z_iz_j)
$$
使用前面的定义$K(X,Z)=\phi(X)\phi(Z)$，假设$X=[x_1,x_2,x_3]^T$，那么$\phi(X) = [x_1x_1,x_1x_2,x_1x_3,x_2x_1,x_2x_2,x_2x_3,x_3x_1,x_3x_2,x_3x_3]^T$。注意，这里计算$\phi(X)$的时间复杂度为$O(N^2)$，而$K(X,Z)$的时间复杂度为$O(N)$。

更一般地，$K(X,Z)=(X^TZ+c)^2=\sum_\limits{i,j=1}^{n}(x_ix_j)(z_iz_j)+\sum_\limits{i=1}^{n}(\sqrt{2c}x_i)(\sqrt{2c}z_i)+c^2$。而对应的$\phi(X)$为：

$\phi(X) = [x_1x_1,x_1x_2,x_1x_3,x_2x_1,x_2x_2,x_2x_3,x_3x_1,x_3x_2,x_3x_3,\sqrt{2c}x_1,\sqrt{2c}x_2,\sqrt{2c}x_3,c]^T$。这里参数c是用来控制单项式和二项式之间的相关权重的。更进一步推广，$K(X,Z)=(X^TZ+c)^d$，所对应$\phi(X)$的特征维度为 ${\displaystyle {\tbinom {n+d}{d}}}$(二项式定理)。

从另外的角度来思考核函数。如果$\phi(X)$和$\phi(Z)$很相近，那么$K(X,Z)$的值会很大，反之，如果$\phi(X)$和$\phi(Z)$的相关性很小，那么$K(X,Z)$的值也会很小。所以，我们也可以把核函数作为衡量两个向量之间的相似度。

所以，当你遇到需要衡量两个向量之间的相似度时，自然而然的就想到使用核函数来解决，比如高斯核函数：
$$
K(X,Z) = \exp(-\frac{\|X-Z\|^2}{2\sigma^2})
$$
当X和Z很相似的时候，核函数值近似为1，而两者几乎没有相似度时，核函数值近似为0.

### 核函数的有效性

**问题:**如何确定核函数是有效的，也就是是否存在$\phi$使得$K(X,Z)=\phi(X)^T\phi(Z)$？

**证明：**假设有m个训练样本${X^1,X^2,\dots,X^m}$，定义一个$m \times m$的核矩阵K，$K_{ij}=K(X^i,X^j)$。如果K是一个有效核函数，那么一定满足:$K_{ij}=K(X^i,X^j)=\phi(X^i)\phi(X^j)=\phi(X^j)\phi(X^i)=K_{ji}$。也就是核矩阵为对称矩阵。现使用$\phi_k(X)$表示向量$\phi(X)$的第k个元素，对任意向量Z都有:
$$
Z^TKZ = \sum_{i}\sum_{j}z_iK_{ij}z_j \\
= \sum_{i}\sum_{j}z_i\phi(X^i)^T\phi(X^j)z_j \\
= \sum_{i}\sum_{j}z_i(\sum_{k}\phi_k(X^i)\phi_{k}(X^j))z_j \\
= \sum_{k}\sum_{i}\sum_{j}z_i\phi_k(X^i)\phi_k(X^j)z_j \\
= \sum_{k}(\sum_{i}z_i\phi_k(X^i))^2 \geq 0
$$
最后一步跟前面计算$K(X,Z)=(X^TZ)^2$类似。从公式可以看出，如果K是一个有效的核函数，那么在训练集上得到的核函数矩阵应该是半正定的。这也是Mercer定理:

K是有效核函数 <=> 核函数矩阵K是对称半正定的。

从Mercer定理可知证明K是否为有效核函数，不需要找到映射函数$\phi$，只需要在训练集上求出核矩阵K,并判断该矩阵是否是半正定的即可。

注:核函数不仅用在SVM上，但凡算法中出现$\left<X,Z\right>$，都可以使用$K(X,Z)$来替换。

## 软间隔

之前我们讨论过，当样本存在线性不可分的情况时，我们可以通过核函数把特征映射到更高维度的空间中，这样就可以把样本分开了，但是并非对所有的线性不可分情况都有效。所以，如果使用核函数方法依然无法是样本集完全可分，那么我们可以通过对模型进行调整，尽可能的找到那个最好的决策边界。

![svm-4](image/svm-4.png)

上图显示，左图得到一个最优的决策边界，但是右图中由于存在一个离群点导致决策边界发生改变，使得函数间隔变得非常小。由此可见，之前的**最优间隔分类器的模型对噪声非常敏感**。

为了解决这些问题，我们需要对之前的目标函数进行修正，得到新的模型(也成软间隔):
$$
\min_{\gamma,W,b} \frac{1}{2}\|W\|^2 +C\sum_{i=1}^{m}\xi_i \\
S.T. \ y^i(W^TX^i+b) \geq 1-\xi_i \ ， i=1 ,\dots,m \\
\xi_i \geq 0 \ ，i=1,\dots,m.
$$
引入非负参数$\xi_i$(松弛变量)，也就是允许一些样本点被错误分类。而放松限制条件后，我们需要重新调整目标函数，以对离群点进行处罚，目标函数后面加上了$C\sum_{i=1}^{m}\xi_i $表示离群点越多(被错误分类的点越多)，目标函数值越大。这里的C表示离群点的权重，C越大，表示对离群点的惩罚越大，降低离群点对模型的影响。

相应地，拉格朗日函数修正如下:
$$
\mathcal{L}(W,b,\xi,\alpha,\beta) = \frac{1}{2}\|W\|^2+C\sum_{i=1}^{m}\xi_i - \sum_{i=1}^{m}\alpha_i[y^i(X^TW+b)-1+\xi_i]-\sum_{i=1}^{m}\beta_i\xi_i
$$
对偶形式修正如下:
$$
\max_\alpha W(\alpha) = \sum_{i=1}^{m}\alpha_i - \frac{1}{2}\sum_{i,j=1}^{m}y^iy^j\alpha_i\alpha_j\left<X^i,X^j\right> \\
S.T. 0 \leq \alpha_i \leq C \ ，i=1,\dots,m \\
\sum_{i=1}^{m}\alpha_iy^i=0
$$
此时，可以看到参数$\xi_i$没有了，唯一的不同之处在于$\alpha_i$多了一个限制条件即小于C。应用KKT条件，可得:
$$
\alpha_i = 0 \Rightarrow y^i(W^TX^i+b) \geq 1 \tag{1}
$$

$$
\alpha_i=C \Rightarrow y^i(W^TX^i+b) \leq 1 \tag{2}
$$

$$
0 < \alpha_i < C \Rightarrow y^i(W^TX^i+b) = 1 \tag{3}
$$

通过上线的式子可以得知，非支持向量正确分类样本系数为0，表示没有惩罚。对于错误分类的样本系数为C，增加了惩罚项。对于支持向量的样本点，系数为(0,C)之间，也就是说，支持向量对应的样本点也有可能是被错误分类的点。所以存在惩罚项。

## SMO算法

### 坐标上升法

坐标上升法求解对偶形式的目标函数，过程为，固定除了$\alpha_i$以外的所有$\alpha$,找到使得$W(\alpha)$取最大值的$\alpha_i$，然后更新$\alpha_i$，逐步使用同样的方式按照顺序更新$\alpha_1$,$\dots$,$\alpha_m$，直到$W(\alpha)$收敛。

![svm-5](image/svm-5.png)

### SMO

> SMO是二次规划优化算法，特别是针对线性SVM和数据稀疏时性能很好。

前面的坐标上升法是无约束目标函数的求解方式，但是我们的对偶形式目标函数存在约束$\sum_\limits{i=1}^{m}\alpha_iy^i = 0$,$0 \leq \alpha_i \leq C$。因此，我们需要对坐标上升法进行改进，选中两个参数$\alpha_i,\alpha_j$来同时更新，而剩余的参数固定。根据约束条件，可得:
$$
\alpha_1y^1+\alpha_2y^2 = -\sum_{i=3}^{m}\alpha_iy^i
$$
由于剩余的参数是固定的，所以我们可以把等式右边表示为$\zeta$。即：
$$
\alpha_1y^1+\alpha_2y^2 = \zeta
$$
然后，同时满足另外的约束条件，即可得出下图：

![svm-6](image/svm-6.png)

上图所示$L \leq \alpha_2 \leq H$。同时得到$\alpha_1 = (\zeta-\alpha_2y^2)y^1$，然后得到$W(\alpha) = W((\zeta-\alpha_2y^2)y^1,\alpha_2,\dots,\alpha_m)$。$W$函数就变成了关于$\alpha_2$的函数，通过求导，使导数为0，求得$\alpha_2$，然后更新$\alpha_1,\alpha_2$。求得的$\alpha_2$的结果必须满足$L \leq \alpha_2 \leq H$。可以通过以下公式修正求导得到的$\alpha_2$。
$$
\begin{equation}
\alpha_2=\left\{
\begin{aligned}
H & & 若\alpha_2 > H \\
\alpha_2 & & 若 L \leq \alpha_2 \leq H \\
L & & 若\alpha_2 < L
\end{aligned}
\right.
\end{equation}
$$




