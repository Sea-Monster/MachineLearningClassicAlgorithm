# kNN算法
- 属于监督学习
- 非参数学习
- 是解决分类问题的算法，天然可解决多分类问题
- kNN没有模型，可以说是一个（也许也是唯一一个）不需要训练过程的算法
    - 为了和其他算法统一，可以认为训练数据集就是模型本身

## 本质
两个（或几个）样本如果足够相似，那么它们就有极高的概率属于同一个类别。所谓“相似”，就是样本就特征空间中的距离相近。
## 优点
- 思想极其简单
    - 可以解释机器学习算法使用过程中的很多细节问题
    - 更完整的刻画机器学习应用的流程
- 应用数学知识少（近乎为零）
- 效果好
- 天然适合解决多分类问题，同时也适合解决回归问题

## 缺点
- 最大的缺点：效率低下
如果训练集有m个样本，n个特征，则预测每一个新的数据，需要O(m*n)
    - 优化，使用树结构：KD-Tree， Ball-Tree
    - 即便如此，依然效率低下
- 高度数据相关，而且对outlier更敏感
- 预测结果不具有可解释性
只知道属于哪个类别，但是无法解释为什么属于某个类别
- 维数灾难
    - 随着维度的增加，“看似相似”的两个点之间的距离越来越大
    - 解决方法：降维，例如PCA

## kNN的过程
### 计算特征空间中的距离
#### 欧拉距离（最为常见）
- 平面距离：
<a href="https://www.codecogs.com/eqnedit.php?latex=\sqrt{(x^{(a)}-x^{(b)})^2&space;&plus;&space;(y^{(a)}-y^{(b)})^2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sqrt{(x^{(a)}-x^{(b)})^2&space;&plus;&space;(y^{(a)}-y^{(b)})^2}" title="\sqrt{(x^{(a)}-x^{(b)})^2 + (y^{(a)}-y^{(b)})^2}" /></a>
- 立体距离
<a href="https://www.codecogs.com/eqnedit.php?latex=\sqrt{(x^{(a)}-x^{(b)})^2&space;&plus;&space;(y^{(a)}-y^{(b)})^2&space;&plus;&space;(z^{(a)}-z^{(b)})^2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sqrt{(x^{(a)}-x^{(b)})^2&space;&plus;&space;(y^{(a)}-y^{(b)})^2&space;&plus;&space;(z^{(a)}-z^{(b)})^2}" title="\sqrt{(x^{(a)}-x^{(b)})^2 + (y^{(a)}-y^{(b)})^2 + (z^{(a)}-z^{(b)})^2}" /></a>
- n维空间距离
<a href="https://www.codecogs.com/eqnedit.php?latex=\sqrt{\sum\limits_{i=1}^n(x_i^{(a)}-x_i^{(b)})^2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sqrt{\sum\limits_{i=1}^n(x_i^{(a)}-x_i^{(b)})^2}" title="\sqrt{\sum\limits_{i=1}^n(x_i^{(a)}-x_i^{(b)})^2}" /></a>
#### 曼哈顿距离
<a href="https://www.codecogs.com/eqnedit.php?latex=\sum\limits_{i=1}^n|x_i^{(a)}-x_i^{(b)}|" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sum\limits_{i=1}^n|x_i^{(a)}-x_i^{(b)}|" title="\sum\limits_{i=1}^n|x_i^{(a)}-x_i^{(b)}|" /></a>
#### 明可夫斯基距离
<a href="https://www.codecogs.com/eqnedit.php?latex=(\sum\limits_{i=1}^n|x_i^{(a)}-x_i^{(b)}|^p)^{\frac{1}{p}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(\sum\limits_{i=1}^n|x_i^{(a)}-x_i^{(b)}|^p)^{\frac{1}{p}}" title="(\sum\limits_{i=1}^n|x_i^{(a)}-x_i^{(b)}|^p)^{\frac{1}{p}}" /></a>
- 当p=1，相当于曼哈顿距离
- 当p=2，相当于欧拉距离
- 当p=3，其他距离

## 参数
### 超参数
- kNN算法中的k是典型的超参数
    - 默认值为5 （经验数值）
- 距离的权重
    - 距离越近，权重越大
- 关于“距离”的定义
    - 明可夫斯基距离（默认）
    明可夫斯基距离的p取值
        - p=1：曼哈顿距离
        - p=2（默认）：欧拉距离
        - p=3：明可夫斯基距离（其他距离）
    - 其他更多的距离定义
        - 向量空间余弦相似度Cosine Similarity
        - 调整余弦相似度Adjusted Cosine Similarity
        - 皮尔森相关系数 Pearson Correlation Coefficient
        - Jaccard相似系数 Jaccard Coefficient

### 模型参数
kNN算法没有模型参数

## 数据归一化 Feature Scaling
### 需要归一化的原因
如果某些特征数值较大，会主导最终距离的结果
### 解决方案
把所有的数据映射到同一尺度
#### 最值归一化 normalization
把所有数据映射到0~1之间:
<a href="https://www.codecogs.com/eqnedit.php?latex=x_{scale}=\frac{x-x_{min}}{x_{max}-x_{min}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_{scale}=\frac{x-x_{min}}{x_{max}-x_{min}}" title="x_{scale}=\frac{x-x_{min}}{x_{max}-x_{min}}" /></a>

- 适用于分布有明显边界的情况
    - 例如考试分数，最大是100，最小是0
    - 例如每个像素的RGB颜色，都是0~255之间
- 受outlier影响较大
    - 例如收入，有些人特别特别高

#### Standardization（0均值标准化/均值方差归一化）
针对最值归一化的缺憾改进
**把所有数据归一到均值为0，方差为1的分布中**

<a href="https://www.codecogs.com/eqnedit.php?latex=x_{scale}&space;=&space;\frac{x&space;-&space;x_{mean}}{S}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_{scale}&space;=&space;\frac{x&space;-&space;x_{mean}}{S}" title="x_{scale} = \frac{x - x_{mean}}{S}" /></a>

- 并不保证数据在0~1之间
- 但是所有数值的均值在0的位置
- 数据方差/标准差为1

适用于数据分布没有明显的分界（有可能存在极端数据值）。其实数据分布有明显边界的情况也是同样适合的，所以选它一般没错。

### 数据归一化的一些注意事项
#### 对测试数据集如何归一化
例如训练集有均值<a href="https://www.codecogs.com/eqnedit.php?latex=X_{mean}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X_{mean}" title="X_{mean}" /></a>，标准差<a href="https://www.codecogs.com/eqnedit.php?latex=X_{std}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X_{std}" title="X_{std}" /></a>, 那么，测试数据集进行归一化（例如0均值标准化）时，应该使用训练集的均值和标准差，而不是用测试集的均值和标准差。原因有：
1. 测试数据是模拟真实环境，真实环境很可能无法得到所有测试数据的均值和标准差。（个人理解，如果使用测试数据集的均值和标准差，那么以后每有一个新的样例进来，岂不是要重新计算（分配）所有测试样例的均值和标准差？）
2. 对数据的归一化也是算法的一部分

#### 需要保存训练数据集得到的均值和标准差
- 使用skLearn进行数据归一化处理
    - 使用StandardScaler进行0均值标准化


