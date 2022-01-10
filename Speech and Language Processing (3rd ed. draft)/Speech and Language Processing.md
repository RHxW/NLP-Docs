https://web.stanford.edu/~jurafsky/slp3/
# Speech and Language Proccessing
[toc]

## 1 Introduction

## 2 Regular Expressions, Text Normalization, Edit Distance
* regular expressions：正则表达式
* text normalization：文本正则化
* tokenization：分词，token化
* lemmatization：词形还原，判断有相同词根的词，例如sang,sung,sings都是sing的不同形式，也就是说sing是他们的词根。
* stemming：词干提取，简单版的lemmatization，只是去掉后缀。
* sentence segmentation：语句分割，将一段文本分解成单独的语句
* edit distance：一种用于衡量两个字符串间相似度的度量

### 2.1 Regular Expressions
当从一个corpus（语料库）中搜索一个pattern的时候正则表达式很好用。
#### 2.1.1 Basic Regular Expression Patterns
正则表达式的形式为`/xxx/`，它对大小写敏感，可以用`[]`来实现或条件。

#### 2.1.2 Disjunction, Grouping and Precedence
两个词的或逻辑用分离符(disjunction operator)`|`实现，例如`/cat|dog/`代表"cat or dog"
同一个词中某个位置的或逻辑使用`()`实现，例如`/gupp(y|ies)/`代表"guppy or guppies"

### 2.2 Words
需要首先明确一点，什么是一个‘词’(word)。语料库(corpus, plural:corpora)是一个计算机可以读取的文本集合。
一般在句子中存在标点符号(punctuation)，不同的任务对标点符号的处理不同。
有的时候文本中存在utterance，utterance是句子中的口语上发生关系的部分，例如：
`I do uh main- mainly business data processing`
这个utterance中有两个*不流利的地方(disfluencies)*，断开的*main-*成为**fragment**；另外，例如*uh,um*的词成为**filters**或者**filled pauses**.是否将这些词纳入考虑范围之内依然依赖于具体任务。例如对于语音转录系统，一般需要将这些disfluencies去掉；而对于语音识别任务，如*uh,um*这种词可以帮助预测接下来的词，因为这种词一般代表speaker会重启一个观点。

### 2.3 Corpora

### 2.4 Text Normalization
在所有nlp任务前，都需要进行文本正则化。文本正则化至少包含以下3个操作：
1. 分词(Tokenizing/segmenting words)
2. 词归一化(Normalizing word formats)
3. 语句分割(Segmenting sentences)

#### 2.4.1 Unix Tools

#### 2.4.2 Word Tokenization
分词的时候不能简单地从标点符号处分开，因为对于一些缩写，这样做会发生错误，例如m.p.h, Ph.D., AT&T和cap'n等；而一些特殊的表示也要保持原样例如价格和日期的格式；同样还有URLs，hashtags和邮箱地址等。
标点符号在数字中的表示也有不同的含义，例如英语中一般用逗号分隔3位例如**555,500.50**.但是很多欧洲大陆语言例如西班牙语、法语和德语则用逗号代表小数点而用空格对3位数字进行分割例如：**555 500,50**
tokenizer也能够展开用撇号代替的缩写附着词，例如将what're展开成what are.附着词(clitic)无法独自存在，它依附于另一个词。
语言中还有很多词以组合的形式存在，例如New York或者rock 'n' roll应该被识别为一个词。因此分词任务和**命名实体识别(NER, named entity recognition)** 的联系较为紧密。

#### 2.4.3 Byte-Pair Encoding for Tokenization
除了将词和字作为token外，还有第三个选择，就是自动识别token，我们可以利用数据自动得到tokens.这在处理未知词的时候很有效。如果测试集中出现训练集中没有的词，那这个系统就无法正确识别。
为了解决这一问题，现代分词器会将比word小的粒度作为token，成为**subwords**
大多数分词方法都由两部分组成：一个**token learner**和一个**token segmenter**. 
token learner从原始的训练corpus中得到vocabulary也就是一组token
token segmenter将一个原始的测试句子分割成vocabulary中的token

### 2.5 Minimnum Edit Distance
如何衡量两个句子间的差异？


## 3 N-gram Language Models
n-gram是n个词的序列：2-gram(成为**bigram**)是由两个词组成的序列，例如：'please turn','turn your'或者'your homework';3-gram(**trigram**)是三个词组成的序列，例如：'please turn your'或'turn your homework'

### 3.1 N-Grams
条件概率$P(w|h)$代表在给定历史h的条件下词w出现的概率。假设历史h是'its water is so transparent that',则下一个词是the的概率为：
$$
P(the|its\ water\ is\ so\ transparent\ that).
$$
一种估计概率的方式是根据相关频数进行估计：用一个非常大的corpus，在其中统计*its water is so transparent that*出现的次数以及其后跟随*the*的次数。这样就可以计算：
$$
P(the|its\ water\ is\ so\ transparent\ that)= \\
\frac{C(its\ water\ is\ so\ transparent\ that\ the)}{C(its\ water\ is\ so\ transparent\ that)}
$$
尽管这种方法在很多情况下效果不错，但是对于大多数情况，就算是互联网这种corpus也不够大；这是因为语言具有创造性，随时都可能诞生新的语句，而我们无法始终将这些语句统计在内。

类似地，如果我们想要知道一个完整序列出现的概率，就需要统计所有长度相同序列中该序列出现的概率，例如想知道有5个词的序列*"its water is so transparent"*在全部有5个词的序列中出现的概率；这个统计量太大了！

所以需要一种更高效更聪明的方式来估计在给定历史h或者一个固定词序列W的条件下词w出现的条件概率。
将N个词组成的序列表示为$w_1...w_n$或$w_{1:n}$
一个序列关于其中每个词的联合概率$P(X=w_1,Y=w_2,Z=w_3,...,W=w_n)$表示为$P(w_1,w_2,...,w_n)$
那么如何计算整个序列的概率$P(w_1,w_2,...,w_n)$
一种方式可以用概率的链式法则对其进行分解：
$$
P(X_1...X_n)=P(X_1)P(X_2|X_1)P(X_3|X_{1:2})...P(X_n|X_{1:n-1})\\
=\prod_{k=1}^nP(X_k|X_{1:k-1})
$$
使用链式法则进行分解后有：
$$
P(w_{1:n})=P(w_1)P(w_2|w_1)P(w_3|w_{1:2})...P(w_n|w_{1:n-1})\\
=\prod_{k=1}^nP(w_k|w_{1:k-1})
$$
链式法则展示了计算一个序列的联合概率和计算给定词条件下某一词概率间的关系。上式说明可以通过将数个条件概率相乘来估计联合概率。
但是我们仍然不知道该如何计算给定词的条件概率$P(w_n|w_1^{n-1})$.前面说了不能通过统计的方式计算这个概率。
n-gram模型的核心思想在于，相比于计算一个词出现的整个历史，其实可以只统计前几个词来近似这个历史。
例如bigram模型只用前一个词的条件概率来近似概率。换句话说，用概率
$$P(the|that)$$
来近似
$$
P(the|Walden\ Pond's\ water\ is\ so\ transparent\ that)
$$
而不直接计算上面这个概率

因此对于bigram，我们的近似方法是：
$$
P(w_n|w_{1:n-1})\approx P(w_n|w_{n-1})
$$
这个认为一个词出现的概率只依赖于前序词的假设成为马尔可夫假设。马尔可夫模型是一类概率模型，这种模型认为预测某个事件发生的概率并不需要往前追溯太远。

bigram可以扩展为trigram（追溯前2个词）直到n-gram（追溯前n-1个词）

因此n-gram的通用形式为：
$$
P(w_n|w_{1:n-1})\approx P(w_n|w_{n-N+1:n-1})
$$

如何计算n-gram概率？较直观的概率估计方法是最大似然估计(MLE, maximum likelihood estimation). 参数来自一个cropus并且归一化到0～1

例如，要计算一个词y在词x后出现的概率，我们会计算xy出现的频数，再除以所有以x开头序列的频数：
$$
P(w_n|w_{n-1})=\frac{C(w_{n-1}w_n)}{\sum_wC(w_{n-1}w)}
$$
因为所有以$w_{n-1}$开头的序列的频数等于$w_{n-1}$出现的频数，因此上式可以进行简化：
$$
P(w_n|w_{n-1})=\frac{C(w_{n-1}w_n)}{C(w_{n-1})}
$$

下面用一个只有3句话的mini-corpus来举例。用\<s>表示开头，用<\/s>表示结尾:
```
<s> I am Sam </s>
<s> Sam I am </s>
<s> I do not like green eggs and ham </s>
```
可以计算这个corpus的一些bigram概率：
$$
P(\text{I|<s>})=\frac{2}{3}=.67 \qquad
P(\text{Sam|<s>})=\frac{1}{3}=.33 \qquad
P(\text{am|I})=\frac{2}{3}=.67 \\
P(\text{</s>|Sam})=\frac{1}{2}=.5 \qquad
P(\text{Sam|am})=\frac{1}{2}=.5 \qquad
P(\text{do|I})=\frac{1}{3}=.33
$$

在实际应用中，更多使用的是trigram或者4-gram甚至5-gram模型。而且概率使用的是log概率

### 3.2 Evaluating Language Models
#### 3.2.1 Perplexity
实际上我们不使用原始概率二是用一个称为perplexity(PP)的变种作为度量。一个语言模型在一个测试集上的PP等于这个测试集用词数量归一化后的逆概率。对于一个测试集$W=w_1w_2...w_N$:
$$
PP(W)=P(w_1w_2...w_N)^{-\frac{1}{N}} \\
=\sqrt[N]{\frac{1}{P(w_1w_2...w_N)}}
$$
使用链式法则：
$$
PP(W)=\sqrt[N]{\prod_{i=1}^N\frac{1}{p(w_i|w_1...w_{i-1})}}
$$
因此对于bigram模型就有：
$$
PP(W)=\sqrt[N]{\prod_{i=1}^N\frac{1}{p(w_i|w_{i-1})}}
$$
n-gram模型能够提供的信息越多，它的perplexity越低

### 3.3 Sampling sentences from a language model

### 3.4 Generalization and Zeros
n-gram模型和很多统计模型一样依赖于训练数据。一方面，训练集中存在的先验会影响n-gram模型，另一方面可以通过增加训练集中样本数量N的方式提升n-gram模型的性能。
#### 3.4.1 Unknown Words
将系统没见过的(unknown)词称为OOV(out of vocabulary)词。

### 3.5 Smoothing
测试集和训练集的上下文不同就需要平滑操作（某个词在测试集中的上下文在训练集中没有出现过）

#### 3.5.1 Laplace Smoothing
最简单的方法是在归一化前给每个n-gram类别加1，这种方法称为拉普拉斯平滑。
对于unigram模型，每个词的概率为：
$$
p(w_i)=\frac{c_i}{N}
$$
Laplace smoothing对每个count仅加1，因此也称add-one smoothing. 因为vocabulary里面有V个词，而且每个都是递增的，因此考虑到额外的observations，分母也要加上V：
$$
P_{Laplace}(w_i)=\frac{c_i+1}{N+V}
$$

#### 3.5.2 Add-k smoothing
$$
P_{Add-k}^*(w_n|w_{n-1})=\frac{C(w_{n-1}w_n)+k}{C(w_{n-1})+kV}
$$

#### 3.5.3 Backoff and Interpolation
以上讨论的方法可以解决n-gram中出现的0 count/频率的问题。然而还存在额外的信息可供利用。如果想要计算$P(w_n|w_{n-2}w_{n-1})$但是没有$w_{n-2}w_{n-1}w_n$这样的trigram，这个时候可以用bigram概率$P(w_n|w_{n-1})$来估计。同样也可以用unigram来估计bigram

换言之，有时候用更少的上下文是有好处的，可以帮助提升泛化效果。n-gram中的这种“分级”现象的利用方式有两种。在**backoff**方法中，当trigram的证据足够多的时候使用trigram，否则用bigram，或者更少的时候用unigram，也就是说只会在高级gram的证据不足的时候会向低级gram回退。另一个方法是**interpolation**，始终将不同gram的估计概率进行混合，采用trigram，bigram和unigram的加权和

### 3.6 Kneser-Ney Smoothing

TODO


## 4 Naive Bayes and Sentiment Classification（朴素贝叶斯和情感分类）
朴素贝叶斯算法用于文本分类，本章关注文本分类中的一个任务-情感分析，即提取文本中作者表达出的正面和负面的情感。

情感分析最简单的形式是一个二分类任务，评论中的词会提供很有用的线索，例如*great,richly,awesome,pathetic,awful,ridiculously*等是判断情感的很好的线索。

**Spam detection** 垃圾邮件检测是另一个重要的商业应用，二分类任务将email分为垃圾邮件和正常邮件。

其他应用例如判断文字中包含的语言、判断文字作者等也很重要。而最古老的应用要数对文本进行topic分类，比如判断论文类别等。

### 4.1 Naive Bayes Classifiers
为什么称为‘朴素’呢？因为它对特征间关系作出的假设非常简单。

将一个文档(document)表示为一个bag-of-words，即忽略位置关系的词的集合，只记录每个词出现的频数。

朴素贝叶斯是一个概率分类器，意味着对于一个文档d，在所有类别$c\in C$中，分类器会返回根据文档d有最大后验概率的类别$\hat{c}$:
$$
\hat{c}=\argmax_{c\in C}P(c|d)
$$

贝叶斯分类的思想是利用贝叶斯法则将上式转换成更有用的形式。贝叶斯法则如下：
$$
P(x|y)=\frac{P(y|x)P(x)}{P(y)}
$$
因此有：
$$
\hat{c}=\argmax_{c\in C}P(c|d)=\argmax_{c\in C}\frac{P(d|c)P(c)}{P(d)}
$$
我们可以忽略上式中的P(d)因为它对于每个类别都是相同的（因为关注的是同一个文档d）。因此目标可以简化为：
$$
\hat{c}=\argmax_{c\in C}P(c|d)=\argmax_{c\in C}P(d|c)P(c)
$$
我们称朴素贝叶斯模型是一个生成模型，因为可以把上式看作文档生成的隐含假设：首先从P(c)采样一个类别，然后根据P(d|c)生成词。

要得到分类结果，将先验概率P(c)和文档d的似然P(d|c)相乘并选取结果最大者:
$$
\hat{c}=\argmax_{c\in C}\overbrace{P(d|c)}^{\text{likelihood}} \ \overbrace{P(c)}^{\text{prior}}
$$
不考虑泛化loss的情况下，可以将一个文档d表示称一组特征$f_1,f_2,...,f_n$:
$$
\hat{c}=\argmax_{c\in C}\overbrace{P(f_1,f_2,...,f_n|c)}^{\text{likelihood}} \ \overbrace{P(c)}^{\text{prior}}
$$
但是上式这个形式计算起来太困难：要计算每种特征组合的可能所需的参数太多、计算量太大而且也没有那么大的训练集。因此朴素贝叶斯分类器作出了两个简单假设：
1. 'bag of words'假设：假设词的位置无关紧要
2. 朴素贝叶斯假设：条件独立假设，概率$P(f_i|c)$在给定类别c的时候是独立的，因此有：
   $$
   P(f_1,f_2,...,f_n|c)=P(f_1|c)\cdot P(f_2|c)\cdot ... \cdot P(f_n|c)
   $$

因此在这样的假设下，朴素贝叶斯分类器最终类别预测结果为：
$$
c_{NB}=\argmax_{c\in C}P(c)\prod_{f\in F}P(f|c)
$$
要在文本上使用贝叶斯分类器需要文本中word的位置信息，保持测试文档中每个词的位置下标即可：
$$
c_{NB}=\argmax_{c\in C}P(c)\prod_{i\in positions}P(w_i|c)
$$
朴素贝叶斯也是在对数空间进行计算，因此上式可以表示为：
$$
c_{NB}=\argmax_{c\in C}\log P(c)+\sum_{i\in positions}\log P(w_i|c)
$$
如上式所示，在对数空间中这个分类器根据特征的计算实际上是一个线性模型。分类器利用输入特征的线性组合做出分类决策，这种类型的分类器（如朴素贝叶斯以及逻辑回归）就称为线性分类器。

### 4.2 Training the Naive Bayes Classifier
那么如何计算$P(c)$和$P(f_i|c)$?首先考虑最大似然估计。如果只利用数据中的频数信息，对于类别先验概率P(c)，可以通过训练集中文档包含c类的数量计算，令$N_c$代表训练集中包含c的文档数量，$N_{doc}$代表文档总数，因此有：
$$
\hat{P}(c)=\frac{N_c}{N_{doc}}
$$
要得到概率$P(f_i|c)$，则假设特征即是一个word在文档的bag of words中存在，因此可以使用$P(w_i|c)$，也就是计算$w_i$在所有c类别相关文档中出现的次数和所有词的比值。首先将所有包含类别c的文档拼接成一个大的c文档文本，然后用其中$w_i$的频数来做最大似然估计：
$$
\hat{P}(w_i|c)=\frac{count(w_i,c)}{\sum_{w\in V}count(w,c)}
$$
其中vocabulary V包含全部类别的所有word类型（不仅是c类的words）

这里其实有一个问题，假设我们要估计"fantastic"在给定positive的似然，但是训练集里二者不存在关联关系，也许训练集里它和negative关联在一起。这种情况下概率的计算结果会是0:
$$
\hat{P}(\text{"fantastic"}|\text{positive})=\frac{count(\text{"fantastic"},\text{positive})}{\sum_{w\in V}count(w,\text{positive})}=0
$$
但是因为朴素贝叶斯会把全部特征的似然相乘（因为它很朴素……），所以0会导致结果为0

最简单的解决方案是Laplace smoothing/Add-one smoothing.虽然在语言模型中一般用更高级的平滑方法，但是朴素贝叶斯一般用这种方法：
$$
\hat{P}(w_i|c)=\frac{count(w_i,c)+1}{\sum_{w\in V}(count(w,c)+1)}=\frac{count(w_i,c)+1}{(\sum_{w\in V}count(w,c))+|V|}
$$
遇到没见过的词怎么办？直接忽略，去掉就好

有的时候还会忽略一些stop words比如a，the这种词。如何获取stop words的列表呢？一种方法是统计频率然后将前多少位的词设定为stop words，另一种方法就是下载现成的stop words list.
但是一般忽略stop words并不能提升性能，所以通常不会使用stop words list

### 4.3 Worked example

### 4.4 Optimizing for Sentiment Analysis
情感分析和其他类似的文本分类任务的优化方法：
1. 在这类任务中，一个word是否出现要比它在文本中出现的频率对结果的影响更大，因此通常将word的count设置为0和1来提升性能，这种方法称为二项朴素贝叶斯(binary multinomial naive Bayes, binary NB)
2. 情感分类中的否定词通常对结果的影响更大，例如'I really like this movie'和'I didn't like this movie'这两句话，尽管都含有like，但是第二句的didn't完全否定了like；类似的还有双重否定表示肯定。常见的办法是在文本正则化阶段将否定词后到下一个标点符号间的所有词都加上一个前缀，例如'NOT_'，从而帮助对否定词的识别。
3. 训练集不足，考虑使用公开标准数据集

### 4.5 Naive Bayes for other text classification tasks

### 4.6 Naive Bayes as a Language Model


## 5 Logistic Regression
logistic regression是一种很适合对特征或线索与特定结果间关系进行挖掘的算法。

logistic regression广泛应用于各个领域的分析工具中，它与神经网络关系很密切，实际上一个神经网络可以视作多个logistic regression分类器的组合。

**Generative and Discriminative Classifiers:** 朴素贝叶斯和logistic regression间最大的区别在于logistic regression是一个判别式分类器，而朴素贝叶斯是一个生成式分类器。

判别式和生成式框架的机器学习模型间的差异非常大；
考虑对于图片的猫狗分类任务，生成式模型的目标是理解猫和狗各自长什么样，因此可以让模型生成一个狗或者猫。对于测试图片，这样的模型会考虑狗的模型更符合还是猫的模型更符合，然后得到结果标签。

而判别模型则只会学习分辨二者之间的差别。所以也许训练集中所有狗的图片都戴着项圈而猫都不戴，则测试的时候模型会关注有没有戴项圈这个特征用来判断。

前面提到朴素贝叶斯算法将文档d分为c类的方法并不直接计算$P(c|d)$,而是计算一个likelihood和一个prior
$$
\hat{c}=\argmax_{c\in C}\overbrace{P(d|c)}^{\text{likelihood}} \ \overbrace{P(c)}^{\text{prior}}
$$
如朴素贝叶斯的生成模型会利用这个likelihood项，它表示了如果我们已知一个文档属于类别c，那么该如何生成这个文档的特征。

而判别模型则会直接计算$P(c|d)$，也许会学习到对能直接提升性能的特征加大权重的简单方式。

**Components of a probabilistic machine learning classifier:** 和朴素贝叶斯类似，logistic regression也是一种有监督机器学习的概率分类器。

### 5.1 Classification: the sigmoid
用于将模型输出映射到0-1之间
$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$
sigmoid函数有这样的性质：
$$
1-\sigma(x)=\sigma(-x)
$$

### 5.2 Learning in Logistic Regression

### 5.3 The cross-entropy loss function
二元交叉熵：
$$
L_{CE}(\hat{y},y)=-\log p(y|x)=-[y\log \hat{y}+(1-y)\log (1-\hat{y})]
$$
更一般形式：
$$
L_{CE}(\hat{y},y)=-\sum_iy_i\log \hat{y}_i
$$

### 5.4 Gradient Descent

### 5.5 Regularization
为了避免过拟合现象的出现，需要在目标函数中加入正则化项。

## 6 Vector Semantics and Embeddings
vector semantics是语言含义的表示，称为embeddings

### 6.1 Lexical Semantics
我们应该如何去表示一个词的含义呢？如果用之前提到的方法比如n-gram以及一些经典的nlp应用中，一个词的表示仅仅是一串字母或者一个vocabulary list的下标。

很显然这并不能满足我们的想法，因为我们希望一个语言含义模型能够做各种各样的事。它应该能够告诉我们哪些词是近义词，哪些是反义词，哪些有积极含义，哪些有消极含义。它应该能够区分一个场景中的不同视角，例如支付场景下的买、卖和付款。

### 6.2 Vector Semantics
核心思想是将一个word映射到一个高维的语义空间，words在语义空间中的表示向量成为embeddings

本章会介绍两个最常用的模型：tf-idf和word2vec；tf-idf仅根据临近words的数量来定义一个word的含义，得到的向量很稀疏。word2vec则会得到较为紧密的特征。

### 6.3 Words and Vectors
word含义的向量或分布模型一般基于共生矩阵(co-occurrence matrix)，它是一种表示word间共同出现(co-occur)关系的方法。一般有两种，term-document matrix 和 term-term matrix.

#### 6.3.1 Vectors and documents
term-document matrix里每一行代表vocabulary中的一个word，每一列代表某个文档集合中的一个文档。其中每个元素都代表某个词在某个文档中出现的次数。

一般来说，term-document matrix有|V|行D列。

#### 6.3.2 Words as vectors: document dimensions
行维度可以用来判断哪些文档更相似。

#### 6.3.3 Words as vectors: word dimensions
除了term-document matrix外还可以用term-term matrix来表示word向量，term-term matrix也称word-word matrix或term-context matrix，它是word和word间关系的矩阵，因此尺寸为$|V|\times |V|$，每个元素代表当前词（行）和上下文词（列）共同出现在某些文本（训练数据）中的次数。

### 6.4 Cosine for measuring similarity
余弦相似度

### 6.5 TF-IDF:Weighing terms in the vector
前面提到的co-occurrence matrices，无论是words与文档间关系还是words与其他words间关系，反映的都是频率信息。但是原始频率并不是words间关系的最好度量方式。原始频率的偏见很强而且判别能力较差。如果我们想知道什么样的上下文能够被cherry和strawberry共享但是不被digital和information共享的话，很显然the，it，they这样的词无法提供好的判别能力，因为他们在各种words的上下文中出现的太频繁而且无法提供关于某个词的有用信息。

有点悖论的意思了嗷！频繁出现的词要比只出现一两次的词重要性更高，但是出现太频繁的词却并不重要比如随处可见的the或者good等。该如何平衡这两个相互矛盾的限制呢？

关于这一问题一般有两种解决办法：tf-idf权重法（通常用于维度为文档的时候）和PPMI算法（通常用于维度为words的情况）

tf-idf算法是两项的积，分别关于：
1. term frequency：word t在文档d中的频率。可以将原始频数作为term frequency：
   $$
   tf_{t,d}=count(t,d)
   $$
   一般用$\log_{10}$对原始频率进行一定的压缩，考虑在某一文档中出现100次的词并不应该使它和文档的相关性提高100倍：
   $$
   tf_{t,d}=\log_{10}(count(t,d)+1)
   $$
2. 第二项会针对仅在几个文档中出现的词更高的权重。仅出现于某几个文档的词对于区分这些文档更有用；在所有文档中都出现的词反而用处不大。文档关于一个词t的频率$df_t$代表这个词出现的文档个数，称为document frequency，它和collection frequency不一样，后者是全部文档中某个词出现的总次数，前者是文档个数。

通过inverse document frequency或者idf项权重来强调具有判别能力的词。idf定义为$N/df_t$其中N代表collection中总的文档个数，$df_t$是词t出现的文档个数。因此某个词出现的文档数越少，它的权重就越高。

有的时候corpus没有合适的文档划分，那么就需要手动进行划分然后计算idf

因为数值通常很大，因此idf也计算log值：
$$
idf_t=\log_{10}(\frac{N}{df_t})
$$

### 6.6 Pointwise Mutual Information(PMI)
另一种用于term-term matrices的权重函数是PPMI(positive pointwise mutual information).PPMI指出衡量两个词间关系的最好方式是

PMI可以算是NLP领域中最重要的概念之一。它衡量了两个项目x和y共同出现的频率与假设它们相互独立时同时出现的期望：
$$
I(x,y)=\log_2\frac{P(x,y)}{P(x)P(y)}
$$
则目标词w和上下文词c间的PMI为：
$$
PMI(w,c)=\log_2\frac{P(w,c)}{P(w)P(c)}
$$
分子代表这两个词被同时观测到的频率。分母代表在相互独立的假设下二者同时出现的概率。
PMI在衡量两个词间关系中很有用。
PMI会出现负值，负值对我们没什么用，所以一般用Positive PMI，将全部负值clip到0:
$$
PPMI(w,c)=\max(\log_2\frac{P(w,c)}{P(w)P(c)},0)
$$
在遇到很罕见词的时候PMI会出现较明显偏置，罕见词的PMI值会非常高。一种解决办法是修改其上下文词概率P(c)的计算方法，修改为频数求$\alpha$次方的$P_{\alpha}(c)$：
$$
PPMI_{\alpha}(w,c)=\max(\log_2\frac{P(w,c)}{P(w)P_{\alpha}(c)},0) \\
P_{\alpha}(c)=\frac{count(c)^{\alpha}}{\sum_c count(c)^{\alpha}}
$$
$\alpha$的经验值一般取0.75

### 6.7 Applications of the tf-idf or PPMI vector models
总之，到目前为止介绍的向量语意模型是将一个词映射到一个高维语义空间的方法。
通过计算两个词的tf-idf或PPMI向量的余弦相似度来判断二者是否相似。这种向量通常都是稀疏的

tf-idf模型一般用于判断两个文档是否相似。

### 6.8 Word2vec
接下来介绍一个更强的word表示：embeddings，它是更短更稠密的向量。
从现实角度来看，稠密向量在几乎任何任务中的表现都优于稀疏向量。其中的原因我们目前并不清楚，但是直觉理解，更短更稠密的向量要求分类器的参数量更少，因此泛化能力更强。

本节介绍一种计算embeddings的方法：skip-gram with negative sampling，又称SGNS，它是word2vec这个软件中的两个算法之一，因此有时也不严谨地称为word2vec算法。Word2vec的特征是静态特征，也就是说任何词的embedding都是固定不变的；例如BERT类算法的表示就是动态的，一个词对于不同上下文的特征是不同的。

word2vec的思想是训练一个二分类的分类器用来判断一个词w是否出现在上下文c周围，然后用分类器的权重作为词的embedding（和图像中的识别任务一样）。巧妙的是，这是一个自监督任务，不需要标注。

#### 6.8.1 The classifier
skip-gram训练一个分类器，对于一个测试目标词w和上下文窗口长度L包含的上下文词$c_{1:L}$，分类器会根据上下文窗口与目标词的相似度进行概率预测。因此模型需要保存|V|个词的2|V|个特征，其中有一半是作为目标词的矩阵，另一半是作为上下文和噪声词的矩阵。也就是说skip-gram实际为每个词保存了两个特征embeddings，一个作为目标，一个作为上下文。

#### 6.8.2 Learning skip-gram embeddings
训练的时候，正样本就是窗口内的上下文词，然后随机采样k个负样本（实际上是按权重采样负样本）。

在训练的时候模型会保存两个矩阵，一个是目标矩阵W一个是上下文矩阵C，实际用的时候可以只用目标矩阵W的权重

#### 6.8.3 Other kinds of static embeddings
fasttext是一个word2vec的扩展，它解决了未知词的问题，方法是将单词拆开到字母级别的n-gram表示。

另一个应用广泛的静态embedding 模型是GloVe(Global Vectors)，该模型捕捉全局统计信息。

实际上例如word2vec这种稠密向量与稀疏向量间存在很严格的数学关系，因此word2vec可以视为PPMI矩阵的隐式优化。

### 6.9 Visualizing Embeddings
特征可视化方法，例如聚类或者t-SNE等

### 6.10 Semantic properties of embeddings
介绍embeddings的一些语义性质。
**Different types of similarity or association:** 无论稀疏向量还是稠密向量模型都涉及的一个参数是上下文窗口的长度。一般设定为每侧1-10个词（也就是总共上下文为2-20个词）。

具体的选择依赖于表示的目的。短一些的上下文窗口得到的表示更倾向于关注于句法(syntactic)，因为信息的来源非常近。当使用较短的上下文窗口来计算向量的时候，对于目标词w的最相近词更有可能来源于文本同一部分的语法上类似的词。而较长的上下文窗口会使相似的概念转变为话题相关。

例如对于哈利波特中的Hogwarts这个词，如果用$\pm 2$的窗口，则相近词是其他作品中的虚构的学校名称。但是如果用$\pm 5$的窗口，与之相关的就是哈利波特中的其他名次如Dumbledore, Malfoy和half-blood等。

对这两种不同的相似类型进行区分很有必要。如果两个词通常比较靠近，则称他们有first-order co-occurrence（有时也称syntagmatic association），比如wrote与book或者poem有一阶关联关系。如果两个词有相似的邻居，则称他们有second-order co-occurrence（有时也称paradigmatic association），因此wrote与例如said或remarked等词有二阶关联关系。

**Analogy/Relational Similarity:** embeddings的另一个语义性质是他们有捕捉关联含义的能力。

向量的平行四边形关系

#### 6.10.1 Embeddings and Historical Semantics
对于不同的历史时期，词的含义有可能不同，可以在不同的embedding空间中对其进行分析。

### 6.11 Bias and Embeddings
归纳偏置

虽然embeddings拥有从文本中学习词语含义的能力，但是也会产生隐含于文本里的隐式的偏置和刻板印象。这种刻板印象会导致一个称为allocational harm的结果，意思是系统将资源进行不公平的分配。

### 6.12 Evaluating Vector Models
向量模型最重要的评价方式是进行外部任务评价（放在实际场景下）。但是最常用的评价方法是计算相似度。


## 7 Neural Networks and Neural Language Models

### 7.1 Units
神经元
$$
z=b+\sum_iw_ix_i
$$

### 7.2 The XOR problem

### 7.3 Feedforward Neural Networks

### 7.4 Feedforward networks for NLP: Classification
分类任务

### 7.5 Feedforward Neural Language Modeling
language modeling：根据前面的文本预测接下来出现的词

和n-gram模型相比，neural language models可以处理更长的历史、在相似词的文本上的泛化能力更强并且预测的精度更高。
和n-gram类似也用N个词来近似：
$$
P(w_t|w_1,...,w_{t-1})\approx P(w_t|w_{t-N+1},...,w_{t-1})
$$

#### 7.5.1 Forward inference in the neural language model

### 7.6 Training Neural Nets

#### 7.6.1 Loss function
通常使用的loss函数是交叉熵loss，其一般形式为：
$$
L_{CE}(\hat{y},y)=-\sum_{k=1}^Ky_k\log \hat{y}_k
$$
在交叉熵函数前要用softmax把输入归一化成概率向量

#### 7.6.2 Computing the Gradient


## 8 Sequence Labeling for Parts of Speech and Named Entities
命名实体(named entity)可以理解成专有名词。

词性(POS, parts of speech)和命名实体是理解语句结构和含义的很有用的线索。知道一个词是名词还是动词可以告诉我们它相邻词可能是什么，以及语法结构是什么。

sequence labeling任务指为输入序列的每个词打一个标签，经典的sequence labeling算法有隐马尔可夫模型(HMM, Hidden Markov Model)和条件随机场(CRF, Conditional Random Field)，前者是生成式后者是判别式。

### 8.1 (Mostly) English Word Classes
词性可以分为两大类：封闭词类和开放词类。封闭词类指那些组成固定的词类，例如介词，助动词等。而名次动词等都属于开放词类。封闭词类一般是一些功能性词类，通常较短，在文本中出现频繁并且在语法中有结构性功能。

### 8.2 Part-of-Speech Tagging
Part-of-Speech Tagging是一个消除歧义的任务，一个词可能有多个词性，而tagging的目的是找到当前场景下的正确词性。比如book，有动词和名词两种词性。而POS-tagging的目的就是消除歧义。

POS-tagging算法的精度非常高，基本都在97%以上。

大多数词（85-86%）都是不存在歧义的，但是存在歧义的那一小部分（14-15%）却很常见，连续文本中大概55-67%的词token是有歧义的。

![Figure 8.2](Figs/8.2.png 'Figure 8.2')

### 8.3 Named Entities and Named Entity Tagging
一般可以将命名实体分为四类：
* People(PER)
* Organization(ORG)
* Location(LOC)
* Geo-Political Enitity(GPE)

对很多自然语言处理任务，named entity tagging都是很有用的第一步。

和POS-tagging每个词只有一个词性的特点不同，NER存在分割问题，也就是一个实体可能跨越不止一个词，所以需要确定哪些是entities哪些不是，而且需要确定entities的边界位置。另一个问题是类型的歧义，比如JFK可以代表一个人也可以代表一个机场或者学校、桥梁、街道等。

像NER这种范围识别问题序列标注的一种标准的解决办法是BIO tagging. 这种方法可以将NER当作word-by-word的序列标注任务，标签同时捕捉了边界和命名实体类别。

BIO标注将任何命名实体类别的起始词加上前缀B，中间词加上前缀I，其他的都用O来表示。因此假如有n个命名实体类别，就有2n+1种标签。

IO标注就是实体前加I，非实体用O表示。

BIOES标注是开头的实体加B，中间实体加I，结束实体加E，只有一个词构成的实体前加S，其他非实体加O

训练一个序列标注器（HMM，CRF，RNN，Transformer等）来给每个token的实体类别进行标注。

### 8.4 HMM Part-of-Speech Tagging
一个序列标注算法：隐马尔可夫模型。它是一个概率序列模型：给定一个序列（词的，字母的，句子的或者什么的序列都行），它可以计算一个关于可能的标签序列的概率分布，并且选择最优的标签序列。

#### 8.4.1 Markov Chains
HMM是基于马尔可夫链的模型。马尔可夫链是一个可以告诉我们一个随机变量序列概率的状态的模型。马尔可夫链作出了一个非常强的假设：只有当前状态影响未来状态，也就是说如果要对未来状态进行预测，那么只需要考虑当前的状态即可。例如要想预测明天的天气，可以考察今天的天气，但是不能回顾昨天的天气。

马尔可夫链需要一个初始概率分布$\pi$和一个转移概率分布

正式的表示为，对于一个状态变量的序列，一个马尔可夫模型具体化了关于这个序列概率的马尔科夫假设：当预测未来的时候，过去不产生影响，只有当前最重要：
$$
P(q_i=a|q_1...q_{i-1})=P(q_i=a|q_{i-1})
$$

从形式上，一个马尔可夫链由三部分组成：
1. 一组N个状态的集合$Q=q_1q_2...q_N$
2. 一个转移概率矩阵$A=a_{11}a_{12}...a_{N1}...a_{NN}$
3. 一个初始概率分布$\pi=\pi_1,\pi_2,...,\pi_N$

#### 8.4.2 The Hidden Markov Model
在计算一个可观测事件序列的概率的时候，马尔可夫链是很好用的。但是在很多情况下，我们关心的时间是隐含的，也就是无法直接观测得到。例如通常我们无法直接观测得到文本中的词性标签，而是观测到词，然后根据整个词的序列来推测词性。之所以把标签成为隐含的是因为无法直接观测到。

使用隐马尔可夫模型就可以同时处理可观测事件（例如句子中的词）和隐含事件（例如词的词性），将它们作为概率模型的因素(causal factors)。

一个HMM的组成部分有：
1. 一组N个状态的集合$Q=q_1q_2...q_N$
2. 一个转移概率矩阵$A=a_{11}a_{12}...a_{ij}...a_{NN}$
3. T个观测值的序列$O=o_1o_2...o_T$
4. 一个观测似然（结果）的序列，也称为发射概率$B=b_i(o_t)$. 每个值都代表观测$o_t$通过状态$q_i$生成的概率。
5. 一个初始概率分布$\pi=\pi_1,\pi_2,...,\pi_N$

一个一阶隐马尔可夫模型实例化了两个简化的假设：
1. 和一阶马尔可夫链一样，某个具体状态的概率仅依赖前一个状态
2. 输出观测$o_i$的概率仅依赖于产生状态$q_i$的状态

#### 8.4.3 The components of an HMM tagger
HMM有两个组成部分，概率A和B

A矩阵包含标签转移概率$P(t_i|t_{i-1})$，代表给定标签下某个标签出现的概率。例如助动词will后面一般会接一个一般形式的动词，所以这个转移概率会很高。根据频数来计算这一转移概率的最大似然估计：
$$
P(t_i,t_{i-1})=\frac{C(t_{i-1},t_i)}{C(t_{i-1})}
$$

#### 8.4.4 HMM tagging as decoding
对于任何一个包含隐变量的模型，例如HMM，根据观测值序列来确定隐变量序列的任务称为解码。

对于词性标注，HMM解码的目标是根据给定的观测序列(w)选择一个最有可能的标签序列(t):
$$
\hat{t}_{1:n}=\argmax_{t_1...t_n}P(t_1...t_n|w_1...w_n)
$$
使用贝叶斯公式：
$$
\hat{t}_{1:n}=\argmax_{t_1...t_n}{\frac{P(w_1...w_n|t_1...t_n)P(t_1...t_n)}{P(w_1...w_n)}}
$$
上式忽略分母$P(w_1^n)$有：
$$
\hat{t}_{1:n}=\argmax_{t_1...t_n}P(w_1...w_n|t_1...t_n)P(t_1...t_n)
$$

HMM标注算法进一步做了两个简化假设：
1. 一个词出现的概率仅依赖于它自己的标签而不依赖于相邻的词和标签：
   $$
   P(w_1...w_n|t_1...t_n) \approx \prod_{i=1}^n P(w_i|t_i)
   $$
2. bigram假设：一个标签的概率仅依赖于前一个标签而非整个标签序列：
   $$
   P(t_1...t_n)\approx \prod_{i=1}^n P(t_i|t_{i-1})
   $$

根据这两个假设，进一步有：
$$
\hat{t}_{1:n}=\argmax_{t_1...t_n}P(t_1...t_n|w_1...w_n)\approx\argmax_{t_1...t_n}\prod_{i=1}^n \overbrace{P(w_i|t_i)}^{\text{emission}}\overbrace{P(t_i|t_{i-1})}^{\text{transition}}
$$
上式这两项正好对应我们之前提到的发射概率B和转移概率A

#### 8.4.5 The Viterbi Algorithm
HMM使用Viterbi算法作为解码算法

Viterbi算法首先建立一个概率矩阵，其每列队迎一个观测值$o_t$,每行对应状态图中的一个状态。矩阵中的每个元素$v_t(j)$代表在给定HMM参数$\lambda$情况下，根据前t个观测值得到的以及经过最有可能状态序列$q_1,...,q_{t-1}$得到的状态j的概率。每个元素$v_t(j)$通过递归地采用最可能路径计算得到。形式上，矩阵中每个元素表示的概率为：
$$
v_t(j)=\max_{q_1,...,q_{t-1}} P(q_1...q_{t-1},o_1,o_2,...,o_t,q_t=j|\lambda)
$$
同其他动态规划算法一样，Viterbi算法递归地计算矩阵中的每个元素值。假如我们已经得到t-1时刻的状态值，Viterbi算法会选择到当前元素/状态的最有可能的路径。对一个t时刻的给定状态$q_j$，$v_t(j)$为：
$$
v_t(j)=\max_{i=1}^N v_{t-1}(i)a_{ij}b_j(o_t)
$$
其中$v_{t-1}(i)$代表当前时刻以前的Viterbi路径概率，$a_{ij}$是从状态$q_i$到$q_j$的转移概率，$b_j(o_t)$是给定当前状态j的观测$o_t$的状态观测似然值。

#### 8.4.6 Working through an example

### 8.5 Conditional Random Fields(CRFs)
尽管HMM是一个很强大很有用的模型，但是它需要一些扩展才能达到较高准确率。例如在词性标注等任务中，经常会遇到未知词：经常会诞生的专有名词和缩写，而且新的普通名词和动词产生的速度也很惊人。也许借助大写或构词学的特点可以帮助添加任意特征。或者前后词也可以作为有用的特征。

尽管我们可以尝试修改HMM，让它可以兼容这些能力，但是一般来说向HMM这种生成模型直接加入随机特征是很困难的。我们已经见过一种可以用规范的方式合并任意特征的模型：log线性模型，例如logistic regression模型。但是logistic regression不是序列模型，它为一个样本分配一个类别。

有一种基于log线性模型的判别式序列模型：条件随机场(CRF, conditional random field). 接下来会描述线性链CRF(linear chain CRF), 它是语言处理中最常用的CRF，也是与HMM最相近的CRF

假设有一个输入序列$X=x_1...x_n$，想要计算得到输出标签$Y=y_1...y_n$. HMM模型会计算使$P(Y|X)$最大的标签序列，计算方式基于贝叶斯法则和似然$P(X|Y)$:
$$
\begin{aligned}
   \hat{Y}&=\argmax_Y p(Y|X) \\
   &=\argmax_Y p(X|Y)p(Y) \\
   &=\argmax_Y \prod_i p(x_i|y_i)\prod_i p(y_i|y_{i-1})
\end{aligned}
$$

而CRF则会直接计算后验概率$p(Y|X)$, 训练CRF在所有可能的标签序列中进行判别：
$$
\hat{Y}=\argmax_{Y\in \mathcal{Y}} P(Y|X)
$$
但是CRF不会在每个时刻为每个标签计算一个概率。实际上它在每个时刻都对一组相关特征计算一个log线性函数，然后对这些本地特征进行聚合和归一化得到整个序列的全局特征。

形式上定义，CRF是一个log线性模型，给定输入序列X，它在所有可能的序列$\mathcal{Y}$中将一个概率分配给一整个输出（标签）序列Y. 可以将CRF想像成针对单一token的一个巨大的多元logistic regression版本。CRF中，函数F将整个输入序列X和整个输出序列Y映射称一个特征向量。假设有K个特征，每个特征$F_k$的权重是$w_k$:
$$
p(Y|X)=\frac{\exp\Bigg( \sum_{k=1}^K w_kF_k(X,Y) \Bigg)}{\sum_{Y'\in \mathcal{Y}}\exp\Bigg( \sum_{k=1}^K w_kF_k(X,Y') \Bigg)}
$$
一般将分母表示称一个函数Z(X):
$$
p(Y|X)=\frac{1}{Z(X)}\exp\Bigg( \sum_{k=1}^K w_kF_k(X,Y) \Bigg) \\
Z(X)=\sum_{Y'\in \mathcal{Y}}\exp\Bigg( \sum_{k=1}^K w_kF_k(X,Y') \Bigg)
$$
将这K个函数$F_k(X,Y)$称为全局特征，因为它们每一个都是整个输入序列X和输出序列Y的性质。通过将它们分解成Y中每个位置i的局部特征来对它们进行计算：
$$
F_k(X,Y)=\sum_{i=1}^n f_k(y_{i-1},y_i,X,i)
$$
在线性链CRF中的每个特征$f_k$都可以利用当前输出token，前一个输出token，整个输入序列（或一部分）以及当前位置。这一仅依赖于当前和前一个输出token的限制是线性链CRF的特点。而正是这一特点使利用HMM的Viterbi算法和前向后向算法成为可能。

一般的CRF可以利用任何输出token，因此适用于决策依赖于长距离输出tokens的任务，推理复杂，通常不用于语言处理。

#### 8.5.1 Features in a CRF POS Tagger
为什么说判别式序列模型在引入特征上更方便呢？对于HMM这种生成式模型，它所有的计算都基于两个概率$P(tag|tag)$和$P(word|tag)$, 如果想要在tagging过程中引入一些知识源，就必须找到一种将信息编码进这两个概率的方法。因此，每次引入一个新特征都需要做很多复杂的条件计算，而且这样做的难度随着引入特征的增加而增加。

尽管使用什么样的特征是由设计者决定的，但是具体的特征则是根据特征模版自动选择的。

而且未知词的特征也很重要，其中最重要的特征之一是词形特征，它表示词的抽象字母模式，它将小写字母用x代替，大写字母用X代替，数字用d代替并保持标点符号。例如I.M.F映射为X.X.X, DC10-30映射为XXdd-dd. 还有一种更短的词形特征，省略连续的相同表示，例如DC10-30映射为Xd-d. 另外，前后缀的特征也有用。

在CRF中并不是对每个局部特征都学习一个权重，而是首先将句子里的每个局部特征的值加一起生成每个全局特征，然后将全局特征乘以权重。这样，无论对于训练还是推理都只有一个固定尺寸的权重集合。

#### 8.5.2 Features for CRF Named Entity Recognizers
用于NER的CRF特征和POS tagger所用的特征很类似。

对于位置信息的很有用的特征是地名词典(gazetteer)，它是一个地名的列表，提供了上百万个地理位置的地理和政治信息。它可以作为一个二元特征实现，指示了一个短语是否出现在一个列表中。

#### 8.5.3 Inference and Training for CRFs
如何找到最优的tag序列$\hat{Y}$?根据之前有：
$$
\begin{aligned}
   \hat{Y}&=\argmax_{Y\in \mathcal{Y}} P(Y|X) \\
      &=\argmax_{Y\in \mathcal{Y}}\frac{1}{Z(X)}\exp\Bigg( \sum_{k=1}^K w_kF_k(X,Y) \Bigg) \\
      &= \argmax_{Y\in \mathcal{Y}} \exp\Bigg( \sum_{k=1}^K w_k \sum_{i=1}^nf_k(y_{i-1},y_i,X,i)\Bigg) \\
      &= \argmax_{Y\in \mathcal{Y}} \sum_{k=1}^K w_k \sum_{i=1}^nf_k(y_{i-1},y_i,X,i) \\
      &= \argmax_{Y\in \mathcal{Y}} \sum_{i=1}^n \sum_{k=1}^K w_k f_k(y_{i-1},y_i,X,i)
\end{aligned}
$$
上式中忽略了exp函数和分母Z因为exp函数不影响argmax，而对于一个给定序列X，它对应的分母Z是固定的常数。

该怎么找到这个最优tag序列$\hat{y}$?和HMMs一样，使用Viterbi算法，这是因为和HMM一样，线性链CRF中，每个时刻仅依赖于前一个输出token$y_{i-1}$.

CRFs的学习过程和logistic regression的监督学习算法相同。

### 8.6 Evaluation of Named Entity Recognition


## 9 Deep Learning Architectures for Sequence Processing
语言有内在的时间属性，说一种语言实际上是在沿时间输出一个序列。

之前用前馈网络进行语言建模的方法是滑动窗口法，效果不错，但是还不够好。滑动窗口的问题在于受限的语义，很多语言模型需要的信息是不定距离的；而且滑动窗口还难以学习语言的组合模式，例如短语词组等。

本章介绍两个重要的深度学习架构：循环神经网络和transformer，这两个模型都能直接处理语言的序列特点，因此可以捕捉和利用语言的时序特点。RNN提供了一个针对先前上下文的表示方法，允许模型依赖过去上百个词的信息进行决策。Transformer提供了一个新的机制（包括自注意力和位置编码）帮助表示时间以及关注于词与词在长距离之间的关系。

### 9.1 Language Models Revisited
概率语言模型根据前面的文本来预测一个序列中后面的文本。并使用perplexity这个指标来对语言模型进行评估。

### 9.2 Recurrent Neural Networks
循环神经网络代表一类内部存在循环链接的网络，意味着有的神经元会直接或非直接地依赖于其自身的更早输出作为当前输入。虽然很强大，但是这种网络难以推理和训练。但是在RNN的大类中，有一些限制版本的架构在语言任务上的效果非常好。

对于普通的前馈网络，用一个输入向量代表当前输入，将它乘以一个权重矩阵，然后经过非线性激活函数来计算每个隐藏层神经元的值，然后用这个隐藏层来计算对应的输出。之前提到的滑动窗口方法，序列中的元素是按每次一个的顺序输入网络的。接下来会使用下标代表时间，因此$x_t$代表t时刻的输入x. 

前一个时刻的隐藏层提供了一种形式的记忆或称上下文，它可以编码更早处理过程并对稍后时刻作出的决定进行指导。严格地说，这种方法不限制之前上下文的长度；前序隐藏层提供的上下文信息可以一直扩招到序列开始位置。

加入这样的时域的维度令RNN比无循环架构更复杂。但实际上它们的差异并没有那么大。

如Figure 9.3所示，二者之间最大的改变是增加了一个权重矩阵U，它也是通过反向传播来训练的。
![Figure 9.3](Figs/9.3.png 'Figure 9.3')

#### 9.2.1 Inference in RNNs
RNN需要之前时刻的值来计算当前时刻值的特点实际上是执行了一个增量推理算法。简单循环网络的序列特点同样可以视为将网络在时间维度展开，如Figure 9.5所示。
![Figure 9.5](Figs/9.5.png 'Figure 9.5')

对于一些输入序列格外长的应用，例如语音识别、字符级处理或连续输入流处理，要展开整个序列可能行不通。对于这样的应用，可以将输入分解成多个固定长度的段然后对每个段单独处理。

### 9.3 RNNs as Language Models
RNN语言模型每次处理一个输入序列中的词，目标是根据当前的词以及之前的隐含状态来预测下一个词。RNNs没有n-gram模型的上下文受限的问题，因为隐藏状态原则上能够表示之前所有词的信息。

RNN推理的输入序列是每个词的one-hot表示，输出预测是vocabulary的概率分布。每一步，模型都会使用词嵌入矩阵E获取当前词的embedding，然后将其和前一步隐藏层合并计算得到一个新的隐藏层。这个隐藏层接下来会用于生成一个输出层，然后经过softmax生成整个vocabulary的概率分布向量，也就是说在t时刻有：
$$
\begin{aligned}
   e_t&=Ex_t \\
h_t&=g(Uh_{t-1}+We_t) \\
y_t&=\text{softmax}(Vh_t)
\end{aligned}
$$

你也许会发现输入的embedding矩阵E和最终层矩阵V很相似。E的列代表了vocabulary中每个word在训练中学习到的embeddings，相似的词embeddings也相似。

