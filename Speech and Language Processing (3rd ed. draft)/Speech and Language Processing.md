https://web.stanford.edu/~jurafsky/slp3/

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
语言中还有很多词以组合的形式存在，例如New York或者rock 'n' roll应该被识别为一个词。因此分词任务和**命名实体识别(NER, named entity recognition)**的联系较为紧密。

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
