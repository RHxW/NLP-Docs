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