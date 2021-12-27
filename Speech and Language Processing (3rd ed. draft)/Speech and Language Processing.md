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
