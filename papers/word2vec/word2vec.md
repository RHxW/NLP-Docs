# Efficient Estimation of Word Representations in Vector Space

## Abstract
提出两个用于在非常大规模的数据集上计算连续特征表示的新模型架构。这些表示的质量通过一个word similarity任务衡量

## 1 Introduction
当前很多NLP技术将words视为具有原子性的单元，也就是说因为words的表达在vocabulary中表示为下标的形式，因此它们之间没有相似度的概念。这种方式有一些优点，例如简单、鲁棒而且大数据量训练的简单模型优于小数据量训练的复杂模型。N-gram模型就是其中之一，它现在已经可以在所有可获取的数据上进行训练。
但是，