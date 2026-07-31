import nltk
from nltk.tokenize import word_tokenize

# 测试分词功能（punkt 是分词模型）
text = "Hello world! This is a test."
tokens = word_tokenize(text)
print(tokens)  # 输出：['Hello', 'world', '!', 'This', 'is', 'a', 'test', '.']