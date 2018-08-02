#!/usr/bin/env python
# -*- coding: utf-8  -*-
# 去除停用词
import codecs, sys


def stopWord(sourceFile, targetFile, stopkey,Num):
    sourcef = codecs.open(sourceFile, 'r',encoding="utf-8")
    targetf = codecs.open(targetFile, 'w',encoding="utf-8")
    print('open source file: ' + sourceFile)
    print('open target file: ' + targetFile)
    wordCounts = []
    lineNum = 1
    line = sourcef.readline()
    while line:
        if line=='\n':
            print('---processing ', lineNum, 'space line not article---')
            lineNum = lineNum + 1
            line = sourcef.readline()
            continue
        print('---processing ', lineNum, ' article---')
        sentence = delstopword(line, stopkey)

        #计算每行的词语数目并且求出最大值,最小值,与平均值

        word_count=len(sentence.split(' '))
        print("word_count:",word_count)
        wordCounts.append(word_count)

        sentence=delNotEnoughWord(sentence,Num)

        if sentence=="":
            lineNum = lineNum + 1
            line = sourcef.readline()
            continue
        # print sentence
        targetf.writelines(sentence + '\n')
        lineNum = lineNum + 1
        line = sourcef.readline()
    print("wordCount MAX:",max(wordCounts))
    print("wordCount MIN:", min(wordCounts))
    print("wordCount AVG:", sum(wordCounts)/len(wordCounts))
    print('well done.')
    sourcef.close()
    targetf.close()


# 删除停用词
def delstopword(line, stopkey):
    wordList = line.split(' ')
    sentence = ''
    for word in wordList:
        word = word.strip()
        if word not in stopkey:
            if word != '\t':
                sentence += word + " "
    return sentence.strip()

#删除不够指定长度的文本数量
def delNotEnoughWord(line,lineNum):
    sentence=line.split(' ')
    if len(sentence) < lineNum:
        return ""
    else:
        return line

if __name__ == '__main__':
    stopkey = [w.strip() for w in codecs.open('data\stopWord.txt', 'r', encoding='utf-8').readlines()]

    # sourceFile = 'data/yuliao/angry_out.txt'
    # targetFile = 'data/yuliao/angry_out_nostop.txt'
    # stopWord(sourceFile, targetFile, stopkey)

    sourceFile = 'data/yuliao/angry_out.txt'
    targetFile = 'data/yuliao/angry_out_nostop.txt'
    stopWord(sourceFile, targetFile, stopkey,5)

    sourceFile = 'data/yuliao/fear_out.txt'
    targetFile = 'data/yuliao/fear_out_nostop.txt'
    stopWord(sourceFile, targetFile, stopkey, 5)

    sourceFile = 'data/yuliao/happy_out.txt'
    targetFile = 'data/yuliao/happy_out_nostop.txt'
    stopWord(sourceFile, targetFile, stopkey, 5)

    sourceFile = 'data/yuliao/sad_out.txt'
    targetFile = 'data/yuliao/sad_out_nostop.txt'
    stopWord(sourceFile, targetFile, stopkey, 5)

    sourceFile = 'data/yuliao/surprise_out.txt'
    targetFile = 'data/yuliao/surprise_out_nostop.txt'
    stopWord(sourceFile, targetFile, stopkey, 5)


