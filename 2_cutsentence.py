#!/usr/bin/env python
# -*- coding: utf-8  -*-
# 逐行读取文件数据进行jieba分词
import codecs, sys, string, re
import jieba.posseg as pseg


# 文本分词
def prepareData(sourceFile, targetFile):
    f = codecs.open(sourceFile, 'r',encoding="utf-8")
    target = codecs.open(targetFile, 'w',encoding="utf-8")
    print('open source file: ' + sourceFile)
    print('open target file: ' + targetFile)

    lineNum = 1
    line = f.readline()
    while line:
        if line=='\r\n' or line=='\n':
            print('---processing ', lineNum, 'space line not article---')
            lineNum = lineNum + 1
            line = f.readline()
            continue
        print('---processing ', lineNum, ' article---')
        line = clearTxt(line)
        seg_line = sent2word(line)
        if seg_line=='':
            print('---processing ', lineNum, 'space line not article---')
            lineNum = lineNum + 1
            line = f.readline()
            continue
        target.writelines(seg_line + '\n')
        lineNum = lineNum + 1
        line = f.readline()
    print('well done.')
    f.close()
    target.close()


# 清洗文本
def clearTxt(line):
    if line != '':
        line = line.strip()
        # 去除文本中的英文和数字
        line = re.sub("[0-9]", "", line)
        # 去除文本中的中文符号和英文符号
        line = re.sub("[\s+\.\!\/_,$%^*(+\"\'；：“”．]+|[+——！，。？?、~@#￥%……&*（）]+", "", line)
    return line


# 分词，词性标注，词和词性构成一个元组
def sent2word(line,pos_dict=["a","d","n","v","i","p","un"]):
    pos_data = pseg.cut(line)
    segSentence = ''
    for w in pos_data:
        if w.flag in pos_dict:
            segSentence += w.word + " "
    return segSentence



if __name__ == '__main__':
    # sourceFile = '2000_neg.txt'
    # targetFile = '2000_neg_cut.txt'
    # prepareData(sourceFile, targetFile)
    #
    sourceFile = 'data/yuliao/angry.txt'
    targetFile = 'data/yuliao/angry_out.txt'
    prepareData(sourceFile, targetFile)

    sourceFile = 'data/yuliao/fear.txt'
    targetFile = 'data/yuliao/fear_out.txt'
    prepareData(sourceFile, targetFile)

    sourceFile = 'data/yuliao/happy.txt'
    targetFile = 'data/yuliao/happy_out.txt'
    prepareData(sourceFile, targetFile)

    sourceFile = 'data/yuliao/sad.txt'
    targetFile = 'data/yuliao/sad_out.txt'
    prepareData(sourceFile, targetFile)

    sourceFile = 'data/yuliao/surprise.txt'
    targetFile = 'data/yuliao/surprise_out.txt'
    prepareData(sourceFile, targetFile)