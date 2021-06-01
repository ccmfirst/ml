#!/usr/bin/python3
import math
import csv
import datetime
import heapq
import pandas as pd


def BuildTarin(startTime):
    """
    处理数据集，格式：
    用户ID:{物品ID,评分}
    """
    train = dict()
    i = 0
    with open('/Users/ccmfirst/pyproject/ml/ml-latest-small/ratings.csv') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            if i == 0:
                i += 1
                break
            userId = row[0]
            itemId = row[1]
            score = row[2]
            train.setdefault(int(userId), {})
            train[int(userId)][int(itemId)] = float(score)
    endTime = datetime.datetime.now()
    print('处理数据集耗时：{0}秒'.format((endTime - startTime).seconds))
    return train


def ItemSimilarity(train):
    """
    计算物品-物品相似度矩阵
    物品相似度 w(i,j)=(N(i)∩N(j))/sqrt(N(i)*N(j))
    """
    startTime = datetime.datetime.now()
    # 物品-物品矩阵 格式：物品ID1:{物品ID2:同时给两件物品评分的人数}
    C = dict()
    # 物品-用户矩阵 格式：物品ID:给物品评分的人数
    N = dict()

    for userId, items in train.items():
        for itemId, source in items.items():
            N.setdefault(itemId, 0)
            # 给物品打分的人数+1
            N[itemId] += 1

            C.setdefault(itemId, {})
            for i in items.keys():
                if (i == itemId):
                    continue
                C[itemId].setdefault(i, 0)
                # 同时给两个物品打分的人数
                C[itemId][i] += 1
                # 如果要对活跃用户惩罚
                # C[itemId][i]+=1/math.log(1+len(items)*1.0)
    endTime = datetime.datetime.now()
    print('处理矩阵耗时：{0}秒'.format((endTime - startTime).seconds))
    ###############################################################
    startTime = datetime.datetime.now()
    # 计算物品相似度矩阵
    W = dict()
    for itemId, relatedItems in C.items():
        W.setdefault(itemId, [])
        for relatedItemId, count in relatedItems.items():
            W[itemId].append([relatedItemId, count / math.sqrt(N[itemId] * N[relatedItemId])])
        # 归一化
        wmax = max(item[1] for item in W[itemId])
        for item in W[itemId]:
            item[1] /= wmax

    endTime = datetime.datetime.now()
    print('计算物品相似度耗时：{0}秒'.format((endTime - startTime).seconds))
    return W


def Recommendation(train, userId, W, K, N):
    """给用户推荐物品列表
    Args:
        train:训练集
        userId:用户ID
        W:物品相似度矩阵
        K:取和物品j最相似的K个物品
        N:推荐N个物品
    Return:
        推荐列表
    """
    startTime = datetime.datetime.now()
    rank = dict()
    items = train[userId]
    # 遍历用户评分的物品列表
    for itemId, score in items.items():
        # 取出与物品itemId最相似的K个物品
        for j, wij in sorted(W[itemId], key=lambda x: x[1], reverse=True)[0:K]:
            # 如果这个物品j已经被用户评分了，舍弃
            if j in items.keys():
                continue
            # 对物品ItemID的评分*物品itemId与j的相似度 之和
            # rank.setdefault(j,0)
            # rank[j] += score*wij
            rank.setdefault(j, {})
            rank[j].setdefault("weight", 0.0)
            rank[j].setdefault("reason", {})
            rank[j]["weight"] += score * wij
            rank[j]["reason"][itemId] = score * wij
    endTime = datetime.datetime.now()
    print('推荐耗时：{0}秒'.format((endTime - startTime).seconds))
    # 堆排序，推荐权重前N的物品
    return heapq.nlargest(N, rank.items(), key=lambda x: x[1]['weight'])


if __name__ == "__main__":
    startTime = datetime.datetime.now()
    # 构建训练集
    train = BuildTarin(startTime)
    # 计算物品相似度矩阵
    W = ItemSimilarity(train)
    # 给用户推荐TopN
    topN = Recommendation(train, 1, W, 10, 5)
    endTime = datetime.datetime.now()
    print('总耗时：{0}秒'.format((endTime - startTime).seconds))
    print(topN)
