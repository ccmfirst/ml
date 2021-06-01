import math
import pandas as pd
import heapq
import datetime

'''
    电影推荐
    数据来源：https://grouplens.org/datasets/movielens/
    算法：基于物品的协同过滤推荐算法
    
'''
class ItemCF:
    def __init__(self):
        pass

    def build_train(self, start_time):
        """
        处理数据集，格式：
        用户ID:{物品ID, 评分}
        """
        train = dict()

        data = pd.read_csv('/Users/ccmfirst/pyproject/ml/ml-latest-small/ratings.csv')

        for i in range(len(data)):
            user_id = data['userId'][i]
            item_id = data['movieId'][i]
            score = data['rating'][i]
            train.setdefault(int(user_id), {})
            train[int(user_id)][int(item_id)] = float(score)

        self.train = train
        end_time = datetime.datetime.now()
        print('处理数据集耗时：{0}秒'.format((end_time - start_time).seconds))

    def item_similarity(self):
        """
        计算物品-物品相识度矩阵
        物品相识度 w(i,j)=(N(i)∩N(j))/sqrt(N(i)*N(j))
        """
        start_time = datetime.datetime.now()
        # 物品-物品矩阵 格式：物品ID1：{物品ID2:同时给两件物品评分的人数}
        C = dict()
        # 物品-用户矩阵 格式：物品ID：给物品评分的人数
        N = dict()

        for user_id, items in self.train.items():
            for item_id, score in items.items():
                N.setdefault(item_id, 0)
                N[item_id] += 1

                C.setdefault(item_id, {})
                for i in items.keys():
                    if i == item_id:
                        continue

                    C[item_id].setdefault(i, 0)
                    # 同时给两个物品打分的人数
                    C[item_id][i] += 1

                    # 如果要对活跃用户惩罚
                    C[item_id][i] += 1 / math.log(1 + len(items) * 1.0)
        end_time = datetime.datetime.now()
        print('处理矩阵耗时：{0}秒'.format((end_time - start_time).seconds))

        W = dict()
        for item_id, related_items in C.items():
            W.setdefault(item_id, [])
            for related_item_id, count in related_items.items():
                W[item_id].append([related_item_id, count / math.sqrt(N[item_id] * N[related_item_id])])

            # 归一化
            w_max = max(item[1] for item in W[item_id])
            for item in W[item_id]:
                item[1] /= w_max

        end_time = datetime.datetime.now()
        print('计算物品相似度耗时：{0}秒'.format((end_time - start_time).seconds))
        self.W = W

    def recommendation(self, user_id, k, n):

        start_time = datetime.datetime.now()

        rank = dict()
        items = self.train[user_id]

        for item_id, score in items.items():
            for j, wij in sorted(self.W[item_id], key=lambda x: x[1], reverse=True)[0:k]:
                if j in items.keys():
                    continue

                rank.setdefault(j, {})
                rank[j].setdefault("weight", 0.0)
                rank[j].setdefault("reason", {})
                rank[j]["weight"] += score * wij
                rank[j]["reason"][item_id] = score * wij

        end_time = datetime.datetime.now()
        print('推荐耗时：{0}秒'.format((end_time - start_time).seconds))

        return heapq.nlargest(n, rank.items(), key=lambda x: x[1]['weight'])


if __name__ == '__main__':
    start_time = datetime.datetime.now()

    model = ItemCF()

    model.build_train(start_time)

    model.item_similarity()

    topN = model.recommendation(1, 10, 5)

    end_time = datetime.datetime.now()

    print('总耗时：{0}秒'.format((end_time - start_time).seconds))

    print(topN)