import numpy as np
import pandas as pd

df = pd.DataFrame(columns = ['Age'], index = np.arange(1,14))
df['Age'] = [1, 1, 2, 3, 3, 3, 2, 1, 1, 3, 1, 2, 2]
df['Incoming'] = [3, 3, 3, 2, 1, 1, 1, 2, 1, 2, 2, 2, 3]
df['Student'] = [1, 1, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2]
df['Credit Rating'] = [2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1, 1, 2]
df['Buying'] = [1, 1, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2]
df = df.rename_axis('User id').reset_index()
from sklearn import tree
import graphviz

Var = ['Age', 'Incoming', 'Student', 'Credit Rating']
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf.fit(df[Var].values, df['Buying'])
def plot_tree(clf, fn, cn):
    dot_data = tree.export_graphviz(clf,
                                    out_file = None,
                                    feature_names = fn,
                                    class_names = cn,
                                    filled = True,
                                    rounded = True,
                                    special_characters = True)
    graph = graphviz.Source(dot_data)
    graph.render('graph')
    return graph

plot_tree(clf, Var, 'Buying')
# 预测 Age = 3, Incoming = 2, Student = 1, Credit Rating = 1的Buying结果
print('年龄(50), 收入（Medium）,非学生，信用记录 (excellent)的新用户会购买产品，Buying = ',clf.predict([[3, 2, 1, 1]])[0], '(yes).')