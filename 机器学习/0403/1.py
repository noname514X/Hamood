import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
item_list = [['牛奶','面包'],
['面包','尿布','啤酒','土豆'],
['牛奶','尿布','啤酒','可乐'],
['面包','牛奶','尿布','啤酒'],
['面包','牛奶','尿布','可乐']]


te = TransactionEncoder()
te_ary = te.fit(item_list).transform(item_list)
df = pd.DataFrame(te_ary, columns=te.columns_)


frequent_itemsets = fpgrowth(df, min_support=0.5, use_colnames=True)
print(frequent_itemsets)


frequent_itemsets.sort_values(by='support', ascending=False, inplace=True)


print(frequent_itemsets[frequent_itemsets.itemsets.apply(lambda x: len(x)) == 2])


from mlxtend.frequent_patterns import association_rules


association_rule = association_rules(frequent_itemsets,metric='confidence',min_threshold=0.5)


association_rule.sort_values(by='lift',ascending=False,inplace=True)
association_rule