## 4.4 CcxGBM

#### CcxGBM算法原理

​	CcxGBM是一个基于决策树的梯度提升集成模型，该算法的设计思路主要是两点：1)单个机器在不牺牲速度的情况下，尽可能多地用上更多的数据；2)多机并行时，通信代价能够尽可能地低，并且在计算上可以做到线性加速。基于上述两个需求，LightGBM利用基于Histogram的算法，通过将连续特征值分段为discrete bins来加快训练的速度并减少内存的使用。Histogram算法仅需要存储feature bin value (离散化后的数值)，不需要原始的feature value，也不用排序，能够减少内存使用。而这一特点也使得在进行数据分割和分割点增益的计算时能够大幅提升计算速度，且在数据并行的时候，用Histogram可以大幅降低通信代价。在Histogram算法之上，LightGBM进行进一步的优化，它采用Leaf-wise (best-first)策略来生长树，每次从当前所有叶子中，找到分裂增益最大(一般也是数据量最大)的一个叶子，然后分裂，如此循环。同大多数GBDT工具使用的Level-wise决策树生长策略相比，在分裂次数相同的情况下，Leaf-wise可以降低更多的误差，得到更好的精度。Leaf-wise的缺点是可能会长出比较深的决策树，产生过拟合。因此LightGBM在Leaf-wise之上增加了一个最大深度的限制，在保证高效率的同时防止过拟合。

#### CcxGBM使用建议

根据CcxGBM算法原理和实践经验提出如下使用建议：

- LightGBM算法训练效率高，且具有较好的准确率，适用于大规模数据场景，能够快速模型训练结果。

- LightGBM使用leaf-wise决策树生成策略算法，因此在调节树的复杂程度时，使用的是num_leaves而不是max_depth，而num_leaves取值应小于2^(max_depth)，以防止过拟合。

- 对于给定的学习率，通过对决策树参数调优(num_leaves，min_child_weight，min_gain_to_split，subsample，colsample_bytree)。在确定一棵树的过程中，我们可以选择不同的参数。

- LightGBM的正则化参数的调优，reg_lambda参数可以降低模型的复杂度，从而提高模型的表现。

#### CcxGBM模型超参数设置

- **数据划分**：系统提供了分层抽样和随机抽样两种数据划分方法；

- **测试集比例**：用于分配模型训练中训练样本与测试样本的比例；

- **num_round**：万象智模平台默认值500，即最大的迭代次数或轮数，也就是需要使用到的决策树数量，取值太小容易欠拟合，取值太大，计算量会太大，而且虽然算法在较多决策树时仍能保持稳健，但可能会发生过拟合，同时需要和参数learing_rate同步进行设置，采用交叉验证CV进行检验。

- **colsample_bytree**：万象智慧平台默认值0.8，控制每棵树随机采样的列数的记录占比。

- learning_rate：万象智模平台默认值0.1，取值范围[0，1]，建议取值0.01-0.3。学习率通过减少每一步的权重，可以提高模型的鲁棒性。

- **min_child_weight**：万象智模平台默认值5，建议取值1-100。最小叶子结点样本权重是控制过拟合的参数之一，值较大时，模型过于保守，即导致欠拟合，值较小时，模型会过拟合，需要通过交叉验证调整。

- **min_gain_to_split**：万象智模平台默认值2，结点分裂所需的最小损失函数减少值，值越大，算法越保守。

- **num_leaves**：万象智模平台默认值31，叶子结点总数用于调节树的复杂程度，与树的最大深度同等作用，大致转换关系为：num_leaves=2^(max_depth)-1。

- **reg_lambda**：万象智模平台默认值300，L2正则化项的权重，可以控制模型复杂度，防止过拟合。lambda取值越大，模型越保守。

- **subsample**：万象智模平台默认值0.9，控制每棵树随机采样的记录的比例，比例小，容易导致欠拟合，比例大，容易导致过拟合。

- **boosting_type**：万象智模平台默认值gbdt，每次迭代选择的模型，gbdt：传统提升决策树模型；dart：Dropouts meet Multiple Additive Regression Trees；goss：基于梯度的单侧采样模型。

- **is_unbalance**：万象智模平台有True和False两个选项，用于处理样本类别分布不均衡的问题。当样本类别分布不均衡时，把参数设置为True时会把负样本的权重设为：正样本数/负样本数。

- **寻优策略**
  - **init_points**：万象智模平台默认值为2，确定寻优方式后，算法对应的模型超参数组合的初始个数。
  - **num_iter**：万象智模平台默认值为5，寻优算法的迭代次数。

- **最优选择指标**

  - **测试集auc最优**：最优模型选择标准为测试集上模型的评估指标auc最大。

  - **测试集/训练集auc综合最优**：最优模型选择标准为综合测试集和训练集上模型的评估指标auc最大。