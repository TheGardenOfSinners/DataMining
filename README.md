特征工程
1 单变量数量统计特征：

voice统计用户记录数，用户不同opp_num记录数

voice统计用户不同 opp_head记录数

voice统计用户 不同 opp_len 的记录数

统计用户不同 call_tyoe 记录数

sms统计用户sms 不同opp_num记录数

sms统计用户sms 不同opp_head 记录数

sms统计用户 不同opp_len记录数

sms不同in_out 记录数


2. one-hot 类数目统计特征:

voice是几点打的电话one-hot统计

voice对opp_num one-hot 统计记录数

sms是几点收发短信one-hot统计

sms opp_head one-hot 统计记录数

2.4 时间统计量：

voice通话时长统计量

模型：
lgb单模型