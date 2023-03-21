import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
df = sns.load_dataset('iris', data_home='seaborn-data', cache=True)

# 绘图显示 - 01. 小提琴图
sns.violinplot(x=df["species"], y=df["sepal_length"])
plt.show()

# 绘图显示 - 02. 核密度估计图
sns.kdeplot(df['sepal_width'])
plt.show()