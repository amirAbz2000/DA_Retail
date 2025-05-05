#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('git clone https://github.com/rapidsai/rapidsai-csp-utils.git')
get_ipython().system('python rapidsai-csp-utils/colab/pip-install.py')


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


get_ipython().system('pip install dask')


# In[ ]:


get_ipython().system('pip install openpyxl')


# In[1]:


import dask.dataframe as dd
import pandas as pd
df_pandas = pd.read_excel('/Users/zeynali/Downloads/OnlineـRetail.xlsx')


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df_dask = dd.from_pandas(df_pandas, npartitions=3)  


# In[4]:


summary = df_dask.describe()
print(summary.compute())


# In[5]:


print("Columns:", df_dask.columns)
print("Dtypes:\n", df_dask.dtypes)
print("Number of partitions:", df_dask.npartitions)
print("Approximate number of rows:", df_dask.shape[0].compute())


# In[6]:


print(df_dask.isna().sum().compute())


# In[7]:


df_dask.head()


# In[8]:


df_dask = df_dask.drop("CustomerID", axis=1)


# In[9]:


print(df_dask.isna().sum().compute())


# In[10]:


description_map = df_dask.dropna(subset=['Description'])[['StockCode', 'Description']]\
    .drop_duplicates(subset=['StockCode'])\
    .compute()\
    .set_index('StockCode')['Description']\
    .to_dict()


# In[11]:


print(len(description_map))


# In[12]:


unique_values = df_dask['StockCode'].unique()
print(len(unique_values))


# In[13]:


def fill_description(row):
    if pd.isna(row['Description']):
        return description_map.get(row['StockCode'], 'Unknown')
    return row['Description']

df_dask['Description'] = df_dask.apply(
    fill_description,
    axis=1,
    meta=('Description', 'string[pyarrow]')
)


# In[14]:


print(df_dask.isna().sum().compute())


# In[15]:


df_dask['Quantity'] = df_dask['Quantity'].astype('int32')
df_dask['InvoiceDate'] = df_dask['InvoiceDate'].astype('datetime64[ns]')
df_dask['UnitPrice'] = df_dask['UnitPrice'].astype('float32')


# # in invoice column row that starts with C means it is cancelled

# In[16]:


df_dask['IsCancelled'] = df_dask['InvoiceNo'].str.startswith('C').astype('bool')


# In[17]:


df_dask.head(500)


# In[18]:


cancelled_count = df_dask['IsCancelled'].sum().compute()
print(f"Total number of cancelled rows: {cancelled_count}")


# In[19]:


df_dask.head()


# In[20]:


#پیدا کردن دیتاهایی که از نظر منطقی درست نیستند در کد پایین


# In[21]:


abnormal_df = df_dask[(df_dask["Quantity"] < 0) | (df_dask["UnitPrice"] < 0)]

print(len(abnormal_df))
abnormal_df.head()


# In[22]:


#  رابطه ای بین وجود حرف سی در ابتدای شماره فاکتور و تعداد منفی وجود دارد
#درواقع همه ای فاکتور های با کاراکتر سی تعداد منفی دارند که یعنی احتمالا برگشت خورده اند
# تعداد منفی احتمالا برای بالانش شدن در حساب کردن مبالغ اعمال شده است


# In[23]:


cancelled_df = df_dask[df_dask['IsCancelled']==True]


# In[24]:


cancelled_df.head()


# In[25]:


print(len(cancelled_df))


# all in cancelled_df has negative qunatity

# In[26]:


#هیچ فرضیه ای منطقی برای گزارش دادن بر حسب  قیمت منفی و صفر نمیشود داشته باشیم
# و بیشتر شامل کالاهای خراب یا با دیتای ناقص میوشود
# و نیز درصد بسیار کوچکی از کل را شامل میشوند و از دیتای اصلی پاک خواهد شد


# In[27]:


abnormal_price_df =  df_dask[df_dask['UnitPrice']< 0  ]
print(len(abnormal_price_df))
abnormal_price_df.head()


# In[28]:


abnormal_price_df2 =  df_dask[df_dask['UnitPrice']== 0  ]
print(len(abnormal_price_df2))
abnormal_price_df2.head(10)


# In[29]:


zero_price_by_country = abnormal_price_df2.groupby('Country')['UnitPrice'].count().compute()
zero_price_by_country = zero_price_by_country.sort_values(ascending=False)
print(zero_price_by_country)


# In[30]:


zero_price_by_descrip = abnormal_price_df2.groupby('Description')['UnitPrice'].count().compute()
zero_price_by_descrip = zero_price_by_descrip.sort_values(ascending=False)
print(zero_price_by_descrip)


# In[31]:


# ایجاد دیتاست نهایی


# In[32]:


df =   df_dask[df_dask['UnitPrice'] >0]


# In[33]:


summary = df.describe()
print(summary.compute())


# In[34]:


df.head(5)


# In[35]:


print("Columns:", df.columns)
print("Dtypes:\n", df.dtypes)
print("Number of partitions:", df.npartitions)
print("Approximate number of rows:", df.shape[0].compute())


# In[ ]:


plt.figure(figsize=(12, 6))

sns.barplot(x=df[df['IsCancelled']==True].groupby('Country')['IsCancelled'].count().compute().values
,
            y=df[df['IsCancelled']==True].groupby('Country')['IsCancelled'].count().compute().index
,
            palette='viridis'
           )
plt.xlabel('Count of IsCancelled Items', fontsize=12)
plt.ylabel('Country', fontsize=12)
plt.tight_layout()

# Step 6: Show the plot
plt.show()


# # پیدا کردن بیشترین جنس فروخته شده

# بیشترین کشور با فاکتور کنسل شده

# In[37]:


print( df[df['IsCancelled']==True].groupby('Country')['IsCancelled'].count().compute().sort_values(ascending=False) )


# In[ ]:


plt.figure(figsize=(12, 6))

sns.barplot(x=df[df['IsCancelled']==True].groupby('Description')['IsCancelled'].count().compute().sort_values(ascending=False).head(20).values
,
            y=df[df['IsCancelled']==True].groupby('Description')['IsCancelled'].count().compute().sort_values(ascending=False).head(20).index
,
            palette='viridis'
           )
plt.xlabel('Count of IsCancelled Items', fontsize=12)
plt.ylabel('Country United kingdom', fontsize=12)
plt.tight_layoutprint( df[df['IsCancelled']==True].groupby('Country')['IsCancelled'].count().compute().sort_values(ascending=False) )

# Step 6: Show the plot
plt.show()


# In[ ]:


بیشترین جنس کنسل شده


# In[39]:


print( df[df['IsCancelled']==True].groupby('Description')['IsCancelled'].count().compute().sort_values(ascending=False) )


# بیشترین جنس کنسل شده در بیشترین کشور دارای کنسلی

# #بیشترین برند کنسل شده در بیشترین کشور کنسل شده

# In[40]:


print( df[(df['IsCancelled']==True) &(df['Country']=='United Kingdom' )].groupby('Description')['IsCancelled'].count().compute().sort_values(ascending=False) )


# دیتاست نهایی دیتاست مبتنی بر فاکتور های فروخته شده

# In[68]:


df_final = df[(df['IsCancelled']==False )]


# In[69]:


df_final.describe().compute()


# In[70]:


# فرض می‌کنیم df_dask دیتافریم اصلی شماست

# 1. محاسبه مجموع فروش هر کالا در هر کشور
sales_by_country_product = df_final.groupby(['Country', 'Description'])['Quantity'].sum().reset_index()

# 2. برای اطمینان از عملکرد درست در Dask
sales_by_country_product = sales_by_country_product.persist()

# 3. مرتب‌سازی نزولی برای گرفتن 3 کالای پرفروش
top_products = (
    sales_by_country_product
    .groupby('Country')
    .apply(lambda df_final: df_final.nlargest(3, 'Quantity'), meta=sales_by_country_product)
)

# 4. مرتب‌سازی صعودی برای گرفتن 3 کالای کم‌فروش
bottom_products = (
    sales_by_country_product
    .groupby('Country')
    .apply(lambda df_final: df_final.nsmallest(3, 'Quantity'), meta=sales_by_country_product)
)

# 5. محاسبه نهایی
top_result = top_products.compute()
bottom_result = bottom_products.compute()


# In[71]:


top_result.describe()


# In[72]:


top_result.describe()


# In[73]:


top_result.head(115)


# In[74]:


bottom_result.describe()


# In[75]:


bottom_result.head(115)


# In[76]:


print(top_result['Country'].drop_duplicates().tolist())


# In[77]:


print(bottom_result['Country'].drop_duplicates().tolist())


# In[52]:


# !pip install dash


# In[ ]:


favor_country = input("Enter your country from above to see 3 most and less sold products: ")

def top_less(x):
    print("=== Top 3 Products ===")
    print(top_result[top_result['Country'] == x][['Description', 'Quantity']])
    print("\n#################\n")
    print("=== Least 3 Products ===")
    print(bottom_result[bottom_result['Country'] == x][['Description', 'Quantity']])

top_less(favor_country)


# In[81]:


desc_quantity = df.groupby('Description')['Quantity'].sum().compute().sort_values(ascending=False)
print(desc_quantity)


# In[ ]:




