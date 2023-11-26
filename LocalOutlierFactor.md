# カラムごとのLocal Outlier Factor (LOF)の実行

想定として、ある時間に200～800nmまでの波長データで何かしらの値を取得できるセンサーがあり、そのログデータから異常値を予測したいものとします。


具体的には、ノイズを加えたsin波を例に異常検知を実装します。<br>カラムは、200～800まで0.5刻みで記録されています。<br> 100%分類できてるみたいで面白くないけどあくまで例ということで<br>データは訓練用データ、検証用（異常値）データ、テスト用（異常値）データ、検証用（異常値）データです。<br>データの内容
- `./data/train`:学習に使用するノイズの少ないsin波データ
- `./data/test/normal`:検証とテスト兼用の正常sin波データ
- `./data/test/anomaly/test`:異常値のテストデータ
- `./data/test/anomaly/valid`:異常値の検証用データ


まずデータ全体を見て異常検知を行ったあと、<br>カラムごとに異常検知を行い、カラムごとに複数回異常と判定されたカラムとその異常スコアの最大値をDataFrameとして出力してみます。


```python
import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('seaborn')
# from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix
from sklearn. metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
```

## 分析用疑似データ作成


```python
# 保存用フォルダ作成
if not(os.path.isdir('./data')):
    os.mkdir('./data')
if not(os.path.isdir('./data/train')):
    os.mkdir('./data/train')

if not(os.path.isdir('./data/test/normal')):
    os.mkdir('./data/test/normal')
if not(os.path.isdir('./data/test')):
    os.mkdir('./data/test')

if not(os.path.isdir('data/test/anomaly')):
    os.mkdir('data/test/anomaly')
if not(os.path.isdir('data/test/anomaly/test')):
    os.mkdir('data/test/anomaly/test')
if not(os.path.isdir('data/test/anomaly/valid')):
    os.mkdir('data/test/anomaly/valid')
```

#### データ内容の確認

それぞれどのようなデータが入るかを可視化して確認してみます。


```python
def Make_data(index, eta=0.2, outlier=3, mode='train'):
    '''
    sin波にノイズを加えたデータを作成する
    '''
    np.random.seed(42)
    data_size = 1200                                                         # データ数
    X = np.linspace(0,1, data_size)                                          # 0～1まで20個データを作成する
    noise = np.random.uniform(low= -1.0, high=1.0, size=data_size) * eta    # -1～1までの値をデータサイズ個作成し、引数倍する
    y = np.sin(2.0 * np.pi * X) + noise                                      # sin波にノイズを追加する
    # 外れ値の設定
    # 40個ごとにランダムで0～引数outlierまでの1個の値を加算
    if mode == 'test':
        for cnt in range(data_size):
            if cnt % 40 == 0:
                y[cnt] += np.random.randint(0, outlier, 1)
    plt.subplots(figsize=(16, 9))                                            # 表示サイズ指定
    plt.scatter(X, y)                                                        # 散布図
    plt.show()
    # DataFrameを引数をカラム名として作成
    df = pd.DataFrame({
        index:y
    })
    
    return df
```


```python
# train
Make_data(1, 0.05).head()
```


![png](output_6_0.png)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.012546</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.050312</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.033680</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.025586</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.013438</td>
    </tr>
  </tbody>
</table>
</div>




```python
# valid_anomaly
Make_data(1, 0.15, outlier=2.5, mode='test').head()
```


![png](output_7_0.png)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.037638</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.140455</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.080079</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.045318</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.082235</td>
    </tr>
  </tbody>
</table>
</div>




```python
# test_anomaly
Make_data(1, 0.2, outlier=3, mode='test').head()
```


![png](output_8_0.png)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.050184</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.185526</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.103278</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.055184</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.116633</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 可視化の内容を消去して再度宣言
def Make_data(index, eta=0.05, outlier=3, mode='train'):
    '''
    sin波にノイズを加えたデータを作成する
    '''
    np.random.seed(42)
    data_size = 1200                                                         # データ数
    X = np.linspace(0,1, data_size)                                          # 0～1まで20個データを作成する
    noise = np.random.uniform(low= -1.0, high=1.0, size=data_size) * eta    # -1～1までの値をデータサイズ個作成し、引数倍する
    y = np.sin(2.0 * np.pi * X) + noise                                      # sin波にノイズを追加する
    # 外れ値の設定
    # 100個ごとにランダムで0～引数outlierまでの1個の値を加算
    if mode == 'test':
        for cnt in range(data_size):
            if cnt % 100 == 0:
                y[cnt] += np.random.randint(0, outlier, 1)
    # DataFrameを引数をカラム名として作成
    df = pd.DataFrame({
        index:y
    })
    return df
```

## 訓練用データを作成

関数`Make_data()`を使用して訓練用データを作成します。<br>ノイズはデフォルトの`0.2`でデータ数`160個`作成します。


```python
df_train = pd.DataFrame([])
for i in range(1, 161):
    df_base =  Make_data(i).T
    df_train = pd.concat([df_train, df_base])
# 200nm～800nmをカラムとして1桁で丸めて設定
df_train.columns =  np.round(np.linspace(200, 800, 1200), 1)
df_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>200.0</th>
      <th>200.5</th>
      <th>201.0</th>
      <th>201.5</th>
      <th>202.0</th>
      <th>202.5</th>
      <th>203.0</th>
      <th>203.5</th>
      <th>204.0</th>
      <th>204.5</th>
      <th>...</th>
      <th>795.5</th>
      <th>796.0</th>
      <th>796.5</th>
      <th>797.0</th>
      <th>797.5</th>
      <th>798.0</th>
      <th>798.5</th>
      <th>799.0</th>
      <th>799.5</th>
      <th>800.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>-0.012546</td>
      <td>0.050312</td>
      <td>0.03368</td>
      <td>0.025586</td>
      <td>-0.013438</td>
      <td>-0.008202</td>
      <td>-0.012755</td>
      <td>0.073292</td>
      <td>0.052022</td>
      <td>0.067953</td>
      <td>...</td>
      <td>-0.090411</td>
      <td>-0.033694</td>
      <td>-0.052086</td>
      <td>-0.019345</td>
      <td>-0.071625</td>
      <td>0.016194</td>
      <td>0.031628</td>
      <td>0.036407</td>
      <td>0.019725</td>
      <td>-0.036991</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.012546</td>
      <td>0.050312</td>
      <td>0.03368</td>
      <td>0.025586</td>
      <td>-0.013438</td>
      <td>-0.008202</td>
      <td>-0.012755</td>
      <td>0.073292</td>
      <td>0.052022</td>
      <td>0.067953</td>
      <td>...</td>
      <td>-0.090411</td>
      <td>-0.033694</td>
      <td>-0.052086</td>
      <td>-0.019345</td>
      <td>-0.071625</td>
      <td>0.016194</td>
      <td>0.031628</td>
      <td>0.036407</td>
      <td>0.019725</td>
      <td>-0.036991</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.012546</td>
      <td>0.050312</td>
      <td>0.03368</td>
      <td>0.025586</td>
      <td>-0.013438</td>
      <td>-0.008202</td>
      <td>-0.012755</td>
      <td>0.073292</td>
      <td>0.052022</td>
      <td>0.067953</td>
      <td>...</td>
      <td>-0.090411</td>
      <td>-0.033694</td>
      <td>-0.052086</td>
      <td>-0.019345</td>
      <td>-0.071625</td>
      <td>0.016194</td>
      <td>0.031628</td>
      <td>0.036407</td>
      <td>0.019725</td>
      <td>-0.036991</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.012546</td>
      <td>0.050312</td>
      <td>0.03368</td>
      <td>0.025586</td>
      <td>-0.013438</td>
      <td>-0.008202</td>
      <td>-0.012755</td>
      <td>0.073292</td>
      <td>0.052022</td>
      <td>0.067953</td>
      <td>...</td>
      <td>-0.090411</td>
      <td>-0.033694</td>
      <td>-0.052086</td>
      <td>-0.019345</td>
      <td>-0.071625</td>
      <td>0.016194</td>
      <td>0.031628</td>
      <td>0.036407</td>
      <td>0.019725</td>
      <td>-0.036991</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.012546</td>
      <td>0.050312</td>
      <td>0.03368</td>
      <td>0.025586</td>
      <td>-0.013438</td>
      <td>-0.008202</td>
      <td>-0.012755</td>
      <td>0.073292</td>
      <td>0.052022</td>
      <td>0.067953</td>
      <td>...</td>
      <td>-0.090411</td>
      <td>-0.033694</td>
      <td>-0.052086</td>
      <td>-0.019345</td>
      <td>-0.071625</td>
      <td>0.016194</td>
      <td>0.031628</td>
      <td>0.036407</td>
      <td>0.019725</td>
      <td>-0.036991</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1200 columns</p>
</div>



#### 訓練データの保存


```python
# index不要のとき
df_train.to_csv('data/train/X_train.csv', index=False)
```

## 検証とテスト兼用の正常sin波データ

ノイズはデフォルトの`0.2`でデータ数`80個`作成します。


```python
df_normal = pd.DataFrame([])
for i in range(1, 81):
    df_base =  Make_data(i).T
    df_normal = pd.concat([df_normal, df_base])
# 200nm～800nmをカラムとして1桁で丸めて設定
df_normal.columns =  np.round(np.linspace(200, 800, 1200), 1)
df_normal.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>200.0</th>
      <th>200.5</th>
      <th>201.0</th>
      <th>201.5</th>
      <th>202.0</th>
      <th>202.5</th>
      <th>203.0</th>
      <th>203.5</th>
      <th>204.0</th>
      <th>204.5</th>
      <th>...</th>
      <th>795.5</th>
      <th>796.0</th>
      <th>796.5</th>
      <th>797.0</th>
      <th>797.5</th>
      <th>798.0</th>
      <th>798.5</th>
      <th>799.0</th>
      <th>799.5</th>
      <th>800.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>-0.012546</td>
      <td>0.050312</td>
      <td>0.03368</td>
      <td>0.025586</td>
      <td>-0.013438</td>
      <td>-0.008202</td>
      <td>-0.012755</td>
      <td>0.073292</td>
      <td>0.052022</td>
      <td>0.067953</td>
      <td>...</td>
      <td>-0.090411</td>
      <td>-0.033694</td>
      <td>-0.052086</td>
      <td>-0.019345</td>
      <td>-0.071625</td>
      <td>0.016194</td>
      <td>0.031628</td>
      <td>0.036407</td>
      <td>0.019725</td>
      <td>-0.036991</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.012546</td>
      <td>0.050312</td>
      <td>0.03368</td>
      <td>0.025586</td>
      <td>-0.013438</td>
      <td>-0.008202</td>
      <td>-0.012755</td>
      <td>0.073292</td>
      <td>0.052022</td>
      <td>0.067953</td>
      <td>...</td>
      <td>-0.090411</td>
      <td>-0.033694</td>
      <td>-0.052086</td>
      <td>-0.019345</td>
      <td>-0.071625</td>
      <td>0.016194</td>
      <td>0.031628</td>
      <td>0.036407</td>
      <td>0.019725</td>
      <td>-0.036991</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.012546</td>
      <td>0.050312</td>
      <td>0.03368</td>
      <td>0.025586</td>
      <td>-0.013438</td>
      <td>-0.008202</td>
      <td>-0.012755</td>
      <td>0.073292</td>
      <td>0.052022</td>
      <td>0.067953</td>
      <td>...</td>
      <td>-0.090411</td>
      <td>-0.033694</td>
      <td>-0.052086</td>
      <td>-0.019345</td>
      <td>-0.071625</td>
      <td>0.016194</td>
      <td>0.031628</td>
      <td>0.036407</td>
      <td>0.019725</td>
      <td>-0.036991</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.012546</td>
      <td>0.050312</td>
      <td>0.03368</td>
      <td>0.025586</td>
      <td>-0.013438</td>
      <td>-0.008202</td>
      <td>-0.012755</td>
      <td>0.073292</td>
      <td>0.052022</td>
      <td>0.067953</td>
      <td>...</td>
      <td>-0.090411</td>
      <td>-0.033694</td>
      <td>-0.052086</td>
      <td>-0.019345</td>
      <td>-0.071625</td>
      <td>0.016194</td>
      <td>0.031628</td>
      <td>0.036407</td>
      <td>0.019725</td>
      <td>-0.036991</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.012546</td>
      <td>0.050312</td>
      <td>0.03368</td>
      <td>0.025586</td>
      <td>-0.013438</td>
      <td>-0.008202</td>
      <td>-0.012755</td>
      <td>0.073292</td>
      <td>0.052022</td>
      <td>0.067953</td>
      <td>...</td>
      <td>-0.090411</td>
      <td>-0.033694</td>
      <td>-0.052086</td>
      <td>-0.019345</td>
      <td>-0.071625</td>
      <td>0.016194</td>
      <td>0.031628</td>
      <td>0.036407</td>
      <td>0.019725</td>
      <td>-0.036991</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1200 columns</p>
</div>



#### 検証とテスト兼用の正常sin波データの保存


```python
# index不要のとき
df_normal.to_csv('./data/test/normal/X_normal.csv', index=False)
```

## 検証用（異常値）データを作成

ノイズは`0.15`で、外れ値は`2.5`、データ数`30個`作成します。


```python
df_valid = pd.DataFrame([])
for i in range(1, 31):
    df_base =  Make_data(i, 0.15, outlier=2.5, mode='test').T
    df_valid = pd.concat([df_valid, df_base])
# 200nm～800nmをカラムとして1桁で丸めて設定
df_valid.columns =  np.round(np.linspace(200, 800, 1200), 1)
df_valid.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>200.0</th>
      <th>200.5</th>
      <th>201.0</th>
      <th>201.5</th>
      <th>202.0</th>
      <th>202.5</th>
      <th>203.0</th>
      <th>203.5</th>
      <th>204.0</th>
      <th>204.5</th>
      <th>...</th>
      <th>795.5</th>
      <th>796.0</th>
      <th>796.5</th>
      <th>797.0</th>
      <th>797.5</th>
      <th>798.0</th>
      <th>798.5</th>
      <th>799.0</th>
      <th>799.5</th>
      <th>800.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>-0.037638</td>
      <td>0.140455</td>
      <td>0.080079</td>
      <td>0.045318</td>
      <td>-0.082235</td>
      <td>-0.077003</td>
      <td>-0.101138</td>
      <td>0.146527</td>
      <td>0.072245</td>
      <td>0.109567</td>
      <td>...</td>
      <td>-0.176941</td>
      <td>-0.017259</td>
      <td>-0.082909</td>
      <td>0.004838</td>
      <td>-0.162476</td>
      <td>0.090501</td>
      <td>0.126326</td>
      <td>0.130183</td>
      <td>0.069655</td>
      <td>-0.110974</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.037638</td>
      <td>0.140455</td>
      <td>0.080079</td>
      <td>0.045318</td>
      <td>-0.082235</td>
      <td>-0.077003</td>
      <td>-0.101138</td>
      <td>0.146527</td>
      <td>0.072245</td>
      <td>0.109567</td>
      <td>...</td>
      <td>-0.176941</td>
      <td>-0.017259</td>
      <td>-0.082909</td>
      <td>0.004838</td>
      <td>-0.162476</td>
      <td>0.090501</td>
      <td>0.126326</td>
      <td>0.130183</td>
      <td>0.069655</td>
      <td>-0.110974</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.037638</td>
      <td>0.140455</td>
      <td>0.080079</td>
      <td>0.045318</td>
      <td>-0.082235</td>
      <td>-0.077003</td>
      <td>-0.101138</td>
      <td>0.146527</td>
      <td>0.072245</td>
      <td>0.109567</td>
      <td>...</td>
      <td>-0.176941</td>
      <td>-0.017259</td>
      <td>-0.082909</td>
      <td>0.004838</td>
      <td>-0.162476</td>
      <td>0.090501</td>
      <td>0.126326</td>
      <td>0.130183</td>
      <td>0.069655</td>
      <td>-0.110974</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.037638</td>
      <td>0.140455</td>
      <td>0.080079</td>
      <td>0.045318</td>
      <td>-0.082235</td>
      <td>-0.077003</td>
      <td>-0.101138</td>
      <td>0.146527</td>
      <td>0.072245</td>
      <td>0.109567</td>
      <td>...</td>
      <td>-0.176941</td>
      <td>-0.017259</td>
      <td>-0.082909</td>
      <td>0.004838</td>
      <td>-0.162476</td>
      <td>0.090501</td>
      <td>0.126326</td>
      <td>0.130183</td>
      <td>0.069655</td>
      <td>-0.110974</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.037638</td>
      <td>0.140455</td>
      <td>0.080079</td>
      <td>0.045318</td>
      <td>-0.082235</td>
      <td>-0.077003</td>
      <td>-0.101138</td>
      <td>0.146527</td>
      <td>0.072245</td>
      <td>0.109567</td>
      <td>...</td>
      <td>-0.176941</td>
      <td>-0.017259</td>
      <td>-0.082909</td>
      <td>0.004838</td>
      <td>-0.162476</td>
      <td>0.090501</td>
      <td>0.126326</td>
      <td>0.130183</td>
      <td>0.069655</td>
      <td>-0.110974</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1200 columns</p>
</div>



#### 検証（異常値）データの保存


```python
# index不要のとき
df_valid.to_csv('./data/test/anomaly/valid/X_valid.csv', index=False)
```

## テスト用（異常値）データを作成

ノイズはデフォルトの`0.2`で外れ値は`3`、データ数`48個`作成します。


```python
df_test = pd.DataFrame([])
for i in range(1, 49):
    df_base =  Make_data(i, 0.2, outlier=3, mode='test').T
    df_test = pd.concat([df_test, df_base])
# 200nm～800nmをカラムとして1桁で丸めて設定
df_test.columns =  np.round(np.linspace(200, 800, 1200), 1)
df_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>200.0</th>
      <th>200.5</th>
      <th>201.0</th>
      <th>201.5</th>
      <th>202.0</th>
      <th>202.5</th>
      <th>203.0</th>
      <th>203.5</th>
      <th>204.0</th>
      <th>204.5</th>
      <th>...</th>
      <th>795.5</th>
      <th>796.0</th>
      <th>796.5</th>
      <th>797.0</th>
      <th>797.5</th>
      <th>798.0</th>
      <th>798.5</th>
      <th>799.0</th>
      <th>799.5</th>
      <th>800.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>-0.050184</td>
      <td>0.185526</td>
      <td>0.103278</td>
      <td>0.055184</td>
      <td>-0.116633</td>
      <td>-0.111403</td>
      <td>-0.14533</td>
      <td>0.183145</td>
      <td>0.082357</td>
      <td>0.130375</td>
      <td>...</td>
      <td>-0.220205</td>
      <td>-0.009042</td>
      <td>-0.098321</td>
      <td>0.016929</td>
      <td>-0.207902</td>
      <td>0.127655</td>
      <td>0.173675</td>
      <td>0.177071</td>
      <td>0.09462</td>
      <td>-0.147966</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.050184</td>
      <td>0.185526</td>
      <td>0.103278</td>
      <td>0.055184</td>
      <td>-0.116633</td>
      <td>-0.111403</td>
      <td>-0.14533</td>
      <td>0.183145</td>
      <td>0.082357</td>
      <td>0.130375</td>
      <td>...</td>
      <td>-0.220205</td>
      <td>-0.009042</td>
      <td>-0.098321</td>
      <td>0.016929</td>
      <td>-0.207902</td>
      <td>0.127655</td>
      <td>0.173675</td>
      <td>0.177071</td>
      <td>0.09462</td>
      <td>-0.147966</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.050184</td>
      <td>0.185526</td>
      <td>0.103278</td>
      <td>0.055184</td>
      <td>-0.116633</td>
      <td>-0.111403</td>
      <td>-0.14533</td>
      <td>0.183145</td>
      <td>0.082357</td>
      <td>0.130375</td>
      <td>...</td>
      <td>-0.220205</td>
      <td>-0.009042</td>
      <td>-0.098321</td>
      <td>0.016929</td>
      <td>-0.207902</td>
      <td>0.127655</td>
      <td>0.173675</td>
      <td>0.177071</td>
      <td>0.09462</td>
      <td>-0.147966</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.050184</td>
      <td>0.185526</td>
      <td>0.103278</td>
      <td>0.055184</td>
      <td>-0.116633</td>
      <td>-0.111403</td>
      <td>-0.14533</td>
      <td>0.183145</td>
      <td>0.082357</td>
      <td>0.130375</td>
      <td>...</td>
      <td>-0.220205</td>
      <td>-0.009042</td>
      <td>-0.098321</td>
      <td>0.016929</td>
      <td>-0.207902</td>
      <td>0.127655</td>
      <td>0.173675</td>
      <td>0.177071</td>
      <td>0.09462</td>
      <td>-0.147966</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.050184</td>
      <td>0.185526</td>
      <td>0.103278</td>
      <td>0.055184</td>
      <td>-0.116633</td>
      <td>-0.111403</td>
      <td>-0.14533</td>
      <td>0.183145</td>
      <td>0.082357</td>
      <td>0.130375</td>
      <td>...</td>
      <td>-0.220205</td>
      <td>-0.009042</td>
      <td>-0.098321</td>
      <td>0.016929</td>
      <td>-0.207902</td>
      <td>0.127655</td>
      <td>0.173675</td>
      <td>0.177071</td>
      <td>0.09462</td>
      <td>-0.147966</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1200 columns</p>
</div>



#### テスト用（異常値）データの保存


```python
# index不要のとき
df_test.to_csv('./data/test/anomaly/test/X_test.csv', index=False)
```

## データの読み込みとラベル付与


```python
start = time.time()             # 実行開始時間の取得
```


```python
# 学習用フォルダ内のファイル一覧を取得
files_train = glob.glob('./data/train/*')
# 検証とテスト兼用の正常sin波データァイル一覧
files_normal  = glob.glob('./data/test/normal/*')
# 異常値のテストデータ
files_anomaly1 = glob.glob('./data/test/anomaly/test/*')
# 異常値の検証用データ
files_anomaly2 = glob.glob('./data/test/anomaly/valid/*')

# CSV形式のファイルを読み込み、学習データを全てx_trainに格納
X_train = pd.DataFrame([])
for file_name in files_train:
    csv = pd.read_csv(filepath_or_buffer=file_name)
    X_train = pd.concat([X_train, csv])
```


```python
# StandardScalerでz標準化
# sc = StandardScaler()
# sc.fit(X_train)
# X_train = sc.transform(X_train)
```

#### 正解ラベルの付与


```python
# 正常データのラベルを1，異常データのラベルを-1として
# テストデータ用y_test_true，検証用データ用y_valid_trueに格納
x_test_normal, x_test_anomaly1, x_test_anomaly2, X_test, X_valid = pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])
y_test_true, y_valid_true = [], []

# 検証とテスト兼用の正常sin波データを読み込み
# 正常データのラベル1を付与
for file_name in files_normal:
    csv = pd.read_csv(file_name)
    x_test_normal = pd.concat([x_test_normal, csv])
    for i in range(0, len(csv)):
        y_test_true.append(1)
        y_valid_true.append(1)
# 異常値の検証用データを読み込み
# 異常データのラベル-1を付与
for file_name in files_anomaly1:
    csv = pd.read_csv(file_name)
    x_test_anomaly1 = pd.concat([x_test_anomaly1, csv])
    for i in range(0, len(csv)):
        y_test_true.append(-1)
# 異常値のテスト用データを読み込み
# 異常データのラベル-1を付与
for file_name in files_anomaly2:
    csv = pd.read_csv(file_name)
    x_test_anomaly2 = pd.concat([x_test_anomaly2, csv])
    for i in range(0, len(csv)):
        y_valid_true.append(-1)
        
# テストデータx_test，検証用データx_validを正常データと異常データを組み合わせて用意
X_test = pd.concat([x_test_normal, x_test_anomaly1])
X_valid = pd.concat([x_test_normal, x_test_anomaly2])
```


```python
# z標準化
# X_test = sc.transform(X_test)
# X_valid = sc.transform(X_valid)
```


```python
# 正常データ数，異常データ数（テストデータ），テストデータ総数，検証用データ総数を確認
print('data size')
print('訓練データ:{}'.format(X_train.shape[0]))
print('テスト・検証用正常データ:{}'.format(x_test_normal.shape[0]))
print('異常データ数（テストデータ）:{}'.format(x_test_anomaly1.shape[0]))
print('異常データ数（検証データ）:{}'.format(x_test_anomaly2.shape[0]))
print('テストデータ総数:{}'.format(X_test.shape[0]))
print('検証用データ総数:{}'.format(X_valid.shape[0]))
```

    data size
    訓練データ:160
    テスト・検証用正常データ:80
    異常データ数（テストデータ）:48
    異常データ数（検証データ）:30
    テストデータ総数:128
    検証用データ総数:110
    

## Local Outlier Factor (LOF)の実行

データ全体を使用して学習を実行してみます。

検証用データ（学習に使用しない新規データ）を使用して局所密度を計算する場合は、`nobelty=True`とします。<br>しない場合は、してねとエラーが出ます


```python
# LOFの近傍数kを変化させて検証用データに対するF値を取得
idx, f_score = [], []
for k in range(1,11):
    # 近傍数を設定してLOFをインスタンス化
    lof = LocalOutlierFactor(n_neighbors=k, novelty=True, contamination='auto')
    lof.fit(X_train)
    # 検証データの平均F値を追加
    # _predictで局所異常因子を異常度としてある閾値で切った時の2値ラベルの取得
    prec_rec_f = precision_recall_fscore_support(y_valid_true, lof._predict(X_valid))
    f_score.append(np.average(prec_rec_f[2]))
    idx.append(k)
```


```python
# F値が最大となる近傍数kを取得し，LOFに再適合
plt.plot(idx, f_score, 'b-')
plt.xlabel('n_neighbors')
plt.ylabel('F-score')
plt.show()

best_k = np.argmax(f_score)+1
print('Local Outlier Factor result (n_neighbors={}):'.format(best_k))
IF = LocalOutlierFactor(n_neighbors=best_k, novelty=True)
IF.fit(X_train)
```


![png](output_36_0.png)


    Local Outlier Factor result (n_neighbors=1):
    




    LocalOutlierFactor(n_neighbors=1, novelty=True)




```python
# 平均精度・再現率・F値と混同行列の表示
y_pred = IF.predict(X_test)
prec_rec_f = precision_recall_fscore_support(y_test_true, y_pred)
print('Ave. Precision {:0.4f}, Ave. Recall {:0.4f}, Ave. F-score {:0.4f}'.format(np.average(prec_rec_f[0]), np.average(prec_rec_f[1]), np.average(prec_rec_f[2])))
print('Confusion Matrix')
df = pd.DataFrame(confusion_matrix(y_test_true, y_pred))
df.columns = [u'anomaly', u'normal']
df
```

    Ave. Precision 1.0000, Ave. Recall 1.0000, Ave. F-score 1.0000
    Confusion Matrix
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>anomaly</th>
      <th>normal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>48</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>80</td>
    </tr>
  </tbody>
</table>
</div>




```python
fpr, tpr, thresholds = roc_curve(y_test_true, IF.decision_function(X_test), pos_label=1)
roc_auc = roc_auc_score(y_test_true, IF.decision_function(X_test))
plt.plot(fpr, tpr, 'b--',label='ROC for test data (AUC = {:0.2f})'.format(roc_auc), lw=2, linestyle='-')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
```




    <matplotlib.legend.Legend at 0x27ed6a836a0>




![png](output_38_1.png)


## 列ごとに学習を行い、異常発生時のカラムを検出する

カラムごとに学習を行い、テストデータで予測を行って42データ以上異常が予測されたものを異常値として抽出します。

※ROC曲線、混同行列の表示は冗長なのでコメントアウトしてます。


```python
# 異常と判断された波長一覧とその異常値スコア最大値、AUCの保存先
anomaly_spectrim = []
anomaly_score = []
anomaly_roc_auc_score = []
anomaly_count = []
test_count = []
anomaly_count_rate = []
```


```python
# ファイル保存ディレクトリの作成
# if not(os.path.isdir('image')):
#     os.mkdir('image')
```


```python
for column_name, item in X_train.iteritems():
    # LOFの近傍数kを変化させて検証用データに対するF値を取得
    idx, f_score = [], []
    # ベストなk値を取得
    for k in range(1,11):
        # 近傍数を設定してLOFをインスタンス化
        lof = LocalOutlierFactor(n_neighbors=k, novelty=True, contamination='auto')
        # pd.Series.valuesでnumpy.ndarray型にし、np.reshape(-1, 1)でn行1列に変形
        lof.fit(item.values.reshape(-1, 1))
        # 検証データの平均F値を追加
        # _predictで局所異常因子を異常度としてある閾値で切った時の2値ラベルの取得
        prec_rec_f = precision_recall_fscore_support(y_valid_true, lof._predict(X_valid[column_name].values.reshape(-1, 1)))
        f_score.append(np.average(prec_rec_f[2]))
        idx.append(k)
        
    # F値が最大となる近傍数kを取得し，LOFに再適合
    # plt.figure()
    # plt.plot(idx, f_score, 'b-')
    # plt.xlabel('n_neighbors')
    # plt.ylabel('F-score')
    # plt.title('{}Spectrim_F-score.png'.format(column_name))
    # plt.savefig('./image/{}Spectrim_F-score.png'.format(column_name))
    
    # ベストなk値で再学習
    best_k = np.argmax(f_score)+1
    lof = LocalOutlierFactor(n_neighbors=best_k, novelty=True)
    lof.fit(item.values.reshape(-1, 1))
    
    # 最適な近傍数を使用して，テストデータに対する結果を表示
    # print('--------------------')
    # print('Local Outlier Factor result (n_neighbors={})'.format(best_k))
    # print('--------------------')
    # 平均精度・再現率・F値と混同行列の表示# 平均精度・再現率・F値と混同行列の表示
    # y_pred = lof._predict(X_test[column_name].values.reshape(-1, 1))
    # prec_rec_f = precision_recall_fscore_support(y_test_true, y_pred)
    # print('Ave. Precision {:0.4f}, Ave. Recall {:0.4f}, Ave. F-score {:0.4f}'.format(np.average(prec_rec_f[0]), np.average(prec_rec_f[1]), np.average(prec_rec_f[2])))
    # print('Confusion Matrix')
    # df = pd.DataFrame(confusion_matrix(y_test_true, y_pred))
    # df.columns = [u'anomaly', u'normal']
    # print(df)
    # print('--------------------')
    
    # 正解ラベル（y_test_true）と識別関数の出力値（lof._decision_function(X_test[column_name].values.reshape(-1, 1))）を受け取って，ROC曲線を描画
    # fpr, tpr, thresholds = roc_curve(y_test_true, lof._decision_function(X_test[column_name].values.reshape(-1, 1)), pos_label=1)
    # roc_auc = roc_auc_score(y_test_true, lof._decision_function(X_test[column_name].values.reshape(-1, 1)))
    # plt.figure()
    # plt.plot(fpr, tpr, 'k--',label='ROC for test data (AUC = {:0.2f})'.format(roc_auc, lw=2, linestyle="-"))
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('{}Spectrim_ROC curve.png'.format(column_name))
    # plt.legend(loc="lower right")
    # plt.savefig('./image/{}Spectrim_ROC_curve.png'.format(column_name))
    
    # print('--------------------')
    # 42データ以上、異常と判断した時カラム名を表示
    if np.count_nonzero(lof._predict(X_test[column_name].values.reshape(-1, 1)) == -1) >= 42:
        # print('異常と判断した波長:{}nm:'.format(column_name))
        # print('異常スコアの最高値:{}:'.format(lof.score_samples(X_test[column_name].values.reshape(-1, 1)).min()))
        
        # 異常と判断された波長一覧とその異常値スコア最大値の保存先
        anomaly_spectrim.append(column_name)
        anomaly_score.append(lof.score_samples(X_test[column_name].values.reshape(-1, 1)).min())
        anomaly_roc_auc_score.append(roc_auc_score(y_test_true, lof._decision_function(X_test[column_name].values.reshape(-1, 1))))
        anomaly_count.append(np.sum(np.count_nonzero(lof._predict(X_test[column_name].values.reshape(-1, 1) == -1))))
        test_count.append(X_test.shape[0])
        anomaly_count_rate.append(np.count_nonzero(lof._predict(X_test[column_name].values.reshape(-1, 1)) == -1) / X_test.shape[0])
        
```

可視化した結果のイメージ

![png](./256.0Spectrim_F-score.png)

![png](./256.0Spectrim_ROC_curve.png)


```python
# 異常波長とその異常スコアの最大値をDataFrame化
result_df = pd.DataFrame({
    'anomaly_spectrim':anomaly_spectrim,
    'anomaly_score':anomaly_score,
    'anomaly_roc_auc_score':anomaly_roc_auc_score,
    'anomaly_count':test_count,
    'anomaly_count_rate':anomaly_count_rate
})
# 異常値スコアでソート
result_df = result_df.sort_values('anomaly_score')
# 異常度スコアが高いほど異常度が高いと読める用に値を反転
result_df['anomaly_score'] = result_df['anomaly_score'] * (-1)
# 異常度スコアの先頭5データを表示
print(result_df.head())

# ファイル保存ディレクトリの作成
# if not(os.path.isdir('result_anomaly_data')):
#     os.mkdir('result_anomaly_data')
# result_df.to_csv('./result_anomaly_data/result_anomaly.csv', index=False)

# 処理に要した時間を出力
print("Computation time:{0:.3f} sec".format(time.time() - start))
```

         anomaly_spectrim  anomaly_score  anomaly_roc_auc_score  anomaly_count  \
    800             600.3   2.062172e+10                    1.0            128   
    700             550.3   2.009777e+10                    1.0            128   
    900             650.4   1.912366e+10                    1.0            128   
    600             500.3   1.900681e+10                    1.0            128   
    1100            750.5   1.867353e+10                    1.0            128   
    
          anomaly_count_rate  
    800                0.375  
    700                0.375  
    900                0.375  
    600                0.375  
    1100               0.375  
    Computation time:26.206 sec
    


```python
print(result_df.tail())
```

        anomaly_spectrim  anomaly_score  anomaly_roc_auc_score  anomaly_count  \
    231            315.6   7.911280e+06                    1.0            128   
    848            624.4   6.585032e+06                    1.0            128   
    834            617.3   4.896683e+06                    1.0            128   
    649            524.8   4.673404e+06                    1.0            128   
    275            337.6   4.548885e+06                    1.0            128   
    
         anomaly_count_rate  
    231               0.375  
    848               0.375  
    834               0.375  
    649               0.375  
    275               0.375  
    


```python

```
