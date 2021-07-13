# データ処理について
### データ例

        csv_data = '''A,  B ,  C ,  D
                     1.0, 2.0, 3.0, 4.0
                     5.0, 6.0,    , 8.0
                         10.0,11.0,12.0,'''
このデータは３列目の３行目、1列目の4行目がNan値で破損している。
### CSVとして上のものを読み取り
`df = pd.read_csv(StringIO(csv_data))`
### 欠損値の数　取得
`print(df.isnull().sum())`  
#### 出力値

    A    0   
    B    0  
    C    1   
    D    1    
    dtype: int64
### 欠損値の含む行の削除
2行目(3行目)と3行目(4行目)が破損しているため、出力されない   
`print(df.dropna())`  

        A    B    C    D  
    0  1.0  2.0  3.0  4.0
### 欠損値の含む列を削除
C DにそれぞれNan値があるため、出力されない  
`print(df.dropna(axis=1))`  
出力値  

           A     B  
     0   1.0   2.0  
     1   5.0   6.0  
     2  10.0  11.0  
### すべての列がNaNである行だけ削除
`print(df.dropna(axis='all'))`  
thresh=4 非NaN値が４つ未満  
subset=['C'] CにNaNが含まれている行だけ削除  
## 欠測値補完のインスタンスを生成
### strategy meanで平均値補完 NaNを変換
`imr = SimpleImputer(missing_values=np.nan, strategy='mean') `  
他にもmedian(中央値) most_frequent(最頻値)も使用可能  
## データを適合
`imr = imr.fit(df.values)`
## 補完を実行
`imputed_data = imr.transform(df.values)
print(imputed_data)`
## 以上のことをpandasでも可能
`df.fillna(df.mean())`

# 変換器 
SimpleImputerは変換器(transformer)  
変換器クラスはデータの変換に使われ、fit transform の２つになる  
- `fit` 訓練データセットからパラメータを作成
- `transform` 学習したものに対してパラメーター作成  

流れとしては
`fit`でモデルを作成し、`transform`でそれぞれ(test 訓練)を変換する

# 推定器
### predictメソッド(sikit-learn)
transformも使用可能  
流れとしては訓練データや訓練ラベルをfitし、モデルを作成、そこからpredictを行う


# カテゴリデータの処理
### 名義特徴量と順序特徴量
`df = pd.DataFrame([
    ['green', 'M', 10.1, 'class2'],
    ['red', 'L', 13.5, 'class1'],
    ['blue', 'XL', 15.3, 'class2']
])`  

列名  
`df.columns = ['color', 'size', 'price', 'classlabel']`

### Tシャツのサイズと整数を対応させるdictionary ディクショナリーを作成し、自動変換させる
`size_mapping = {'XL': 3, 'L': 2, 'M': 1}
df['size'] = df['size'].map(size_mapping)`  
`print(df)`  

       color  size  price classlabel  
    0  green     1   10.1     class2  
    1    red     2   13.5     class1  
    2   blue     3   15.3     class2  

もとのやつを見たいのであれば  
`inv_size_mapping = {v: k for k, v in size_mapping.items()}
print(df['size'].map(inv_size_mapping))`

    0     M  
    1     L  
    2    XL  
    Name: size, dtype: object

### 右のclass labelに対応させる
`class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}`
`print(class_mapping)`

    {'class1': 0, 'class2': 1}

### 数値は特に関係ない名義特徴量の場合は自動でつけられる
`df['classlabel'] = df['classlabel'].map(class_mapping)`
`print(df)`

        color  size  price  classlabel
    0  green     1   10.1           1
    1    red     2   13.5           0
    2   blue     3   15.3           1

このやり方でも可能である
### ラベルエンコーダーインスタンス作成
`class_le = LabelEncoder()`
### クラスラベルから整数に変換
`y = class_le.fit_transform(df['classlabel'].values)`
`print(y)`

    [1 0 1]

## one-hot Coding
### 色について
`color_le = LabelEncoder()`  
`x = df[['color', 'size', 'price']].values `  
`x[:, 0] = color_le.fit_transform(x[:,0])`  
`print(x)`

    [[1 1 10.1]
      [2 2 13.5]
      [0 3 15.3]]

これだと色が数値で表れてしまい、優劣がついてしまうので、**ONE-HOTエンコーディング**を行う  
これはベクトルのように２値の組み合わせでデータの点として利用  
### ダミー特徴量を作成
scikit-learnのOneHotEncoderクラスを利用  
`color_ohe = OneHotEncoder()`  
`x = df[['color', 'size', 'price']].values`  
`print(color_ohe.fit_transform(x[:,0].reshape(-1, 1)).toarray())`  

    [[0. 1. 0.]
      [0. 0. 1.]
      [1. 0. 0.]]

colorの要素に対して3つの列ができる。  
初期データの他の値に上書きしないように気をつける (name, transformer, columns)の３要素  
`c_transf = ColumnTransformer([('onehot', OneHotEncoder(), [0]), ('nothing', 'passthrough', [1,2])])`  
`print(c_transf.fit_transform(x).astype(float))`  

pandasでもこれを行える  
`print(pd.get_dummies(df[['price', 'color', 'size']]))`  
列が多すぎると計算量（逆行列）が多くなるため正確ではないことも多い。  
その場合は最初の値を消すことができる。  
最初を消しても残り２つのデータは残っているので自ずと分かる  
`color_ohe = OneHotEncoder(categories='auto', drop='first')`  
or    
`print(pd.get_dummies(df[['price', 'color', 'size']], drop_first=True))`
