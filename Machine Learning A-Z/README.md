Machine Learning A-Z
------------------


# Information

- Lecture : [Udemy Lecture](https://www.udemy.com/course/machinelearning/)
- Dataset : https://www.superdatascience.com/pages/machine-learning
- [학습 결과 정리](https://github.com/timetobye/Machine_learning_study/tree/master/Machine%20Learning%20A-Z)

# Table of contents
- Part 1 - Data Preprocessing
- Part 2 - Regression
- Part 3 - Classification
- Part 4 - Clustering
- Part 5 - Association Rule Learning
- Part 6 - Reinforcement Learning
- Part 7 - Natural Language Processing
- Part 8 - Deep Learning
- Part 9 - Dimensionality Reduction
- Part 10 - Model Selection & Boosting


## Part 1 - Data Preprocessing

#### Dataframe to numpy array

```python
# df.values 를 하면 array로 바꿔준다.
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

print(y)

array(['No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes'],
      dtype=object)
```

#### Missing data
- How to deal with missing data?
  - sklearn.preprocessing의 Imputer를 사용하면 쉽게 대응 가능
  - strategy 를 살려보면 중간값(meadian), high frequency 등을 선택 가능
  - Imputation of missing values : https://scikit-learn.org/stable/modules/impute.html

```python
# taking case of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer.fit(X[:, 1:3])
Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)
X[:, 1:3] = imputer.transform(X[:, 1:3])
```

#### categorical data
- LabelEncoder를 이용해서 숫자로 변환하자
  - https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
- dummy encoding
  - 나라 이름을 볼 때 어떤 나라가 숫자가 커야 할 이유가 있나?
  - OneHotEncoder를 써보자
  - https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
- 번외
  - pandas.get_dummies
  - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
```

#### split data set into the train_set and target_set
- 0.2 ~ 0.3이 test_size에 좋은 값이다.
- random_state = 0 넣으면 계속 같은 값 얻음
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state = 0
                                                   )
```

#### feature scaling
- 각 column의 값들이 너무 차이가 나면 머신 러닝에 도움이 안 됨
  - scaling을 통해 값을 조정해준다.
  - standardisation : (x-mean(x)) / std(x)
  - Normalization : (x - min(x)) / (max(x) - min(x))
  - StandardScaler를 사용합시다
  - 정말?
- scikit-learn에서는 다음과 같은 스케일링 클래스를 제공한다.
  - StandardScaler(X): 평균이 0과 표준편차가 1이 되도록 변환.
  - RobustScaler(X): 중앙값(median)이 0, IQR(interquartile range)이 1이 되도록 변환.
  - MinMaxScaler(X): 최대값이 각각 1, 최소값이 0이 되도록 변환
  - MaxAbsScaler(X): 0을 기준으로 절대값이 가장 큰 수가 1또는 -1이 되도록 변환
- 사용방법은 다음과 같다.
  - (1) 학습용 데이터의 분포 추정: 학습용 데이터를 입력으로 하여 fit 메서드를 실행하면 분포 모수를 객체내에 저장
  - (2) 학습용 데이터 변환: 학습용 데이터를 입력으로 하여 transform 메서드를 실행하면 학습용 데이터를 변환 
  - (3) 검증용 데이터 변환: 검증용 데이터를 입력으로 하여 transform 메서드를 실행하면 검증용 데이터를 변환
  - (1)번과 (2)번 과정을 합쳐서 fit_transform 메서드를 사용할 수도 있다.
  - https://datascienceschool.net/view-notebook/f43be7d6515b48c0beb909826993c856/

---------------
## Part 2 - Regression

### Simple Linear regression

#### key concept
어렵게 생각하지 말자. 1차 방정식 해를 구한다는 가벼운 접근으로 시작하면 된다.
- y = b0 + b1 * x1
  - y : dependent variable(DV)
  - x1 : independent variable(IV)
  - b1 : coefficient
  - b0 : constant
  
- find best fitting line
  - line에서 벗어나는 "(y - y^)**2" 의 합이 작을수록 best fitting line
  - Mean square error가 낮은 방향으로 가자.
  - https://en.wikipedia.org/wiki/Mean_squared_error
  
### Multiple regression
마찬가지로 어렵게 생각하지 말자. 다항 함수의 해를 구한다는 가벼운 접근으로 시작하면 된다.
- y = b0 + b1 * x1 + b2 * x2 + b3 * x3 + ???
  - state is categorical value
  - create dummy data
    - create new column one is "New York", another is "California"
    - 각 항목에 1, 0을 교차로 해서 dummy variables를 만든다.
    - 근데 실상은 new york 칼럼만 쓸거임
    
- y = b0 + b1 * x1 + b2 * x2 + b3 * x3 + b4 * D1
  - 1 : new york, 0 : california
  - dummy variables trap을 조심하자.
    - dummy를 생성하면 3개의 feature에 숫자값이 배정되지만, 실상 다 배정될 필요는 없다.
    - 3개 중에 2개만 표현을 할 수 있으면 나머지는 자연스럽게 결정되기 때문이다.
  - https://www.algosome.com/articles/dummy-variable-trap-regression.html
  - http://blog.naver.com/PostView.nhn?blogId=magnking&logNo=221150858186
  - https://stackoverflow.com/questions/49210548/dummy-variable-trap-in-linear-regression

### What is a p-value?
- https://www.mathbootcamps.com/what-is-a-p-value/
- https://www.wikihow.com/Calculate-P-Value

### building a model
- https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/P14-Step-by-step-Blueprints-For-Building-Models.pdf
- 5 methods of building models
  - All-in
  - Backward Elimination
    - p-value를 기준으로 기준 variable들을 조정해서 best model을 찾아감
  - Forward Selection
    - 하나씩 늘려나가면서 best model을 찾아감
  - Bidirectional Elimination
    - pdf 참고
  - Score Comparison
    - pdf 참고

### Polynomial regression
역시 쉽게 접근해보자. 고차 함수이다. 끝
- y = b0 + b1 * x1 + b2 * x2**2 + b3*x3**3 + .... + bn * xn ** n
- 자세한건 jupyter notebook 참고


## Part 3 - Classification
## Part 4 - Clustering
## Part 5 - Association Rule Learning
## Part 6 - Reinforcement Learning
## Part 7 - Natural Language Processing
## Part 8 - Deep Learning
## Part 9 - Dimensionality Reduction
## Part 10 - Model Selection & Boosting


```python


```