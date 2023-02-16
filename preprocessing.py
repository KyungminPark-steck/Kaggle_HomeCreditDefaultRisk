app_train.shape, app_test.shape

apps = pd.concat([app_train, app_test])
apps.shape

apps['TARGET'].value_counts(dropna = False)

apps.info()

object_columns = apps.dtypes[apps.dtypes == 'object'].index.tolist()

object_columns = apps.dtypes[apps.dtypes == 'object'].index.tolist()

for column in object_columns:
  apps[column] = pd.factorize(apps[column])[0]
apps.isnull().sum().head(100)
# -999로 모든 컬럼들의 Null값 변환
apps = apps.fillna(-999)
apps.isnull().sum().head(100)

app_train = apps[apps['TARGET'] != -999]
app_test = apps[apps['TARGET'] == -999]

app_train.shape, app_test.shape

app_test = app_test.drop('TARGET', axis = 1, inplace = False)

app_test.shape

ftr_app = app_train.drop(['SK_ID_CURR', 'TARGET'], axis =1) #x값, 120개
target_app = app_train['TARGET'] #y값, 2개 

from sklearn.model_selection import train_test_split
train_x, valid_x, train_y, valid_y = train_test_split(ftr_app, target_app, test_size = 0.3, random_state = 2020)
train_x.shape, valid_x.shape

from lightgbm import LGBMClassifier

clf = LGBMClassifier(
        n_jobs=-1, #다 쓰겠다
        n_estimators=1000, #반복횟수
        learning_rate=0.02, 
        num_leaves=32,
        subsample=0.8,
        max_depth=12,
        silent=-1,
        verbose=-1
        )

clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
        eval_metric= 'auc', verbose= 100, early_stopping_rounds= 50) #50번 반복. 성능 좋아지지 않으면 멈춘다.

from lightgbm import plot_importance

plot_importance(clf, figsize=(16, 32))

preds = clf.predict_proba(app_test.drop(['SK_ID_CURR'], axis = 1))[:, 1]

clf.predict_proba(app_test.drop(['SK_ID_CURR'], axis = 1))

app_test['TARGET'] = preds
app_test["TARGET"].head()

app_test[['SK_ID_CURR', "TARGET"]].to_csv('app_baseline_01.csv', index = False)

!ls
