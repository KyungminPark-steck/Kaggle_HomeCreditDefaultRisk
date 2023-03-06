#필요한 라이브러리 로딩
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 200)

import os, sys
from google.colab import drive

drive.mount('/content/gdrive')

%cd '/content/gdrive/My Drive/'
!ls

#타입 지정하고 모든 데이터 세트 불러오는 함수. (실행은 아직)
def get_dataset():
  prev_dtype = {
        'SK_ID_PREV':np.uint32, 'SK_ID_CURR':np.uint32, 'HOUR_APPR_PROCESS_START':np.int32, 'NFLAG_LAST_APPL_IN_DAY':np.int32,
        'DAYS_DECISION':np.int32, 'SELLERPLACE_AREA':np.int32, 'AMT_ANNUITY':np.float32, 'AMT_APPLICATION':np.float32,
        'AMT_CREDIT':np.float32, 'AMT_DOWN_PAYMENT':np.float32, 'AMT_GOODS_PRICE':np.float32, 'RATE_DOWN_PAYMENT':np.float32,
        'RATE_INTEREST_PRIMARY':np.float32, 'RATE_INTEREST_PRIVILEGED':np.float32, 'CNT_PAYMENT':np.float32,
        'DAYS_FIRST_DRAWING':np.float32, 'DAYS_FIRST_DUE':np.float32, 'DAYS_LAST_DUE_1ST_VERSION':np.float32,
        'DAYS_LAST_DUE':np.float32, 'DAYS_TERMINATION':np.float32, 'NFLAG_INSURED_ON_APPROVAL':np.float32
    }
    
  bureau_dtype = {
      'SK_ID_CURR':np.uint32, 'SK_ID_BUREAU':np.uint32, 'DAYS_CREDIT':np.int32,'CREDIT_DAY_OVERDUE':np.int32,
      'CNT_CREDIT_PROLONG':np.int32, 'DAYS_CREDIT_UPDATE':np.int32, 'DAYS_CREDIT_ENDDATE':np.float32,
      'DAYS_ENDDATE_FACT':np.float32, 'AMT_CREDIT_MAX_OVERDUE':np.float32, 'AMT_CREDIT_SUM':np.float32,
      'AMT_CREDIT_SUM_DEBT':np.float32, 'AMT_CREDIT_SUM_LIMIT':np.float32, 'AMT_CREDIT_SUM_OVERDUE':np.float32,
      'AMT_ANNUITY':np.float32
  }
  
  bureau_bal_dtype = {
      'SK_ID_BUREAU':np.int32, 'MONTHS_BALANCE':np.int32,
  }
  
  pos_dtype = {
      'SK_ID_PREV':np.uint32, 'SK_ID_CURR':np.uint32, 'MONTHS_BALANCE':np.int32, 'SK_DPD':np.int32,
      'SK_DPD_DEF':np.int32, 'CNT_INSTALMENT':np.float32,'CNT_INSTALMENT_FUTURE':np.float32
  }
  
  install_dtype = {
      'SK_ID_PREV':np.uint32, 'SK_ID_CURR':np.uint32, 'NUM_INSTALMENT_NUMBER':np.int32, 'NUM_INSTALMENT_VERSION':np.float32,
      'DAYS_INSTALMENT':np.float32, 'DAYS_ENTRY_PAYMENT':np.float32, 'AMT_INSTALMENT':np.float32, 'AMT_PAYMENT':np.float32
  }
  
  card_dtype = {
      'SK_ID_PREV':np.uint32, 'SK_ID_CURR':np.uint32, 'MONTHS_BALANCE':np.int16,
      'AMT_CREDIT_LIMIT_ACTUAL':np.int32, 'CNT_DRAWINGS_CURRENT':np.int32, 'SK_DPD':np.int32,'SK_DPD_DEF':np.int32,
      'AMT_BALANCE':np.float32, 'AMT_DRAWINGS_ATM_CURRENT':np.float32, 'AMT_DRAWINGS_CURRENT':np.float32,
      'AMT_DRAWINGS_OTHER_CURRENT':np.float32, 'AMT_DRAWINGS_POS_CURRENT':np.float32, 'AMT_INST_MIN_REGULARITY':np.float32,
      'AMT_PAYMENT_CURRENT':np.float32, 'AMT_PAYMENT_TOTAL_CURRENT':np.float32, 'AMT_RECEIVABLE_PRINCIPAL':np.float32,
      'AMT_RECIVABLE':np.float32, 'AMT_TOTAL_RECEIVABLE':np.float32, 'CNT_DRAWINGS_ATM_CURRENT':np.float32,
      'CNT_DRAWINGS_OTHER_CURRENT':np.float32, 'CNT_DRAWINGS_POS_CURRENT':np.float32, 'CNT_INSTALMENT_MATURE_CUM':np.float32
  }

  app_train = pd.read_csv('application_train.csv')
  app_test = pd.read_csv('application_test.csv')
  apps = pd.concat([app_train, app_test])
  prev = pd.read_csv('previous_application.csv')
  bureau = pd.read_csv('bureau.csv', dtype = bureau_dtype)
  bureau_bal = pd.read_csv('bureau_balance.csv', dtype = bureau_bal_dtype)
  pos_bal = pd.read_csv('POS_CASH.balance.csv', dtype = pos_dtype)
  install = pd.read_csv('installments_payments.csv', dtype = install_dtype)
  card_bal = pd.read_csv('credit_card_balance.csv', dtype = card_dtype)

  return apps, prev, bureau, bureau_bal, pos_bal, install, card_bal
