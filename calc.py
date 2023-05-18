import pandas as pd
import numpy as np
import gspread
import json
import streamlit as st
from google.oauth2 import service_account
import datetime
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots #2軸
from PIL import Image
import matplotlib.pyplot as plt

from logging import debug
# from pandas.core.frame import DataFrame
import openpyxl
import math

from google.oauth2 import service_account
from googleapiclient.discovery import build
import os

import statsmodels.api as sm #統計用
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools #グリッドサーチ

import graph_collection as gc

import warnings

st.set_page_config(page_title='shop_all')
st.markdown('#### shop 来場者分析')

#current working dir
cwd = os.path.dirname(__file__)

#グラフクラスのインスタンス化
graph = gc.Graph()


#******************************************************************************************関数
def get_data(shop_key, SP_SHEET):
    # 秘密鍵jsonファイルから認証情報を取得
    #第一引数　秘密鍵のpath　第二引数　どこのAPIにアクセスするか
    #st.secrets[]内は''で囲むこと
    #scpes 今回実際に使うGoogleサービスの範囲を指定
    credentials = service_account.Credentials.from_service_account_info(st.secrets['gcp_service_account_kurax'], scopes=[ "https://www.googleapis.com/auth/spreadsheets", ])

    #OAuth2のクレデンシャル（認証情報）を使用してGoogleAPIにログイン
    gc = gspread.authorize(credentials)

    # IDを指定して、Googleスプレッドシートのワークブックを取得
    sh = gc.open_by_key(st.secrets[shop_key])

    # シート名を指定して、ワークシートを選択
    worksheet = sh.worksheet(SP_SHEET)

    data= worksheet.get_all_values()

    # スプレッドシートをDataFrameに取り込む
    df = pd.DataFrame(data[1:], columns=data[0])

    return df

def get_file_from_gdrive(cwd, file_name):
    #*********email登録状況のチェック
    # Google Drive APIを使用するための認証情報を取得する
    creds_dict = st.secrets["gcp_service_account_pydrive"]
    creds = service_account.Credentials.from_service_account_info(creds_dict)

    # Drive APIのクライアントを作成する
    #API名（ここでは"drive"）、APIのバージョン（ここでは"v3"）、および認証情報を指定
    service = build("drive", "v3", credentials=creds)

    # 指定したファイル名を持つファイルのIDを取得する
    #Google Drive上のファイルを検索するためのクエリを指定して、ファイルの検索を実行します。
    # この場合、ファイル名とMIMEタイプを指定しています。
    query = f"name='{file_name}' and mimeType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'"
    #指定されたファイルのメディアを取得
    results = service.files().list(q=query).execute()
    items = results.get("files", [])

    if not items:
        st.warning(f"No files found with name: {file_name}")
    else:
        # ファイルをダウンロードする
        file_id = items[0]["id"]
        file = service.files().get(fileId=file_id).execute()
        file_content = service.files().get_media(fileId=file_id).execute()

        # ファイルを保存する
        file_path = os.path.join(cwd, 'data', file_name)
        with open(file_path, "wb") as f:
            f.write(file_content)

#*************************************************************************************データ取得
#*******************************************shop 来店者データ取得
shop_sendai_id = 'SP_SHEET_KEY_SENDAI'
SENDAI_SHEET = 'フォームの回答 1'

df_shopsendai = get_data(shop_sendai_id, SENDAI_SHEET)
df2 = df_shopsendai.copy()

#ゼロ埋め
miseinen_list = []
n20_list = []
n30_list = []
n40_list = []
n50_list = []
n60_list = []
male_list = []
female_list = []
for num in df2['年齢層（未成年）']:
    if num in  ['1', '2', '3', '4', '5']:
        miseinen_list.append(num)
    else:
        miseinen_list.append(0)

for num in df2['年齢層（20代）']:
    if num in  ['1', '2', '3', '4', '5']:
        n20_list.append(num)
    else:
        n20_list.append(0)

for num in df2['年齢層 （30代）']:
    if num in  ['1', '2', '3', '4', '5']:
        n30_list.append(num)
    else:
        n30_list.append(0)

for num in df2['年齢層（40代）']:
    if num in  ['1', '2', '3', '4', '5']:
        n40_list.append(num)
    else:
        n40_list.append(0)

for num in df2['年齢層（50代）']:
    if num in  ['1', '2', '3', '4', '5']:
        n50_list.append(num)
    else:
        n50_list.append(0)

for num in df2['年齢層（60代）']:
    if num in  ['1', '2', '3', '4', '5']:
        n60_list.append(num)
    else:
        n60_list.append(0)

for num in df2['性別（男性）']:
    if num in  ['1', '2', '3', '4', '5']:
        male_list.append(num)
    else:
        male_list.append(0)

for num in df2['性別（女性）']:
    if num in  ['1', '2', '3', '4', '5']:
        female_list.append(num)
    else:
        female_list.append(0)

df_temp = pd.DataFrame(list(zip(miseinen_list, n20_list, n30_list, n40_list, n50_list, n60_list, male_list, female_list)),\
                    columns=df_shopsendai.columns[1:9], index=df_shopsendai.index)

#datetime型に変換
df2['timestamp'] = pd.to_datetime(df2['timestamp'])

#空欄の0埋め処理したdf_tempとtimestampをmerge df3
df_date = (df2[['timestamp']])

df3 = df_date.merge(df_temp, left_index=True, right_index=True, how='inner')

#時刻データを削除した列の新設 str化
df3['timestamp2'] = df3['timestamp'].map(lambda x: x.strftime('%Y-%m-%d'))
#datetime化
df3['timestamp2'] = pd.to_datetime(df3['timestamp2'])

df3['timestamp3'] = df3['timestamp'].map(lambda x: x.strftime('%Y-%m'))
#datetime化
df3['timestamp3'] = pd.to_datetime(df3['timestamp3'])


#組数の算出 日にち
kumi_dict = {}
for  date in df3['timestamp2'].unique():
    df = df3[df3['timestamp2']==date]
    kumi_dict[date] = len(df)

df_kumi = pd.DataFrame(kumi_dict, index=['組数']).T

#組数の算出 月
kumi_month_dict = {}
for  month in df3['timestamp3'].unique():
    df = df3[df3['timestamp3']==month]
    kumi_month_dict[month] = len(df)

df_kumi_month = pd.DataFrame(kumi_month_dict, index=['組数']).T


#************************************shop売上データ取得

#driveからファイル取得dataに保存
file_name = "shopsendai79j.xlsx"
get_file_from_gdrive(cwd, file_name)

#****************************************************************ファイルの加工

@st.cache_data(ttl=datetime.timedelta(hours=1))
def make_data_now(file_path):
    
    df_now = pd.read_excel(
    file_path, sheet_name='受注委託移動在庫生産照会', \
        usecols=[1, 15, 16], parse_dates=True) 
    #index　ナンバー不要　index_col=0, parse_date index datetime型

    # ***INT型への変更***
    df_now['金額'] = df_now['金額'].fillna(0).astype('int64')
    #fillna　０で空欄を埋める

    df_now['伝票番号2'] = df_now['伝票番号'].apply(lambda x: str(x)[:-3])

    #datetime化
    # df_now['受注日'] = pd.to_datetime(df_now['受注日'])

    return df_now

# ***今期受注***
#受注データのpath取得
path_jnow = os.path.join(cwd, 'data', file_name)

df_sales = make_data_now(path_jnow)

#小物の成約行をカット
df_sales2 = df_sales[df_sales['金額'] > 10000]


#売上の算出 日にち
sales_dict = {}
for  date in df_sales2['受注日'].unique():
    df = df_sales2[df_sales2['受注日']==date]
    sales_dict[date] = df['金額'].sum()

#成約件数の算出 日にち
kensuu_dict = {}
for  date in df_sales2['受注日'].unique():
    df = df_sales2[df_sales2['受注日']==date]
    kensuu_dict[date] = df['伝票番号2'].nunique()

df_day_sales = pd.DataFrame(sales_dict, index=['売上']).T
df_day_kensuu = pd.DataFrame(kensuu_dict, index=['成約件数']).T

df_daym = df_day_sales.merge(df_day_kensuu, left_index=True, right_index=True, how='outer')

#*****************組数と売上のmerge
df_all = df_kumi.merge(df_daym, left_index=True, right_index=True, how='left')
df_all = df_all.fillna(0)

start_date = datetime.datetime(2022,10,1,0,0,0)
st.write(start_date)

df_all2 = df_all[df_all.index >= start_date]


#**************日付の欠損を埋める
# 新しい連続した日付のindexを作成
new_index = pd.date_range(start=df_all2.index.min(), end=df_all2.index.max(), freq='D')

# 元のDataFrameを新しいindexで再インデックス
df_all2 = df_all2.reindex(new_index).fillna(0)
st.write(df_all2)

#**************可視化
graph.make_line([df_all2['組数'], df_all2['成約件数']], ['組数', '成約件数'], df_all2.index)

#*************波形分解
fig = sm.tsa.seasonal_decompose(df_all2['組数'].values, period=7).plot()
st.pyplot(fig)

#*************未来予測
df_train = df_all2['2022-10-01': '2023-02-28']
df_test = df_all2['2023-03-01': '2023-05-15']

y = df_train['成約件数']
X = df_train['組数']
sarima_model = sm.tsa.SARIMAX(y, X, order=(2, 1, 1),seasonal_order=(0, 1, 1, 21))
result = sarima_model.fit()

st.write(result.summary())

# #グリッドサーチ
# p = q = range(0, 3)
# sp = sd = sq = range(0, 3)
# # freq = [14, 21]

# #和分1は固定/itertools.product全組み合わせ
# pdq = [(x[0], 1, x[1]) for x in list(itertools.product(p, q))]
# seasonal_pdqf = [(x[0], x[1], x[2], 21) for x in list(itertools.product(sp, sd, sq))]

# warnings.filterwarnings('ignore')

# best_result = [0, 0, 1000000000] #parameter/sesonal/aic

# # X = df_test['組数']
# for param in pdq:
#     for param_seasonal in seasonal_pdqf:
#         try:
#             mod = SARIMAX(y, X, order=param, seasonal_order=param_seasonal)
#             results = mod.fit()
#             # st.write('ARIMA{}, 季節変動{}, AIC{}'.format(param, param_seasonal, results.aic))

#             if results.aic < best_result[2]:
#                 best_result = [param, param_seasonal, results.aic]
#         except:
#             continue

# st.write('AIC最小のモデル:', best_result)

#*****************ホワイトノイズ
st.write('ホワイトノイズ')
st.pyplot(result.plot_diagnostics(lags=21))
st.caption('左上:周期性がないと良い 右上/左下:正規分布に近いと良い 右下: 周期性が無いと良い')

#**************************予測
best_pred = result.predict('2023-03-01', '2023-05-15', exog=df_test['組数'])
# plt.plot(df_train['成約件数'])
# plt.plot(best_pred, 'r')


fig, ax = plt.subplots()
ax.plot(df_all2.index, df_all2['成約件数'])
ax.plot(df_test.index, best_pred, 'r')

st.pyplot(fig)







