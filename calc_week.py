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
from matplotlib import mlab

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

from scipy import stats

st.set_page_config(page_title='shop_analysis')
st.markdown('#### shop分析')

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

df_all2 = df_all[df_all.index >= start_date]


#**************日付の欠損を埋める
# 新しい連続した日付のindexを作成
new_index = pd.date_range(start=df_all2.index.min(), end=df_all2.index.max(), freq='D')

# 元のDataFrameを新しいindexで再インデックス
df_all2 = df_all2.reindex(new_index).fillna(0)

#********************累計列の準備
df_all2['cum組数'] = df_all2['組数'].cumsum()
df_all2['cum売上'] = df_all2['売上'].cumsum()
df_all2['cum成約件数'] = df_all2['成約件数'].cumsum()

with st.expander('df_all2', expanded=False):
    st.write(df_all2)

#***************************月集計
st.markdown('### 月')
df_month = df_all2.resample('M').sum()

df_month['売上/組数'] = df_month['売上'] / df_month['組数']
df_month['売上/成約件数'] = df_month['売上'] / df_month['成約件数']
df_month['成約件数/組数'] = df_month['成約件数'] / df_month['組数']

with st.expander('df_week', expanded=False):
    st.write(df_month)

st.write('来店客数/月')
graph.make_line([df_month['組数']], ['組数'], df_month.index)

st.write('売上/月')
graph.make_line([df_month['売上']], ['売上'], df_month.index)

st.write('売上/組数')
graph.make_line([df_month['売上/組数']],['売上/組数'], df_month.index ) 

st.write('売上/成約件数')
graph.make_line([df_month['売上/成約件数']],['売上/成約件数'], df_month.index ) 

st.write('成約件数/組数')
graph.make_line([df_month['成約件数/組数']],['成約件数/組数'], df_month.index )

#***************************************************************************day
st.markdown('### 日')

df_day = df_all2.copy()

df_day['cum売上/組数'] = df_day['cum売上'] / df_day['cum組数']
df_day['cum売上/成約件数'] = df_day['cum売上'] / df_day['cum成約件数']
df_day['cum成約件数/組数'] = df_day['cum成約件数'] / df_day['cum組数']

with st.expander('df_day', expanded=False):
    st.write(df_day)

#**********************************可視化
st.write('cum売上/組数')
graph.make_line([df_day['cum売上/組数']],['cum売上/組数'], df_day.index ) 

st.write('cum売上/成約件数')
graph.make_line([df_day['cum売上/成約件数']],['cum売上/成約件数'], df_day.index ) 

st.write('cum成約件数/組数')
graph.make_line([df_day['cum成約件数/組数']],['cum成約件数/組数'], df_day.index )



# 標準化（平均0、分散1）
df_std = stats.zscore(df_day)

with st.expander('日にち/標準化済', expanded=False):
    st.write(df_std) #確認

st.write('相互相関コレログラム 組数/成約件数')
# 相互相関コレログラム（原系列）
fig0, ax = plt.subplots()
xcor_value = plt.xcorr(df_std['組数'], 
                       df_std['成約件数'],
                       detrend=mlab.detrend_none, 
                       maxlags=60)
st.pyplot(fig0)

# 相互相関の値
xcor_pd = pd.DataFrame(xcor_value[1],xcor_value[0])
xcor_pd.index.name = 'lag'
xcor_pd.columns = ['xcor']

#相互相関のソート
xcor2 = xcor_pd[xcor_pd.index >= 1]
xcor2 = xcor2.sort_values('xcor', ascending=False)

#max値の抽出
df_max = xcor2[xcor2['xcor'] == xcor2['xcor'].max()]
max_lag = list(df_max.index.values)[0]
max_cor = df_max.iat[0, 0]

st.metric(label='何日後に成約する傾向が高いか', value=f'{max_lag}日')
st.caption(f'相関係数: {max_cor}')

with st.expander('非単位/相関係数', expanded=False):
    st.write('一覧')
    st.write(xcor_pd) #確認
    st.write('一覧/ソート')
    st.write(xcor2) #確認


#***************************************************************************week
st.markdown('### 週')

df_week = df_all2.copy()
df_week = df_week.resample('W').sum()

df_week['cum売上/組数'] = df_week['cum売上'] / df_week['cum組数']
df_week['cum売上/成約件数'] = df_week['cum売上'] / df_week['cum成約件数']
df_week['cum成約件数/組数'] = df_week['cum成約件数'] / df_week['cum組数']

with st.expander('df_week', expanded=False):
    st.write(df_week)

#**********************************可視化
st.write('cum売上/組数')
graph.make_line([df_week['cum売上/組数']],['cum売上/組数'], df_week.index ) 

st.write('cum売上/成約件数')
graph.make_line([df_week['cum売上/成約件数']],['cum売上/成約件数'], df_week.index ) 

st.write('cum成約件数/組数')
graph.make_line([df_week['cum成約件数/組数']],['cum成約件数/組数'], df_week.index )


# 標準化（平均0、分散1）
df_stdw = stats.zscore(df_week)
with st.expander('週単位/標準化済', expanded=False):
    st.write(df_stdw) #確認

st.write('相互相関コレログラム 組数/成約件数')
# 相互相関コレログラム（原系列）
fig1, ax = plt.subplots()
xcor_value = ax.xcorr(df_std['組数'], 
                       df_std['成約件数'],
                       detrend=mlab.detrend_none, 
                       maxlags=12)
st.pyplot(fig1)

# 相互相関の値
xcor_pd = pd.DataFrame(xcor_value[1],xcor_value[0])
xcor_pd.index.name = 'lag'
xcor_pd.columns = ['xcor']
with st.expander('週単位/相関係数', expanded=False):
    st.write(xcor_pd) #確認

#相関係数の最大値の検出
xcor2 = xcor_pd[xcor_pd.index >= 1]
df_max = xcor2[xcor2['xcor'] == xcor2['xcor'].max()]
max_lag = list(df_max.index.values)[0]

max_cor = df_max.iat[0, 0]


st.metric(label='何週後に成約する傾向が高いか', value=f'{max_lag}週')
st.caption(f'相関係数: {max_cor}')

#****************その他
#st.write('売上合計')
sales_sum = df_all2['売上'].sum()
#st.write('組数合計')
kumi_sum = df_all2['組数'].sum()
#st.write('成約件数合計')
seiyaku_sum = df_all2['成約件数'].sum()
#st.write('成約単価')
seiyaku_tanka = sales_sum / seiyaku_sum
#st.write('成約率')
seiyaku_rate = seiyaku_sum / kumi_sum
#st.write('来店1組当たりの売上')
sales_per_kumi = sales_sum / kumi_sum
#st.write('成約に必要な組数')
seiyaku_needed = 1 / (seiyaku_sum / kumi_sum)

val_dict = {
    '売上合計': sales_sum,
    '組数合計': kumi_sum,
    '成約件数合計': seiyaku_sum,
    '成約単価': seiyaku_tanka,
    '成約率': seiyaku_rate,
    '来店1組当たりの売上': sales_per_kumi,
    '成約に必要な組数': seiyaku_needed
}

df_val = pd.DataFrame(val_dict, index=['数値']).T
st.table(df_val)

















