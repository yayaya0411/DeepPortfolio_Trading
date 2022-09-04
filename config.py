
# run.py
# slide = 30

# env.py
to_ratio = True
to_gray = False
buy_stock = 100

# scaler_file = 'Slope_2015_2020_scaler.sav'
scaler_file = 'dataset_2017_2020_scaler.sav'



# Transformer 
batch_size = 32
seq_len = 40 # slide window
slide =  seq_len # slide window

d_k = 256
d_v = 256
n_heads = 12
ff_dim = 256

## Portfolio Stock

# All Stock
ALL ={
    'name':'ALL',
    'portfolio':[
        # '3008',	#光電業
        # '9910',	#其他
        # '2317',	#其他電子業
        '2303', # 聯電
        '2454',	#半導體業
        '1326',	#塑膠工業
        '1101',	#水泥工業
        '2207',	#汽車工業
        '6505',	#油電燃氣業
        '2912',	#貿易百貨
        '4904',	#通信網路業
        '2882',	#金融保險業
        # '2002',	#鋼鐵工業
        '2327',	#電子零組件業
        # '2382',	#電腦及週邊設備業
        # '1216',	#食品工業
    ] 
}

Slope = {
    'name':'Slope',
    'portfolio':[
        # '2379', '2357', '2881', '2882', '1303', '2002', '1402', '2207', '4904','1216'
        '2379', '2357', '2881', '2882', '1303', '2002', '1402', '2207', '4904','1216'
    ]
}

# 0050 Y110Q3
TW0050 ={
    'name':'TW0050',
    'portfolio':[
        "1101", "1216", "1301", "1303", "1326", "1402", "1590", "2002", "2207", "2303", 
        "2308", "2317", "2324", "2327", "2330", "2357", "2379", "2382", "2395", #"2311",
        "2408", "2409", "2412", "2603", "2609", "2615", "2801", "2880", "2881", "2882", 
        "2884", "2885", "2886", "2887", "2891", "2892", "2912", "3008", "3034", "3045", 
        "4904", "4938", "5880", "6415", "6505", "8046", "8454", "9910"
    ] 
}
    
# Tech     
Tech ={
    'name':'tech',
    'portfolio':[
        '2317', # 鴻海
        '2409', # 友達
        '2330', # 台積電
        '2303', # 聯電
        '2301', # 光寶科
        '2308', # 台達電
        '2356', # 英業達
        '6770', # 力積電
        # '2353', # 宏碁
        # '3481', # 群創
        # '2312', # 金寶
        # '2324', # 仁寶
        # '2313', # 華通
        # '2327', # 國巨
    ]
}

# Financial
Fin ={
    'name':'fin',
    'portfolio':[
        '2891', # 中信金
        '2883', # 開發金
        '2885', # 元大金
        '2882', # 國泰金
        '2881', # 富邦金
        '2892', # 第一金
        '5880', # 合庫金
        '2884', # 玉山金
        '2887', # 台新金
        '2801', # 彰銀
        # '2890', # 永豐金
        # '2886', # 兆豐金
        # '2888', # 新光金
    ]
}

