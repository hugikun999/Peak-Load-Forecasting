# Peak Load Forecasting
* Overview

請根據台電歷史資料，預測未來七天的"電力尖峰負載"(MW)。

![](https://i.imgur.com/oiFgMtq.png)

* Evaluation

預測 2019/4/2 ~ 2019/4/8 的每日"電力尖峰負載"(MW)，作業將以 “尖峰負載預測值” 與 "尖峰負載實際數值"之 Root-Mean-Squared-Error (RMSE) 作為評估分數。

* Submission File Format

繳交檔案須包含 submission.csv，欄位為日期以及所預測之每日尖峰負載
```buildoutcfg
date,peak_load(MW)
20190402,22905
20190403,22797
20190404,23637
20190405,27722
20190406,28161
20190407,28739
20190408,26288
```
## Data
* [今日預估尖峰備轉容量率](https://www.taipower.com.tw/d006/loadGraph/loadGraph/load_reserve_.html)： 包含 "2017/1/1 ~ 今日" 之 瞬時尖峰負載(數值與尖峰負載相同)，備轉容量率與備轉容量

* [台灣電力公司_未來一週電力供需預測](https://data.gov.tw/dataset/33462)： 台電自身依據未來一週氣象預報資料及發電機組狀況得出之數值。

* [電力市場資料列表](https://tod.moea.gov.tw/#%7B%22allSearchKeyWord%22:%22%22,%22theme%22:%22%E9%9B%BB%E5%8A%9B%E5%B8%82%E5%A0%B4%E5%A4%9A%E5%85%83%E5%8C%96%22,%22subTopic%22:%22%E5%85%B6%E4%BB%96-%E7%94%A8%E9%9B%BB%E7%B5%B1%E8%A8%88%22,%22status%22:%22%E4%B8%8A%E6%9E%B6%22%7D)
* [今日電力資訊](https://www.taipower.com.tw/tc/page.aspx?mid=206)

* [一週天氣預報](http://opendata.cwb.gov.tw/dataset/forecast/F-A0010-001)
* [交通部中央氣象局](https://www.cwb.gov.tw/V7/climate/monthlyData/mD.htm)

## Environment setting
*   Package installation
```buildoutcfg
$ pip install -r requirements.txt
```
## tutorials
*   Inference

The default setting is to predict the loads after 2019/03/01.You can just pass the day you want to predict.

```buildoutcfg
python inference.py
python inference --start_day 20190101
```
*   Train your own model

After training you will get a `predict.params`.Use the params to be the model weights to predict. There are some parameters can be pass in.Please check out the `train.py`.
```buildoutcfg
python train.py
python inference --weights predict.params
```