from firebase import firebase
from pandas import ExcelWriter
import pandas as pd
import numpy as np
import datetime
def fbse(Num):
  fire_base = firebase.FirebaseApplication('https://esp001-864dd.firebaseio.com', None)
  result = fire_base.get(Num, None)
  # fire_base.post('/001', 22)
  k=0
  miliseconds = []
  timestamps = []
  values = []
  for key, value in result.items():
    k+=1
    # if k>10: break
    ms = value['timestamp']
    timeS = datetime.datetime.fromtimestamp(ms/1000.0)
    timestamps.append(timeS)
    miliseconds.append(value['timestamp'])
    values.append(value['value'])
  data = np.vstack((np.array(miliseconds),np.array(timestamps))).T
  values = np.array(values)[:,np.newaxis]
  data = np.hstack((data,values))
  # print data.shape
  df = pd.DataFrame(data=data, columns=['miliseconds','timestamps','value'])
  # DF TO EXCEL
  df.sort_values(['miliseconds'], ascending=[True], inplace=True)
  df.reset_index(drop=True, inplace=True)
  # writer = ExcelWriter('22PythonExport.xlsx')
  # df.to_excel(writer,'Sheet1')
  # writer.save()
  # print df.head
  print '#', Num[1:], ' : ', len(result)

fbse('/001')
fbse('/002')
fbse('/003')